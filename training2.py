#!/usr/bin/env python
# coding: utf-8

# In[17]:


#DATABASE


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from datetime import timedelta
from sqlalchemy import create_engine
from datetime import datetime, date

db_user = 'postgres'          
db_password = 'UWP12345!' 
db_host = '35.246.149.161'         
db_port = '5432'             
db_name = 'postgres'     

connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)
df = pd.read_sql_table("streckentabelle", con=engine)


# In[18]:


#PREPROCESSING

#TRAIN NUMBER/REIHENFOLGE
df[["train_number", "reihenfolge_from"]] = df["train_reihenfolge_from"].str.split("-", expand=True)

df["train_number"] = df["train_number"].astype(int)
df["reihenfolge_from"] = df["reihenfolge_from"].astype(int)


#DELAY <0
df["arrival_delay_to"] = df["arrival_delay_to"].apply(lambda x: max(x, 0))



#SOLLDAUER

df["planned_departure_from_dt"] = df["planned_departure_from"].apply(lambda t: datetime.combine(date.today(), t))
df["planned_departure_to_dt"] = df["planned_departure_to"].apply(lambda t: datetime.combine(date.today(), t))
df.loc[df["planned_departure_to_dt"] < df["planned_departure_from_dt"], "planned_departure_to_dt"] += pd.Timedelta(days=1)
df["soll_dauer"] = (df["planned_departure_to_dt"] - df["planned_departure_from_dt"]).dt.total_seconds() / 60
df = df[(df["soll_dauer"] >= 0) & (df["soll_dauer"] <= 1000)]

#DAY OF WEEK
df["planned_arrival_date_from"] = pd.to_datetime(df["planned_arrival_date_from"])

#Monday = 0, Sunday = 6
df["day_of_week"] = df["planned_arrival_date_from"].dt.dayofweek

#ARRIVAL DELAY TO DROP
df = df.dropna(subset=["arrival_delay_to"])

#PLANNED HOUR
df["planned_departure_from_dt"] = df["planned_departure_from"].apply(lambda t: datetime.combine(date.today(), t))
df["planned_hour_from"] = df["planned_departure_from_dt"].dt.hour

####### NOT IN RANDOM FOREST

#WEEKEND  1=weekend
df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

#HOLIDAYS 1=Feiertag
import holidays
de_holidays = holidays.Germany(years=df["planned_arrival_date_from"].dt.year.unique())
df["is_holiday"] = df["planned_arrival_date_from"].dt.date.apply(lambda x: 1 if x in de_holidays else 0)

#PEAK TIME
df["is_peak_time"] = df["planned_hour_from"].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)


# In[19]:


#ENCODER

le_from = LabelEncoder()
le_to = LabelEncoder()
le_type = LabelEncoder()

df["station_name_from"] = le_from.fit_transform(df["station_name_from"])
df["station_name_to"] = le_to.fit_transform(df["station_name_to"])
df["train_classification"] = le_type.fit_transform(df["train_classification"])


# In[20]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

#TRAIN/VALIDATION DATASET
df = df.sort_values("planned_arrival_date_from")
unique_days = df["planned_arrival_date_from"].dt.date.unique()
n_days = len(unique_days)
n_train_days = int(n_days * 0.8)
train_days = unique_days[:n_train_days]
val_days = unique_days[n_train_days:]

# SPLIT
df_train = df[df["planned_arrival_date_from"].dt.date.isin(train_days)].copy()
df_val = df[df["planned_arrival_date_from"].dt.date.isin(val_days)].copy()

# Features for Random Forest
feature_cols = ["station_name_from", "station_name_to", "train_number", "day_of_week","planned_hour_from", "soll_dauer", "reihenfolge_from","station_avg_delay_7_30","train_avg_delay_7_30"]
target_col = "arrival_delay_to"

df_train.rename(columns=lambda x: str(x), inplace=True)
df_val.rename(columns=lambda x: str(x), inplace=True)

# MODEL
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df_train[feature_cols], df_train[target_col])



# In[1]:


#WAHRSCHEINLICHKEIT
from sklearn.ensemble import RandomForestClassifier


def classify_delay(delay):
    if delay == 0:
        return 0  # On time
    elif 1 <= delay < 10:
        return 1
    elif 10 <= delay < 20:
        return 2
    elif 20 <= delay < 30:
        return 3
    else:  # 30+
        return 4


df_class = df.copy()
df_class["delay_class"] = df_class["arrival_delay_to"].apply(classify_delay)



# 2. Train/Val Split (classification için)
df_train_cls = df_class[df_class["planned_arrival_date_from"].dt.date.isin(train_days)].copy()
df_val_cls = df_class[df_class["planned_arrival_date_from"].dt.date.isin(val_days)].copy()

# 3. Quoted name fix
df_train_cls.rename(columns=lambda x: str(x), inplace=True)
df_val_cls.rename(columns=lambda x: str(x), inplace=True)

# 4. Model kur
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(df_train_cls[feature_cols], df_train_cls["delay_class"])



# In[ ]:


import pickle

# Modeli kaydet
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Label Encoders'ları da kaydet (prediction sırasında lazım olacak)
with open('encoders.pkl', 'wb') as f:
    pickle.dump({
        'le_from': le_from,
        'le_to': le_to,
        'le_type': le_type
    }, f)

print("Model and encoders saved.")


# In[ ]:


# Classification modelini kaydet
with open('classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)


# In[ ]:


class_mapping = {
    0: "On time",
    1: "1-9 min",
    2: "10-19 min",
    3: "20-29 min",
    4: "30+ min"
}


# Class mapping'i kaydet
with open('class_mapping.pkl', 'wb') as f:
    pickle.dump(class_mapping, f)

