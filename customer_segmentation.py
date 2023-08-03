"""
AUTHOR: BRANDEN ADEMS ANAK KIETHSON - AI04

"""

#%%
# 1. Import Packages and Libraries

import os
import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow
import missingno as msno
import numpy as np
from tensorflow import keras

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,StandardScaler
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
# %%
# 2. Path Setup

TRAIN_DATASET = os.path.join(os.getcwd(), 'Data', 'customer_segmentation.csv')
MODEL_SAVE = os.path.join(os.getcwd(),'Saved_Model','model.h5')
LOG_PATH = os.path.join(os.getcwd(),'Log')
# %%
# 3. Load Dataset
df = pd.read_csv(TRAIN_DATASET)
# %%
# 4. Data Inspection
df.info()
# %%
df.head()
# %%
# 5. Data Clearning
#%%
# Checking for Duplication
df[df.duplicated()]
# %%
# Checking for Missing Values or NaN
df.isna().sum()

#%%
msno.matrix(df)

#%%
# See the historgram graph if it is left skewed 
sns.histplot(df["Work_Experience"], kde=True)
plt.show()
# Right-Skewed (Fill in Median)
#%%
sns.histplot(df["Family_Size"], kde=True)
plt.show()
# Right-Skewed (Fill in Median)
#%%
sns.histplot(df["Var_1"], kde=True)
plt.show()
# %%
# Fill in the missing values with Forward Fill
df['Ever_Married'].fillna(method='ffill',inplace=True)
df['Graduated'].fillna(method='ffill', inplace=True)
df['Profession'].fillna(method='ffill', inplace=True)
df['Var_1'].fillna(method='ffill',inplace=True)

#%%
# Fill in the missing values with Median
df['Family_Size'] = df['Family_Size'].replace(np.NaN, df['Family_Size'].median())
df['Work_Experience'] = df['Work_Experience'].replace(np.NaN, df['Work_Experience'].median())
#%%
df.describe().T
#%%
# Looking for Outliers
numeric_columns = ['Age', 'Work_Experience', 'Family_Size']
df[numeric_columns].boxplot(figsize=(10,10))

# %%
# Checking for Duplicate
df.duplicated().sum()
# %%
# Data.info()
df.info()
# %%
# Check unique values and do one hot encoding
df.nunique()
# %%
# Strings is converted into integer with the used of label encoder
LE = LabelEncoder()
df['Gender'] = LE.fit_transform(df['Gender'])
df['Graduated'] = LE.fit_transform(df['Graduated'])
df['Ever_Married'] = LE.fit_transform(df['Ever_Married'])
df['Spending_Score'] = LE.fit_transform(df['Spending_Score'])
df['Family_Size'] = LE.fit_transform(df['Family_Size'])
df['Var_1'] = LE.fit_transform(df['Var_1'])
df['Segmentation'] = LE.fit_transform(df['Segmentation'])
df['Profession'] = LE.fit_transform(df['Profession'])


#%%
df['Age_group'] = pd.cut(df['Age'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90], labels=['10-20','21-30','31-40', '41-50', '51-60', '61-70', '71-80','81-90'])

#%%
df['Age_group'] = LE.fit_transform(df['Age_group'])


#%%
df.info()
# %%
# 6. Feature selection
X = df.drop(labels =['Segmentation'], axis=1)
y = df['Segmentation']
# %%
mms = MinMaxScaler()
rf = RandomForestClassifier()
X = mms.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#%%
# Tensorboard call backs
log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='loss', patience=10)
#%%
#Data Preprocessing
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#%%
optimizer = optimizers.Adam(learning_rate=0.001)


#%%
rf.fit(X_train, y_train)

#%%
pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
# %%
model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation ='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(4, activation ='softmax'))

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy')
model.summary()

kmeans = KMeans(n_clusters=4)
kmeans.fit(X_train)

hist = model.fit(X_train, y_train, epochs= 250, batch_size=350, 
                 validation_data=(X_test, y_test), callbacks=[tensorboard_callback,early_stopping_callback])
# %%

model.evaluate(X_test,y_test)
# %%
keras.models.save_model(model,MODEL_SAVE)
# %%
keras.utils.plot_model(model)
# %%
