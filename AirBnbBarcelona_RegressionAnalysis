# Predicting AirBnb prices with 87% accuracy

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Using listings.csv dataset
import numpy as np
import pandas as pd
import sklearn

# load dataset
df=pd.read_csv('/kaggle/input/barcelona/listings.csv')

# drop row without target variable
df=df.dropna(subset = ['price'])

#inspect columns
df.info()

print(len(df))

# drop surplus columns
columns_drop=['id','listing_url','scrape_id','last_scraped','source','name','description','neighborhood_overview', 
'picture_url','host_id','host_url','host_name','host_since','host_location','host_about', 'host_thumbnail_url', 
'host_picture_url','host_neighbourhood', 'host_verifications','calendar_updated', 'calendar_last_scraped', 'first_review',
'last_review', 'host_has_profile_pic', 'license', 'neighbourhood','review_scores_location','review_scores_checkin',                        
'review_scores_cleanliness', 'review_scores_accuracy', 'review_scores_value', 'review_scores_communication',                  
'reviews_per_month', 'review_scores_rating']
df=df.drop(columns_drop,axis=1)

df.info()

df.describe()

# removing outlier by row removal and imputing nan values in desired columns
df = df[df.bedrooms <15.0]
df=df[df.minimum_nights<501]
br_median=df.bedrooms.median()
df.bedrooms=df.bedrooms.fillna(br_median)

df.describe()

# construct wifi varibles from amenities
import ast

def extract_words(text):
    try:
        word_list = ast.literal_eval(text)
        return word_list
    except ValueError as e:
        print(f"Error parsing text: {e}")
        return []

flattened_list = [word for text in df.amenities for word in extract_words(text)]
from collections import Counter
element_counts = Counter(flattened_list)
df['wifi']=[1 if 'Wifi' in x else 0 for x in df.amenities]
df=df.drop('amenities', axis=1)

# checking for null values again
counts = df.isna().sum()
print(counts.sort_values(ascending=False))

# imputation and encoding of host response time variable
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

hrt_mode=df.host_response_time.mode()[0]
df.host_response_time=df.host_response_time.fillna(hrt_mode)

df.host_response_time=le.fit_transform(df.host_response_time)

# cleaning and imputation of host response and acceptance rate variables
df['host_response_rate'] = df['host_response_rate'].astype(str).str.rstrip('%').astype('float')
df.host_acceptance_rate = df.host_acceptance_rate.astype(str).str.rstrip('%').astype('float')

hrr_median=df.host_response_rate.mode()
har_median=df.host_acceptance_rate.mode()

df['host_response_rate']=df['host_response_rate'].fillna(hrr_median)
df['host_acceptance_rate']=df['host_acceptance_rate'].fillna(har_median)

# drop rows with missing values if any
df=df.dropna()
df.head()

df.describe()

len(df.select_dtypes(include='number').columns)

# boolean variables encoding
le=LabelEncoder()
df.host_is_superhost=le.fit_transform(df.host_is_superhost)
df.host_identity_verified=le.fit_transform(df.host_identity_verified)
df.has_availability=le.fit_transform(df.has_availability)
df.instant_bookable=le.fit_transform(df.instant_bookable)

# encoding of remaining categorical string variables
le=LabelEncoder()
df.neighbourhood_cleansed=le.fit_transform(df.neighbourhood_cleansed)
df.neighbourhood_group_cleansed=le.fit_transform(df.neighbourhood_group_cleansed)
df.property_type=le.fit_transform(df.property_type)
df.room_type=le.fit_transform(df.room_type)

# encoding of bathrooms_text variable
le=LabelEncoder()
df.bathrooms_text=le.fit_transform(df.bathrooms_text)

# cleaning and chaning type of target variable price
df.price=df.price.astype(str).str.replace(',','')
df.price = df.price.str.lstrip('$').astype('float')

# checking correlation with price variable
df.corr()

# plot correlation between variables
import matplotlib.pyplot as plt
import seaborn
c = df.corr() 
plt.figure(figsize=(20,20)) 
seaborn.heatmap(c, cmap='flare', mask = (np.abs(c) >= 0.5)) 
plt.show()

# checking correlation with price variable
print('Pearson')
print()
corr_matrix = df.corr('pearson')['price']
print(corr_matrix)
print()
print('Kendall')
print()
corr_matrix = df.corr('kendall')['price']
print(corr_matrix)
print()
print('Spearman')
print()
corr_matrix = df.corr('spearman')['price']
print(corr_matrix)

# splitting train and test set
from sklearn.model_selection import train_test_split
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

# applying decision tree regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=101)
dtr.fit(X_train, y_train)
y_pred_dtr = dtr.predict(X_test)
mse_dtr = mean_squared_error(y_test, y_pred_dtr)
r2_dtr = r2_score(y_test, y_pred_dtr)
mse_dtr, r2_dtr

# applying random forest regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
rf_mse, r2_rf
