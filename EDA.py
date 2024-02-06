import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Dataset-SA.csv')

## Basic Preprocessing
df.info()

## Check null values
print(df.isnull().sum())

## Check dataframe shape
df.shape

## Check duplicated values
df.duplicated().sum()


## First review
print(df['product_name'][0])


## Preprocessing Next Steps:
    # Extract Name from Product
    # Check strings in numeric column and dropping them
    # See how product price and Rating is related to sentiment
    # Convert columns to correct format
    
    
# 1. Extract Name from Product
df['product_name'] = df['product_name'].apply(lambda i:i.split("?")[0].strip())

print(df['product_name'].nunique())


# 2. Check strings in numeric column and dropping them

lst =[]
for i in df['product_price']:
    if re.search('[a-zA-Z]', i):
        lst.append((df['product_price']==i).argmax())
        
for i in lst:
    df= df.drop(index=i)        
    
## 3. Plotting how product price is spread

df['product_price'] = pd.to_numeric(df['product_price'])

sns.histplot(data=df,x='product_price',bins=50)

sns.kdeplot(data=df,x='product_price')

## 3. Count of different types of Sentiments
sns.countplot(data=df,x='Sentiment')

## 4. Product price vs Sentiment
round(df.groupby('Sentiment')['product_price'].mean(),2).plot(kind='bar')
plt.show()

## 4. Product rating vs Sentiment
df['Rate'] = pd.to_numeric(df['Rate'])
round(df.groupby('Sentiment')['Rate'].mean(),2).plot(kind='bar')
plt.show()