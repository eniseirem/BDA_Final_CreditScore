import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


missing_values = ["n/a", "na", "--", " ?","?"," "]

data = pd.read_csv('data/Credit Score Dataset.txt', delimiter="\t", na_values=missing_values)

print(data.tail)
print(data.shape)

print(np.where(pd.isnull(data)))
print(data['CRED'].isnull().sum()) #for empty targets i'll drop them to not effect the dataset, to prevent any bias

data.dropna(subset=['CRED'], inplace=True)
print(data.shape) #we just lost 33 rows.
for col in data.columns:
    print(col)

#ok we have a columns for females but we have genders column as well. so we don't need females column.
print(data["GENDER"].unique()) #just F, M so we don't need to check for anyother identification
data.drop(columns=['female'], inplace=True)
#we also have restype (I'm assuming its resident type)
print(data["RESTYPE"].unique()) #ok it has also renter in it. But we have a column says renter too. We don't need that as well

print(data["renter"].unique())
print(data["HOME"].unique())
print(data["CONDO"].unique())
print(data["COOP"].unique())

#so we have all of them as separated... I guess this dataset used with some encoding algorithm but I would like to have clean dataset before
#I use any encoding.
data.drop(columns=['HOME','COOP','renter','CONDO'], inplace=True)

#I decided to get rid of the all columns that I don't know the meaning since I don't have information about them I can't judge or understand if the column is important.
#So, I will be working with, gender, age, restype, income, emp_status.

cleandf = data.drop(columns=['MS','HEQ','DEPC','MOB', 'MILEAGE', 'RES_STA', 'DELINQ', 'NUMTR', 'MRTGI', 'MFDU', 'resp', 'emp1', 'emp2', 'msn','cuscode'])

for col in cleandf.columns:
    print(col)

#%% change our columns datatypes
#we have restype, gender and emp_sta to change first

dist_named = cleandf.copy()

print(cleandf["EMP_STA"].unique()) #It can be years of employment. So we will group them

col_name='EMP_STA'
conditions_exist = [
    data[col_name].isin(['1,2']),
    data[col_name].isin(['0']),
    data[col_name].isin(['3+']),
]
result = [1, 0, 2]
cleandf['EMP_STA']=np.select(conditions_exist,result)

 #we have two genders so;
col_name='GENDER'
conditions_exist = [
    data[col_name].isin(['M']),
    data[col_name].isin(['F']),
]
result = [0, 1]
cleandf['GENDER']=np.select(conditions_exist,result)

#we have 4 type of resident so;
col_name='RESTYPE'
conditions_exist = [
    data[col_name].isin(['RENTER']),
    data[col_name].isin(['CONDO']),
    data[col_name].isin(['COOP']),
    data[col_name].isin(['HOME']),
]
result = [3,2,1,0]
cleandf['RESTYPE']=np.select(conditions_exist,result)

#%% variable and cred correlations

import seaborn as sns
import matplotlib.pyplot as plt

corr = cleandf.corr()
hh = sns.heatmap(corr, cmap="YlGnBu")
plt.show()

st  = pd.DataFrame(corr['CRED'], index=corr.index)
print(st.sort_values('CRED'))

#%% Train-Test for cleaned data
from sklearn.model_selection import train_test_split
y= cleandf["CRED"]
X = cleandf.drop(columns=["CRED"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
