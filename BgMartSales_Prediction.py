import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def Outlet_size_replace(x):
    return size_mode[x]

def itemIdendify(x):
    return x[:2]
  
def outIdentify(x):
    return(x[-2:])

def createOthers(x):
    if x in item_count_less:
        return 'Others'
    else:
        return x
def predict_sales(Item_Weight,Item_Visibility,Item_MRP,Outlet_Identifier,Years,Item_Fat_Content,
                  Item_Type,Item_Identifier,Outlet_Size,Outlet_Location_Type,Outlet_Type):
    uv = np.zeros(xtrain.shape[1])
    
    uv[0] = Item_Weight
    uv[1] = Item_Visibility
    uv[2] = Item_MRP
    uv[3] = Outlet_Identifier
    uv[4] = Years

    uv[np.where(xtrain.columns == Item_Fat_Content)[0][0]] = 1
    uv[np.where(xtrain.columns == Item_Type)[0][0]] = 1
    uv[np.where(xtrain.columns == Item_Identifier)[0][0]] = 1
    uv[np.where(xtrain.columns == Outlet_Size)[0][0]] = 1
    uv[np.where(xtrain.columns == Outlet_Location_Type)[0][0]] = 1
    uv[np.where(xtrain.columns == Outlet_Type)[0][0]] = 1

    return lmodel.predict([uv])

    
  
  
# importing the dataset
df1 = pd.read_csv('Big_Mart.csv')
# Data Pre Processing
mean_weight = pd.DataFrame(df1.groupby('Item_Identifier')['Item_Weight'].mean())
missing = df1['Item_Weight'].isnull()
for i,item in enumerate(df1['Item_Identifier']):
    if (missing[i] == True):
        if item in mean_weight.index:
            df1['Item_Weight'][i] = mean_weight.loc[item][0]
        else:
            df1['Item_Weight'][i] = np.mean(df1['Item_Weight'])
                  
miss_size = df1['Outlet_Size'].isnull()

#pivot table
size_mode = df1.pivot_table(values = 'Outlet_Size' , columns = 'Outlet_Type', 
                            aggfunc = (lambda x:x.mode()))
mvd = df1.loc[miss_size]
nvd = df1.loc[miss_size != True]
mvd['Outlet_Size'] = mvd['Outlet_Type'].apply(Outlet_size_replace)
df2 = pd.concat((mvd,nvd),ignore_index=True)
df2['Item_Weight'] = df2['Item_Weight'].fillna(np.mean(df2['Item_Weight']))
df2['Item_Visibility'] = df2['Item_Visibility'].replace(0,df2['Item_Visibility'].mean())
df2['Item_Fat_Content'] = df2['Item_Fat_Content'].replace({'LF':'Low Fat', 
                                                           'low fat':'Low Fat', 
                                                           'reg':'Regular'})

df2['Item_Identifier'] = df2['Item_Identifier'].apply(itemIdendify)
df2['Item_Identifier'] = df2['Item_Identifier'].replace({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drink'})
df2.loc[df2['Item_Identifier'] == 'Non-Consumable' , 'Item_Fat_Content'] = 'Non-Edible'
df2['Outlet_Identifier'] = df2['Outlet_Identifier'].apply(outIdentify)
df2['Outlet_Identifier'] = df2['Outlet_Identifier'].astype(int)
df2['Years'] = 2014 - df2['Outlet_Establishment_Year']

categorical = []
for i in df2.dtypes.index:
    if (df2[i].dtype == 'object'):
        categorical.append(i)
        
item_count = df2['Item_Type'].value_counts()
item_count_less = item_count[item_count <= 251]      
df2['Item_Type'] = df2['Item_Type'].apply(createOthers)
df2.head()
df3 = df2.drop('Outlet_Establishment_Year',axis=1)
####One Hot Encoding
# df3['Item_Identifier'].unique()
#Food -           1 0 0
#Non-Consumable - 0 1 0
#Drink    -       0 0 1

df_item_identifier = pd.get_dummies(df3['Item_Identifier'])
df_item_fat_content = pd.get_dummies(df3['Item_Fat_Content'])
df_item_type = pd.get_dummies(df3['Item_Type'])
df_outlet_size = pd.get_dummies(df3['Outlet_Size'])
df_outlet_location_type = pd.get_dummies(df3['Outlet_Location_Type'])
df_outlet_type = pd.get_dummies(df3['Outlet_Type'])
df4 = pd.concat((df3,df_item_identifier,df_item_fat_content,df_item_type,
                 df_outlet_size,df_outlet_location_type,df_outlet_type),axis=1)
df5 = df4.drop(categorical,axis=1)
X = df5.drop('Item_Outlet_Sales',axis=1)
Y = df5['Item_Outlet_Sales']
xtrain,xtest,ytrain,ytest = train_test_split(X,Y)

"""#### Train The Dataset"""
lmodel = LinearRegression()
lmodel.fit(xtrain,ytrain)

"""#### Test The Dataset """
ytrain_pred = lmodel.predict(xtrain)
ytest_pred = lmodel.predict(xtest)
print("mean_absolute_error of ytrain,ytrain_pred:",mean_absolute_error(ytrain,ytrain_pred))
print("mean_absolute_error of ytest,ytest_pred:",mean_absolute_error(ytest,ytest_pred))
print("Accuracy while predicting xtrain data:",lmodel.score(xtrain,ytrain))
print("Accuracy while predicting xtest data:",lmodel.score(xtest,ytest))



#### User Can Input Here

# predict_sales(Item_Weight,Item_Visibility,Item_MRP,Outlet_Identifier,Years,Item_Fat_Content,Item_Type,Item_Identifier,Outlet_Size,Outlet_Location_Type,Outlet_Type)
print(predict_sales(9.30,0.03,250,17,10,
                    'Low Fat','Dairy','Food','Medium','Tier 1','Supermarket Type2'))
