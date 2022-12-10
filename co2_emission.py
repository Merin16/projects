#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[ ]:


pip intall django


# In[ ]:


import numpy as np                  # linear algebra
import pandas as pd                 # data processing
import seaborn as sns               #helps you explore and understand your data
import matplotlib.pyplot as plt     #additional plot types and provides some visual improvements to some plots and graphs.


# In[ ]:


import warnings   # displaying the warning message
warnings.filterwarnings("ignore")  #Ignore all warnings


# # Read Data

# In[ ]:


#Load the data then show the first 5 rows
data= pd.read_csv(r"C:\Users\Downloads\CO2 Emissions_Canada (1).csv")
data.head(5)


# # Exploratory Data Analysis

# # Inspecting data

# In[ ]:


#show the info of the data
data.info()


# In[ ]:


#show description of the data
data.describe()


# In[ ]:


#to show the no: of rows and columns
data.shape


# In[ ]:


#to show the column labels
data.columns


# In[ ]:


#checking for missing values
data.isnull().values.any()


# In[ ]:


#to list down missing data in a data set
data.isnull().sum()


# In[ ]:


#checking for duplicate values
data.duplicated().sum()


# In[ ]:


# Drop Duplicated values
data= data.drop_duplicates()
data.shape


# In[ ]:


duplicate=data.duplicated().sum()
print('There are {} duplicated rows in the data'.format(duplicate))


# In[ ]:


#to plot Mean Co2 Emission Vs different features


# In[ ]:


def explore_cat_feature(feature):
    group= data.groupby(feature).mean()
    plt.figure(figsize=[17,5])
    plots = group['CO2 Emissions(g/km)'].sort_values().plot(kind = 'bar', fontsize=15)
    plt.xlabel(feature, fontsize=15);
    plt.ylabel('Mean Co2 Emission', fontsize=15);
    plt.title("Mean Co2 Emission according to {} feature\n".format(feature), fontsize=20)


# In[ ]:


#to plot Mean Co2 Emission Vs Make
for feature in ['Make']:
    explore_cat_feature(feature)


# In[ ]:


#to plot Mean Co2 Emission Vs Vehicle Class
for feature in ['Vehicle Class']:
    explore_cat_feature(feature)


# In[ ]:


#to plot Mean Co2 Emission Vs Engine Size(L)
for feature in ['Engine Size(L)']:
    explore_cat_feature(feature)


# In[ ]:


#to plot Mean Co2 Emission Vs Cylinders
for feature in ['Cylinders']:
    explore_cat_feature(feature)


# In[ ]:


#count the number of records for every combination of unique values for every column
data['Transmission'].value_counts()


# In[ ]:


#to replace a string with another string
data["Transmission"]= data["Transmission"].replace('AV', 'AV3')


# In[ ]:


#to plot Mean Co2 Emission Vs Transmission
for feature in ['Transmission']:
    explore_cat_feature(feature)


# In[ ]:


#splitting the column
data["Gears"]= data['Transmission'].str[-1]
data.head(10)


# In[ ]:


data['Transmission type'] = data['Transmission'].apply(lambda x: x[0])
data.head(5)


# In[ ]:


#returns the count of all unique values in transmission type column
data['Transmission type'].value_counts()


# In[ ]:


data['Gears'].value_counts()


# In[ ]:


#replacing a value with another value
data["Gears"]=data["Gears"].replace('V',3)
data['Gears']=data['Gears'].replace(to_replace='0',value='10')


# In[ ]:


data['Gears'].value_counts()


# In[ ]:


#importing libraries
import plotly.express as px                  #create entire figures at once
import plotly.graph_objs as go               #to create a plot represented by Figure object which represents Figure class
from plotly.subplots import make_subplots    #operates on a variety of types of data and produces easy-to-style figures


# In[ ]:


#to show the top 25 company
data_Make=data['Make'].value_counts().reset_index().rename(columns={'index':'Make','Make':'Count'})[0:25]
data_Make
fig = go.Figure(go.Bar(
    x=data_Make['Make'],y=data_Make['Count'],
    marker={'color': data_Make['Count'], 
    'colorscale': 'Viridis'},  
    text=data_Make['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Top 25 Company',xaxis_title="Company ",yaxis_title="Number Of Vehicles ",title_x=0.5)
fig.show()


# In[ ]:


#top 10 ford model
data_ford=data[data["Make"]=="FORD"]
data_ford_model=data_ford["Model"].value_counts().reset_index().rename(columns={'index':'Model','Model':'Count'})[0:10]
fig = go.Figure(go.Bar(
    x=data_ford_model['Model'],y=data_ford_model['Count'],
    marker={'color': data_ford_model['Count'], 
    'colorscale': 'Viridis'},  
    text=data_ford_model['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Top 10 Ford Model',xaxis_title="Model ",yaxis_title="Number Of Vehicles ",title_x=0.5)
fig.show()


# In[ ]:


#renaming the labels of fuel type column for a better understanding
data["Fuel Type"]=data["Fuel Type"].replace('X','Regular gasoline')
data["Fuel Type"]=data["Fuel Type"].replace('Z','Premium gasoline')
data["Fuel Type"]=data["Fuel Type"].replace('D','Diesel')
data["Fuel Type"]=data["Fuel Type"].replace('E','Ethanol (E85)')
data["Fuel Type"]=data["Fuel Type"].replace('N','Natural gas')
data.head(4)


# In[ ]:


data['Fuel Type'].value_counts()


# In[ ]:


#to get a Series containing counts of unique values
data['Make'].value_counts()


# In[ ]:


data['Model'].value_counts()


# In[ ]:


#moving the columns having numerical values in num_features
num_features = [feature for feature in data.columns if data[feature].dtype != 'O']
data[num_features].head()


# In[ ]:


#from num_featues moving the columns having length of unique values less than 25(having discrete values)
discrete_features = [feature for feature in num_features if len(data[feature].unique()) < 25]
print(discrete_features)


# In[ ]:


#num_features excluding discrete_features ie,having many values
continuous_features = [feature for feature in num_features if feature not in discrete_features]
print(continuous_features)


# In[ ]:


#columns having categorical values ie,columns excluding num_features
cat_features = [feature for feature in data.columns if feature not in num_features]
data[cat_features].head()


# In[ ]:


#value count for cat_features
for feature in cat_features:
    print('{}: {} categories'.format(feature, len(data[feature].unique())))


# In[ ]:


#removing model from cat_features since it is having 2053 categories
cat_features.remove('Model')


# # Feature Engineering

# In[ ]:


#a new object will be created with a copy of the calling object’s data and indices
# Modifications to the data or indices of the copy will not be reflected in the original object 
dataset = data.copy()
dataset.head()


# In[ ]:


#taking log for the numerical values in continuous_feature
for feature in continuous_features:
    dataset[feature] = np.log(dataset[feature])


# In[ ]:


#encoding cat_features
#Converting Categorical features to Numerical features
for feature in cat_features:
    ordinal_labels = dataset.groupby(feature)['CO2 Emissions(g/km)'].max().sort_values().index
    ordinal_dict = {k:i for i, k in enumerate(ordinal_labels, 0)}
    dataset[feature] = data[feature].map(ordinal_dict)


# In[ ]:


#dropping column model from dataset
dataset.drop(['Model'], axis = 1, inplace = True)


# In[ ]:


#first 10 rows of dataset
dataset.head(10)


# In[ ]:


dataset['Transmission type'].value_counts()


# In[ ]:


dataset.to_csv('processed_data.csv', index = False)


# In[ ]:


df = pd.read_csv('processed_data.csv')
df.head()


# In[ ]:


x=dataset.corr()
sns.heatmap(x, annot = True, cmap = plt.cm.CMRmap_r)
sns.set(rc = {'figure.figsize':(15,15)})
plt.show()


# In[ ]:


fig = px.scatter(data, x="Engine Size(L)", y="Cylinders")
fig.update_layout(title_text='Cylinders vs Engine Size ',xaxis_title=" Engine Size (L)",yaxis_title="Cylinders",title_x=0.5)
fig.show()


# In[ ]:


fig = px.histogram(data, x="Cylinders")
fig.update_layout(title_text='Distribution Of Cylinders  ',xaxis_title=" Cylinders ",yaxis_title="Number Of Vehicles ",title_x=0.5)
fig.show()


# In[ ]:


df_Transmission=data['Transmission type'].value_counts().reset_index().rename(columns={'index':'Transmission type','Transmission type':'Count'})
df_Transmission
fig = px.pie(df_Transmission, values='Count', names='Transmission type')

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')
fig.update_layout(title="Transmission type Distribution",title_x=0.5)
fig.show()


# In[ ]:


data_Gears=data['Gears'].value_counts().reset_index().rename(columns={'index':'Gears','Gears':'Count'})
data_Gears
fig = px.pie(data_Gears, values='Count', names='Gears')

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')
fig.update_layout(title="Gears Distribution",title_x=0.5)
fig.show()


# In[ ]:


data_Fuel_Type=data['Fuel Type'].value_counts().reset_index().rename(columns={'index':'Fuel_Type','Fuel Type':'Count'})

fig = go.Figure(go.Bar(
    x=data_Fuel_Type['Fuel_Type'],y=data_Fuel_Type['Count'],
    marker={'color': data_Fuel_Type['Count'], 
    'colorscale': 'Viridis'},  
    text=data_Fuel_Type['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Fuel Type Distribution ',xaxis_title="Fuel Type  ",yaxis_title="Number Of Vehicles ",title_x=0.5)
fig.show()


# In[ ]:


fig = px.scatter(data, x="Fuel Type", y="CO2 Emissions(g/km)")
fig.update_layout(title_text='Fuel Type vs CO2 Emissions(g/km) ',xaxis_title="Fuel Type",yaxis_title="CO2 Emissions(g/km)",title_x=0.5)
fig.show()


# In[ ]:


fig = px.scatter(data, x="Fuel Type", y="Fuel Consumption Comb (L/100 km)")
fig.update_layout(title_text='Fuel Type vs Fuel Consumption ',xaxis_title="Fuel Type",yaxis_title="Fuel Consumption Comb (L/100 km)",title_x=0.5)
fig.show()


# In[ ]:


data.groupby(['CO2 Emissions(g/km)','Fuel Type'])['Fuel Type'].count()


# In[ ]:


data.groupby(['CO2 Emissions(g/km)','Fuel Consumption Comb (L/100 km)'])['Fuel Consumption Comb (L/100 km)'].count()


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=data,x='Fuel Consumption Comb (L/100 km)',y='CO2 Emissions(g/km)',hue='Fuel Type',size='CO2 Emissions(g/km)')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=data,x='CO2 Emissions(g/km)',y='Cylinders',hue='Fuel Type',size='CO2 Emissions(g/km)')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(data=data,x='Fuel Consumption Comb (L/100 km)',y='Cylinders',hue='Fuel Type',size='CO2 Emissions(g/km)')
plt.show()


# In[ ]:


sns.barplot(x='CO2 Emissions(g/km)',y='Make',data=data)
plt.show()


# In[ ]:


sns.barplot(x='CO2 Emissions(g/km)',y='Vehicle Class',data=data)
plt.show()


# In[ ]:


sns.barplot(x='CO2 Emissions(g/km)',y='Transmission type',data=data)
plt.show()


# In[ ]:


sns.barplot(x='CO2 Emissions(g/km)',y='Fuel Type',data=data)
plt.show()


# In[ ]:


data['Make_Type'] = data['Make'].replace(['BUGATTI', 'PORSCHE', 'MASERATI', 'ASTON MARTIN', 'LAMBORGHINI','JAGUAR','SRT'],'Sports')
data['Make_Type'] = data['Make_Type'].replace(['ALFA ROMEO', 'AUDI', 'BMW', 'BUICK','CADILLAC', 'CHRYSLER', 'DODGE', 'GMC','INFINITI', 'JEEP', 'LAND ROVER', 'LEXUS', 'MERCEDES-BENZ','MINI', 'SMART', 'VOLVO'],'Premium')
data['Make_Type'] = data['Make_Type'].replace(['ACURA', 'BENTLEY', 'LINCOLN', 'ROLLS-ROYCE','GENESIS'],'Luxury')
data['Make_Type'] = data['Make_Type'].replace(['CHEVROLET', 'FIAT', 'FORD', 'KIA','HONDA', 'HYUNDAI', 'MAZDA', 'MITSUBISHI','NISSAN', 'RAM', 'SCION', 'SUBARU', 'TOYOTA','VOLKSWAGEN'],'General')
print(data['Make_Type'].unique())
print(data['Make_Type'].value_counts())


# In[ ]:


data = data.drop(['Make'], axis=1)
data.head()


# In[ ]:


sns.barplot(x='CO2 Emissions(g/km)',y='Make_Type',data=data)
plt.show()


# In[ ]:


data['Vehicle_Class_Type'] = data['Vehicle Class'].replace(['COMPACT', 'MINICOMPACT', 'SUBCOMPACT'],'Hatchback')
data['Vehicle_Class_Type'] = data['Vehicle_Class_Type'].replace(['MID-SIZE', 'TWO-SEATER', 'FULL-SIZE', 'STATION WAGON - SMALL','STATION WAGON - MID-SIZE'],'Sedan')
data['Vehicle_Class_Type'] = data['Vehicle_Class_Type'].replace(['SUV - SMALL', 'SUV - STANDARD', 'MINIVAN'],'SUV')
data['Vehicle_Class_Type'] = data['Vehicle_Class_Type'].replace(['VAN - CARGO', 'VAN - PASSENGER', 'PICKUP TRUCK - STANDARD', 'SPECIAL PURPOSE VEHICLE','PICKUP TRUCK - SMALL'],'Truck')
print(data['Vehicle_Class_Type'].unique())
print(data['Vehicle_Class_Type'].value_counts())


# In[ ]:


data = data.drop(['Vehicle Class'], axis=1)
data.head()


# In[ ]:


sns.barplot(x='CO2 Emissions(g/km)',y='Vehicle_Class_Type',data=data)
plt.show()


# In[ ]:


sns.factorplot('Fuel Consumption Comb (L/100 km)','CO2 Emissions(g/km)',col='Fuel Type',data=data)
plt.show()


# In[ ]:


for feature in continuous_features:
    if feature != 'CO2 Emissions(g/km)':
        dataset = data.copy()
        
        dataset[feature] = np.log(dataset[feature])
        
        plt.scatter(data[feature], np.log(dataset['CO2 Emissions(g/km)']))
        plt.xlabel(feature)
        plt.ylabel('emission')
        plt.show()


# # Feature Selection

# In[ ]:


data.corr()[["Engine Size(L)","Fuel Consumption Comb (L/100 km)","CO2 Emissions(g/km)"]]


# In[ ]:


X = df.drop('CO2 Emissions(g/km)', axis = 1)
y = df["CO2 Emissions(g/km)"]


# In[ ]:


cor = X.corr()
sns.heatmap(cor, annot = True, cmap = 'tab20b')
sns.set(rc = {'figure.figsize':(15,8)})
plt.show()


# # Modelling

# In[ ]:


df = data.drop(['Make_Type','Model','Vehicle_Class_Type','Fuel Consumption City (L/100 km)', 'Transmission', 'Fuel Type', 'Fuel Consumption Hwy (L/100 km)','Fuel Consumption Comb (mpg)','Transmission type','Gears'],axis=1)
df.head()


# In[ ]:


x = df.drop(['CO2 Emissions(g/km)'], axis= 1)
y = df["CO2 Emissions(g/km)"]


# # Feature scaling

# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[ ]:


df_scala = data[['Engine Size(L)','Fuel Consumption Comb (L/100 km)', 'Cylinders', 'CO2 Emissions(g/km)']]
df_scala = pd.DataFrame(scaler.fit_transform(df_scala), columns = df_scala.columns)
scaler.fit(x_train)


# In[ ]:


x = df_scala[['Engine Size(L)','Fuel Consumption Comb (L/100 km)', 'Cylinders']]
y = df_scala['CO2 Emissions(g/km)']


# In[ ]:


x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# # Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
import sklearn.metrics as metrics
lr = LinearRegression().fit(x_train_scaled, y_train)
y_pred = lr.predict(x_test_scaled)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


lr_score = r2_score(y_test, y_pred)
lr_rmse = mean_squared_error(y_test, y_pred, squared = False)


# In[ ]:


print("R2 Score : ", lr_score)
print("RMSE : ", lr_rmse)
print('Training Accuracy: ', lr.score(x_train_scaled, y_train)*100)
print('Testing Accuracy: ', lr.score(x_test_scaled, y_test)*100)


# # Lasso Regression 

# In[ ]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.005).fit(x_train, y_train)
y_pred = lasso.predict(x_test)


# In[ ]:


lasso_score=r2_score(y_test,y_pred)
lasso_rmse=np.sqrt(mean_squared_error(y_test, y_pred))


# In[ ]:


print("R2 Score : ", lasso_score)
print("RMSE : ", lasso_rmse)
print('Training Accuracy: ', lasso.score(x_train, y_train)*100)
print('Testing Accuracy: ', lasso.score(x_test, y_test)*100)


# # Ridge regression

# In[ ]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 1)
ridge.fit(x_train, y_train)


# In[ ]:


y_pred=ridge.predict(x_test)
ridge_score=r2_score(y_pred, y_test)
ridge_rmse=np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


print("R2 Score : ", ridge_score)
print("RMSE : ", ridge_rmse)
print('Training Accuracy: ', ridge.score(x_train, y_train)*100)
print('Testing Accuracy: ', ridge.score(x_test, y_test)*100)


# # Simple Vector Machine

# In[ ]:


from sklearn.svm import LinearSVR
svm=LinearSVR()
svm.fit(x_train,y_train)


# In[ ]:


y_pred=svm.predict(x_test)
svm_score=r2_score(y_pred,y_test)
svm_rmse=np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


print("R2 Score : ", svm_score)
print("RMSE : ", svm_rmse)
print('Training Accuracy: ', svm.score(x_train, y_train)*100)
print('Testing Accuracy: ', svm.score(x_test, y_test)*100)


# # KNN

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor()
knn.fit(x_train, y_train)


# In[ ]:


knn_y_pred = knn.predict(x_test)
knn_score=r2_score(y_pred,y_test)
knn_rmse=np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:


print("R2 Score : ", knn_score)
print("RMSE : ", knn_rmse)
print('Training Accuracy: ', knn.score(x_train, y_train)*100)
print('Testing Accuracy: ', knn.score(x_test, y_test)*100)


# # Decision tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor().fit(x_train, y_train)
y_pred = dtr.predict(x_test)


# In[ ]:


dtr_score = r2_score(y_test, y_pred)
dtr_rmse = mean_squared_error(y_test, y_pred, squared = False)


# In[ ]:


print("R2 Score : ", dtr_score)
print("RMSE : ", dtr_rmse)
print('Training Accuracy: ', dtr.score(x_train, y_train)*100)
print('Testing Accuracy: ', dtr.score(x_test, y_test)*100)


# # Random Forest 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor().fit(x_train, y_train)
y_pred = rfr.predict(x_test)


# In[ ]:


rfr_score = r2_score(y_test, y_pred)
rfr_rmse = mean_squared_error(y_test, y_pred, squared = False)


# In[ ]:


print("R2 Score : ", rfr_score)
print("RMSE : ", rfr_rmse)
print('Training Accuracy: ', rfr.score(x_train, y_train)*100)
print('Testing Accuracy: ', rfr.score(x_test, y_test)*100)


# In[ ]:


log = pd.DataFrame({
    'Models' : ['Linear Regression', 'Lasso Regression','Ridge Regression','SVM','KNN','Decision Tree', 'RandomForest',],
    'R2 Score' : [lr_score,lasso_score,ridge_score,svm_score,knn_score, dtr_score, rfr_score,],
    'RMSE' : [lr_rmse,lasso_rmse,ridge_rmse,svm_rmse,knn_rmse, dtr_rmse, rfr_rmse]
    
})
log


# In[ ]:


plt.title('R2 score',fontsize=15)
plt.bar(log['Models'], log['R2 Score'], color = ['r','g','b','y','c','m','k'])
plt.xlabel('Model',fontsize=15)
plt.ylabel('r2 score',fontsize=15)
plt.ylim(0, 1)


# In[ ]:


plt.title('rmse',fontsize=15)
plt.bar(log['Models'], log['RMSE'], color = ['r','g','b','y','c','m','k'])
plt.xlabel('Model',fontsize=15)
plt.ylabel('RMSE',fontsize=15)


# In[ ]:


knn_y_pred


# In[ ]:


y_test


# # KNN and RandomForest has the best performance with R2 Score : 0.973071 , RMSE : 0.022930 and R2 Score : 0.973042 , RMSE : 0.023153 respectively

# In[ ]:





# In[ ]:




