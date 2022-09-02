#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing Assignment

# ### Problem Statement:
# A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 
# 
# Essentially the company wants :
# - To understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19, by creating a linear model.
# - To identify the variables affecting their revenues i.e. Which variables are significant in predicting the demand for shared bikes.
# - To know the accuracy of the model, i.e. How well those variables describe the bike demands
# 
# They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.
# 
# ### Approach: 
# - Step 1 : Reading and understanding the Data
# - Step 2 : Exploratory Data Analysis / Visualising the Data
# - Step 3 : Data Preparation
# - Step 4 : Splitting the Data into Training and Testing Sets
# - Step 5 : Rescaling the Features
# - Step 6 : Building a liner model
# - Step 7 : Residual Analysis of the train data
# - Step 8 : Making Predictions Using the Final Model
# - Step 9: Model Evaluation
# - Final Conclusions
# - Final Recommendations for the Company:
# 

# ## Step 1: Reading and Understanding the Data

# ### 1.1 Reading the Bike sharing dataset

# In[334]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[335]:


import numpy as np
import pandas as pd


# In[336]:


bike_sharing = pd.read_csv("day.csv")


# In[337]:


# Check the head of the dataset
bike_sharing.head()


# ## Inspecting the Dataset

# In[338]:


bike_sharing.shape


# In[339]:


bike_sharing.info()


# In[340]:


bike_sharing.describe()


# ### 1.2 Data Quality Check

# #### Checking for NULL/MISSING values

# In[341]:


# percentage of missing values in each column
round(100*(bike_sharing.isnull().sum()/len(bike_sharing)), 2).sort_values(ascending=False)


# In[342]:


# percentage of missing values in each row
round((bike_sharing.isnull().sum(axis=1)/len(bike_sharing))*100,2).sort_values(ascending=False)


# #### Inferences : 
# No missing/NULL values found

# In[343]:


#### Checking for duplicate values

bike_dup = bike_sharing.copy()

# Checking for duplicates and dropping the entire duplicate row if any
bike_dup.drop_duplicates(subset=None, inplace=True)
bike_dup.shape


# In[344]:


bike_sharing.shape


# ####  Inferences: 
# Duplicate values are not present

# ### 1.3 Data Cleaning
# 
# Checking value_counts() for entire dataframe.
# 
# This will help to identify any Unknown/Junk values present in the dataset.

# Create a copy of the  dataframe, without the 'instant' column,as this will have unique values, and donot make sense to do a value count on it.

# In[345]:


bike_dummy=bike_sharing.iloc[:,1:16]


# In[346]:


for col in bike_dummy:
    print(bike_dummy[col].value_counts(ascending=False), '\n\n\n')


# ####  Inferences: 
# There seems to be no Junk/Unknown values in the entire dataset.

# ### 1.4 Removing redundant & unwanted columns

# Based on the high level look at the data and the data dictionary, the following variables can be removed from further analysis:
# 
# - instant : Its only an index value , we have a default index for the same purpose
# 
# - dteday : This has the date, Since we already have seperate columns for 'year' & 'month',hence, we can carry out our analysis without this column .
# 
# - casual & registered : Both these columns contains the count of bike booked by different categories of customers. Since our objective is to find the total count of bikes and not by specific category, we will ignore these two columns.
# 
# We will save the new dataframe as bike_new, so that the original dataset is preserved for any future analysis/validation

# In[347]:


bike_sharing.columns


# In[348]:


bike_new=bike_sharing[['season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'cnt']]


# In[349]:


bike_new.info()


# ## Step 2: Visualising the Data

# #### Here we'll do the following tasks:
# - We would be able to check if all the variables are linearly related or not (important if we want to proceed with a linear model)
# - Checking if there are any multicollinearity that exist
# - Here's where we can also identify if some predictors directly have a strong association(correlation) with the outcome variable
# 
# We'll visualise our data using `matplotlib` and `seaborn`.

# In[350]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### 2.1  Visualising Numeric Variables : Using a pairplot

# In[351]:


sns.pairplot(data = bike_new, vars=['cnt', 'temp', 'atemp', 'hum','windspeed'])
plt.show()


# ####  Inference: 
# 
# - By visualising the numeric variables, we can conclude that a linear model can be considered in this case because there are atleast some independent variables like atemp , temp etc. that show a positive correlation with the target variable cnt .  

# ### 2.2 Visualising Categorical Variables : Using a Boxplot

# In[352]:


plt.figure(figsize=(20, 15))
plt.subplot(2,4,1)
sns.boxplot(x = 'season', y = 'cnt', data = bike_new)
plt.subplot(2,4,2)
sns.boxplot(x = 'yr', y = 'cnt', data = bike_new)
plt.subplot(2,4,3)
sns.boxplot(x = 'holiday', y = 'cnt', data = bike_new)
plt.subplot(2,4,4)
sns.boxplot(x = 'weekday', y = 'cnt', data = bike_new)
plt.subplot(2,4,5)
sns.boxplot(x = 'workingday', y = 'cnt', data = bike_new)
plt.subplot(2,4,6)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bike_new)
plt.subplot(2,4,7)
sns.boxplot(x = 'mnth', y = 'cnt', data = bike_new)
plt.show()


# #### Inference: 
# 
# - For the variable season, we can clearly see that the category 3 : Fall, has the highest median, which shows that the demand was high during this season. It is least for 1: spring .
# - The year 2019 had a higher count of users as compared to the year 2018
# - The bike demand is almost constant throughout the week.
# - The count of total users is in between 4000 to 6000 (~5500) during clear weather
# - The count is highest in the month of August
# - The count of users is less during the holidays

# ## Step 3: Data Preparation

# #### Mapping the categorical values to their respective categorical string values (reference data dictionary)

# In[353]:


import calendar
bike_new['mnth'] = bike_new['mnth'].apply(lambda x: calendar.month_abbr[x])


# In[354]:


# Maping seasons
bike_new.season = bike_new.season.map({1: 'Spring',2:'Summer',3:'Fall',4:'Winter'})


# In[355]:


# Mapping weathersit
bike_new.weathersit = bike_new.weathersit.map({1:'Clear',2:'Mist & Cloudy', 
                                             3:'Light Snow & Rain',4:'Heavy Snow & Rain'})


# In[356]:


#Mapping Weekday
bike_new.weekday = bike_new.weekday.map({0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thrusday",5:"Friday",6:"Saturday"})


# In[357]:


# Check the dataframe now

bike_new.head()


# In[ ]:





# ### 3.1 Creating Dummy Variables

# The variables `mnth` `weekday` `season` `weathersit` have various levels, for ex, `weathersit` has 3 levels , similarly variable `mnth` has 12 levels.   
# We will create DUMMY variables for these 4 categorical variables namely - `mnth`, `weekday`, `season` & `weathersit`.

# In[358]:


# Get the dummy variables for the features ''season','mnth','weekday','weathersit'' and store it in a new variable - 'dummy'
dummy = bike_new[['season','mnth','weekday','weathersit']]


# In[359]:


dummy = pd.get_dummies(dummy,drop_first=True )


# In[360]:


# Adding the dummy variables to the original dataset
bike_new = pd.concat([dummy,bike_new],axis = 1)


# In[361]:


# Checking the dataframe

bike_new.head()


# In[362]:


#Deleting the orginal columns season.weathersit,weekday,mnth
bike_new.drop(['season'],axis=1,inplace=True)
bike_new.drop(['weathersit'],axis=1,inplace=True)

bike_new.drop(['weekday'],axis=1,inplace=True)

bike_new.drop(['mnth'],axis=1,inplace=True)


bike_new.head()


# In[363]:


bike_new.shape


# In[364]:


bike_new.info()


# ## Step 4: Splitting the Data into Training and Testing Sets
# The first basic step for regression is performing a train-test split.

# In[365]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(bike_new, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[366]:


test.shape


# In[367]:


train.shape


# In[368]:


train.info()


# ## Step 5 :Rescaling the Features

# Although scaling doesnot impact the linear model in the case of simple linear regression, however while performing multiplwe linear regression it might impact the model. As we can see that the value of the feature cnt has much higher values as compared to the other features like temp, atemp etc.So it is extremely important to rescale the variables so that they have a comparable scale. If we don't have comparable scales, then some of the coefficients as obtained by fitting the regression model might be very large or very small as compared to the other coefficients. This might become very annoying at the time of model evaluation. So it is advised to use standardization or normalization so that the units of the coefficients obtained are all on the same scale. There are two common ways of rescaling:
# 
# - Min-Max scaling
# - Standardisation (mean-0, sigma-1)
# 
# This time, we will use MinMax scaling.

# In[369]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[370]:


# Apply scaler() to all the columns except the 'dummy' variables.
num_vars = ['cnt','hum','windspeed','temp','atemp']

train[num_vars] = scaler.fit_transform(train[num_vars])


# In[371]:


train.head()


# In[372]:


train.describe()


# In[ ]:





# #### 5.1 Checking the coefficients to see which variables are highly correlated

# In[373]:


num_features = ["temp","atemp","hum","windspeed","cnt"]
plt.figure(figsize=(15,8),dpi=130)
plt.title("Correlation of numeric features",fontsize=16)
sns.heatmap(bike_new[num_features].corr(),annot= True,cmap="mako")
plt.show()


# In[374]:


plt.figure(figsize = (20, 25))
sns.heatmap(train.corr(), annot = True, cmap='YlGn_r')
plt.show()


# ####  Conclusion:
# As can be seen from the map, `atemp` and `temp` seems to be correlated to the target variable `cnt`. Since, not much can be stated about the other independent variables , hence we'll build a model using all the columns.

# In[375]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Temp",fontsize=16)
sns.regplot(data=bike_new,y="cnt",x="temp")
plt.xlabel("Temperature")
plt.show()


# In[376]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Temp",fontsize=16)
sns.regplot(data=bike_new,y="cnt",x="atemp")
plt.xlabel("Temperature")
plt.show()


# #### Inference: 
# - Demand for bikes is positively correlated to temp.
# - We can see that cnt is linearly increasing with temp indicating linear relation.

# #### 5.2 Dividing into X and Y sets for the model building

# In[377]:


y_train = train.pop('cnt')
X_train = train


# In[378]:


y_train.shape


# In[ ]:





# ## Step 6: Building a linear model
# 
# APPROACH USED :
# 
# - We will use a mixed approach to build the model.
# 
# - Here we are using (RFE) approach for feature selection and then we will use the (statsmodel) approach for building the model 

# #### Feature Selection
# 
# - We start with 15 variables.
# 
# - We need to use the LinearRegression function from SciKit Learn for its compatibility with RFE (which is a utility from sklearn)

# In[379]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[380]:


# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 15)            
rfe = rfe.fit(X_train, y_train)


# In[381]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[382]:


col = X_train.columns[rfe.support_]
col


# In[383]:


X_train.columns[~rfe.support_]


# ### 6.1 Building model using statsmodel, for the detailed statistics

# In[384]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# #### Adding a constant variable 
# For statsmodels, we need to explicitly fit a constant using sm.add_constant(X) because if we don't perform this step, statsmodels fits a regression line passing through the origin, by default.

# In[385]:


import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)


# In[386]:


X_train_rfe.head()


# #### Running the linear model

# In[387]:


lm_1 = sm.OLS(y_train,X_train_rfe).fit()


# In[388]:


# Check the parameters obtained
lm_1.params


# #### Summary of the linear model

# In[389]:


print(lm_1.summary())


# #### Inference:  : 
# Here we see that the p-value for all the variables is < 0.05 . Hence, we keep all the columns and proceed with the model. 

# ### Checking VIF for multicollinearity
# 
# Variance Inflation Factor or VIF, gives a basic quantitative idea about how much the feature variables are correlated with each other. It is an extremely important parameter to test our linear model. The formula for calculating `VIF` is:
# 
# ### 𝑉𝐼𝐹 = 1 / [1−(𝑅𝑖)^2]

# In[390]:


# Calculate the VIFs for the new model
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### We generally want a VIF that is less than 5. So there are clearly some variables we need to drop.

# In[391]:


# dropping `const` column as the vif is > 5
X_train_rfe = X_train_rfe.drop(['const'], axis=1)


# In[392]:


# Calculate the VIFs for the new model again
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[393]:


# dropping `hum` column as the vif is > 5
X_train_rfe = X_train_rfe.drop(['hum'], axis=1)


# In[394]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### Note:
# The VIF value of temp is nearly equal to 5. Hence, we are not dropping this feature.

# ### Preparing the final model

# In[395]:


# Adding a constant variable 
X_train_lm = sm.add_constant(X_train_rfe)

# Create a first fitted model
lm_2 = sm.OLS(y_train,X_train_lm).fit()


# In[396]:


# Check the summary
print(lm_2.summary())


# In[397]:


# Calculate the VIFs for the new model again
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### Conclusion:
# 
# - Here we see that except for temp(that has a vif value slightly more than 5, that can be ignored) rest all the columns have a vif value less than 5.
# 
# - Hence, we finalise `lm_2` as the final model to proceed with the future prdeictions.

# ## Step 7: Residual Analysis of the train data

# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[398]:


y_train_cnt = lm_2.predict(X_train_lm)


# In[399]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# #### Infrences: 
# 
# We can clearly see that the error terms are centred around 0 and follows a normal distribution, this is in accordance with the stated assumptions of linear regression.

# #### Cross-verifying the above conclusion using a qq-plot as well:

# In[400]:


# Plot the qq-plot of the error terms
sm.qqplot((y_train - y_train_cnt), fit=True, line='45')
plt.show()


# ### Conclusion:
# Here we see that most of the data points lie on the straight line , which indicates that the error terms are normally distributed .

# ## Step 8: Making Predictions Using the Final Model

# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the final model that we got.

# #### Applying the scaling on the test sets

# In[401]:


num_vars = ['cnt','hum','windspeed','temp','atemp']


test[num_vars] = scaler.transform(test[num_vars])


# In[402]:


test.describe()


# #### Dividing into X_test and y_test

# In[403]:


y_test = test.pop('cnt')
X_test = test


# In[404]:


# Adding constant variable to test dataframe
X_test = sm.add_constant(X_test)


# #### Predicting using values used by the final model

# In[405]:


test_col = X_train_lm.columns
X_test=X_test[test_col[1:]]
# Adding constant variable to test dataframe
X_test = sm.add_constant(X_test)

X_test.info()


# In[406]:


# Making predictions using the final model

y_pred = lm_2.predict(X_test)


# #### Calculating the r-squared
# 
# R-squared is a goodness-of-fit measure for linear regression models. This statistic indicates the percentage of the variance in the dependent variable that the independent variables explain collectively. R-squared measures the strength of the relationship between your model and the dependent variable on a convenient 0 – 100% scale.

# In[407]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# ## Conclusion:
# 
# We can see that the "r_squared on the test set is 0.813" and the "r-squared on the trained set 0.840" which is quiet reasonable and nearly equal, which means that whatever data the model was trained with, it has been almost able to apply those learnings in the test data.

# ## Step 9: Model Evaluation

# Plotting the graph for actual versus predicted values.

# In[408]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)


# ### Conclusion: 
# We can colude that the final model fit isn't by chance, and has descent predictive power.

# #### Getting the variable names and the coefficient values for the final equation of the best fitted line

# In[409]:


param = pd.DataFrame(lm_2.params)
param.insert(0,'Variables',param.index)
param.rename(columns = {0:'Coefficient value'},inplace = True)
param['index'] = list(range(0,15))
param.set_index('index',inplace = True)
param.sort_values(by = 'Coefficient value',ascending = False,inplace = True)
param


# ## Final Conclusions : 
# By using the above scatter plot and the table , We can see that the equation of our best fitted line is:
# 
# $ cnt = 0.2466 + 0.437 \times  temp + 0.2342  \times  yr + 0.8865 \times season Winter + 0.0682 \times mnth Sept + 0.0033 \times season Summer - 0.0418 \times mnth Nov - 0.04452 \times mnth Dec - 0.0050 \times mnth Jan - 0.0503 \times mnth Jul - 0.0716 \times season Spring - 0.0814 \times weathersit Mist Cloudy - 0.0919 \times holiday - 0.1585 \times windspeed - 0.2928 \times weathersit Light Snow Rain $

# #### All the positive coefficients like temp,season_Summer indicate that an increase in these values will lead to an increase in the value of cnt.
# #### All the negative coefficients indicate that an increase in these values will lead to a decrease in the value of cnt.

# - From R-Sqaured and adj R-Sqaured value of both train and test dataset we could conclude that the above variables can well explain more than 81% of bike demand.
# - Coeffiencients of the variables explains the factors effecting the bike demand
# 
# - Based on final model top three features contributing significantly towards explaining the demand are:
# 
#  - Temperature (0.437655)
#  - weathersit : Light Snow, Light Rain + Mist & Cloudy (-0.292892)
#  - year (0.234287)
# 
# Hence, it can be clearly concluded that the variables `temperature` , `season`/ `weather situation` and `month`  are significant in predicting the demand for shared bikes .

# ##  Final Recommendations for the Company: 

# - The months - Jan , Jul , Sep , Nov , Dec should be considered by the company as they have a higher demand as compared to other months.
# - With an increase in temperature the demand also increases, hence it should keep track of the weather conditions.
# - During the Winter season the demand rises, hence it should be well prepared to meet the high demand

# In[ ]:




