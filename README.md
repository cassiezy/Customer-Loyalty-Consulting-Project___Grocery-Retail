# Customer Loyalty Consulting Project,Grocery Retail
#### Business questions:
Help client understand customer loyalty driver

### 1. Loyalty definition
![image](https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/1%20definition.png)
+ Defining the customer loyalty and understanding the driving factors are essential to increase retention rate and long-term business revenue

+ Inspired by the RFM model, monetary and frequency metrics are considered most appropriate measurement for customer loyalty

+ The primary definition of customer loyalty is chosen to be the average weekly spending, which is a production of F and M
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/2.png" width = '400'>

### 2. Variable proposal (4 categories)
In total, 46 explanatory variables in 4 categories proposed, with explanation and rationale are as followed:
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/variables3 .png" width = '800'>

### 3. Data preparation
- Use Panel Data Before COVID to Better Study Customer Loyalty
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/4.png" width = '800'>

- Additional Filtering Assumptions on Data Cleaning
  + Outliers
    * Remove average basket size > 300  (0.07%)
    * Remove average trips per week > 10  (0.06%)
    * Remove average promotion count percent > 50%  (0.01%)
    * Remove average promotion dollar percent > 50%  (0.01%)
    * Remove average unique skus per transaction > 75  (0.02%)
    * Remove average number of unique stores > 15  (0.01%)
    * Remove average number of departments/transaction > 10  (0.01%)
    * Remove average price per item > 80  (0.03%)
    <img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/5.png" width = '300'>
  + Less informative variables: Remove average sales percent from the following departments 
    * SHOWCASE MEAT
    * THIRD PARTY GIFT CARDS
    * SUSHI FOODS
    * SALAD BAR
    * PIZZARIA
    * PROMOTIONAL TICKETS SALES
    * SCHNUCKS COOKING SCHOOL
#####  Total of 0.2% data were removed due to the above criteria

- Customers are Labeled into 4 Groups to Add Interaction Factors
  + Customers have different shopping habits
  + How loyalty is affecCted by the features may be different across people of different shopping habits
  + Customers are labeled into 4 groups according to their average basket size and average trips per week to avoid estimation bias introduced by their intrinsic differences
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/6.png" width = '900'>

### 4. Model selection
#### Linear Regression was adopted to better facilitate interpretation. 4 more machine learning models to see in appendix, including gradient boosting, random forest, SVM and Neural Network.
#### Confounders
+ Season indicator
+ Most frequent store

#### Interactions
+ Group indicator

#### Model performance
+ Adjusted R-squared: 0.7011

#### Variable significance
+ 250 out of 624 variables

<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/7.png" width = '600'>

### 5. Results interpretation & business insights
#### 5 Key Discoveries and Recommended Business Strategies
#### 1. Significant variables
+ Stimulate customers to try more kinds of products
+ Encourage customers to download mobile app when sign up
+ Offer special gifts with Schnucks logo for long-time customers to improve emotional connection
+ Provide customers of higher price per item with extra rewards points 
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/8.png" width = '400'>

#### 2. Percentage influence of departments
+ Make promotion campaigns for products in the departments with positive influence
+ Optimize products on the departments that have negative impact
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/9.png" width = '400'>

#### 3. Group 1 is more sensitive to promotions 
+ Make specialized email campaign and notification of mobile app of promotion information, and push frequently to Group 1 customers.
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/13.png" width = '200'>
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/10.png" width = '400'>

#### 4. Group 1 and 3 have positive coefficients of num_of_unique_store
+ Encourage group 1&3 customers to go to other Schnucks stores. 
+ Design a ‘collecting stamps’ campaign. When they go to 5 and more stores, they can get special Schnucks-logo gift or coupons.
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/11.png" width = '400'>

#### 5. Group 2 has higher coefficient of avg_pur_dpmt
+ Give incentives to group 2 customers for getting more products in different departments. 
+ Design bundle products to upgrade the baskets.
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/12.png" width = '400'>


### 6. Future scope and extension
#### 1. Segment customers based on the linear loyalty model 
+ Launch specialized coupons and campaigns for different groups
+ Design customized email marketing and push notification
<img src="https://github.com/cassiezy/Customer-Loyalty-Consulting-Project-Grocery-Retail/blob/master/pic/14.png" width = '200'>

#### 2. Predict loyalty score using optimal machine learning model 
+ Make business strategies ahead of time to improve customer experiences and loyalty in advance
+ Use 4 ML models to predict alternative loyalty score that is set as weighted average of RFM (see appendix) as a reference
 
#### 3. Track customer loyalty changes overtime
+ Notice every customer’s  potential loyalty trend to make tactics
+ Decrease churn rate and increase customer satisfaction


### Appendix 1 - Alternative Loyalty Definition
#### Weighted average of recency, frequency, and monetary score
+ RFM score = ⅓ * R score + ⅓ * F score + ⅓ * M score
+ R score = 1 - weighted_recency* scaled between 0 and 1
+ F score = num_of_trips_per_week scaled between 0 and 1
+ M score = basket_size scaled between 0 and 1

### Appendix 2 - Variables Used in ML Model to Predict RFM Score

#### Target
+ rfm_score ( ⅓ * R score + ⅓ * F score + ⅓ * M score)

#### Predictors
+ **Loyalty Program**
  - mobile_use , datediff, duration
+ **Basket Profile**
  - ppi, avg_unique_sku, avg_pur_dpmt, dpmt_sales% ('BAKERY', 'CHEF EXPRESS', 'CIGARETTES', 'COFFEE BAR', 'DAIRY', 'DELI', 'DONATIONS', 'DRUGS', 'FLORAL', 'FROZEN FOOD', 'GENERAL MERCHANDISE', 'GROCERY', 'LIQUOR', 'MEAT', 'PHARMACY', 'PIZZARIA', 'PRODUCE', 'PROMOTIONAL TICKETS SALES', 'RUG DOCTOR INCOME', 'SALAD BAR', 'SCHNUCKS COOKING SCHOOL', 'SEAFOOD', 'SHOWCASE MEAT', 'SUPPLIES', 'SUSHI FOODS', 'THIRD PARTY GIFT CARDS')
+ **Promotion Engagement**
  - avg_promo_count_percent, avg_promo_dollar_percent, redemption_rate, avg_week_redeemed
+ **Geographic Preference**
  - unique_store, most_freq_visit, schn_store_num, other_store_num
  
### Appendix 3 - Gradient Boosting Outperformed 4 Other ML Models

#### Gradient Boosting
+ Best_params = {‘learning rate’ : 0.2, ‘max_depth’ : 2, ‘n_estimators’ : 300, ‘subsample’= 1.0} 
+ R-squared= 0.9137
+ mobile usage and promotion sensitivity are the most important features in the best-performing ML model
#### Random Forest
+ Best_params = {'max_depth': 50, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 201}
+ R-squared = 0.9013
#### SVM Regression
+ Best parameters = {'gamma':0.001,"C":100}
+ R-squared = 0.8719
#### Neural Network
+ 3 layer network with neurons (32, 32, 1)
+ R-squared =0.8657 
#### Linear Regression
+ R-squared = 0.8517

* *Features are based on all past data (including COVID period)*

### Appendix 4 - Gradient Boosting
```Python
from sklearn.ensemble import GradientBoostingRegressor
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
import pandas as pd
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
```

#### Read data
```Python
df = pd.read_csv('team_data/new_train.csv')
df.head()
df.columns

# Get values of X and Y
x_columns = df.columns.drop('rfm_score').drop('sso_user_id').drop('postcode')
x = df[x_columns]
y = df['rfm_score']
```

#### define the grid of values to search
```Python
grid = dict()
grid['n_estimators'] = [100,200,300]
grid['learning_rate'] = [0.1, 0.2]
grid['subsample'] = [0.5, 1.0]
grid['max_depth'] = [1, 2]
```
https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python/
```Python
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = GradientBoostingRegressor()

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring= 'r2')

# execute the grid search
grid_result = grid_search.fit(x, y)
# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
```

#### fit the model on the whole dataset
```Python
model = GradientBoostingRegressor(n_estimators=300, 
                                  learning_rate=0.2,max_depth=2, subsample = 1.0, random_state=0, loss='ls')
model.fit(x, y)
```
```Python
df_test = pd.read_csv('team_data/new_test.csv')
x_test = df_test[x_columns]
y_test = df_test['rfm_score']

pred_y = model.predict(x_test).flatten()


from sklearn import metrics
print('test R2:', metrics.r2_score(y_test,pred_y))
```
#### test R2: 0.9137088736373132

```Python
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(df.columns)[sorted_idx])
plt.title('Feature Importance (MDI)')

result = permutation_importance(model, x_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(df.columns)[sorted_idx])
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()
```
#### mobile usage and promotion sensitivity are the most important features in the best-performing ML model


### Appendix 5 - Random Forest
```Python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
```

#### train_test split
```Python
#split x and y in train data
y=train.loc[:,'rfm_score']
x=train.iloc[:,2:]

#Split validation set (inside test)
from sklearn.model_selection import train_test_split
x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=0.2, random_state=0)

#split x and y in test data (final test)
y_te=test.loc[:,'rfm_score']
x_te=test.iloc[:,2:]
x_te.head()
```

#### build base model with whole train data
```Python
base_rf1 = RandomForestRegressor(n_estimators=40,oob_score = True,random_state=0)
base_rf1.fit(x, y)
y_pred = base_rf1.predict(x)
print('train MSE:', metrics.mean_squared_error(y,y_pred))
print('Out of Bag:',base_rf1.oob_score_)
```
train MSE: 0.00011083334191487102
Out of Bag: 0.8936985746460763

#### use grid search to find optimal parameters
```Python
ntree = range(101,211,10)
min_samples_split = [2, 5]
max_depth=range(10,110,40)
min_samples_leaf = [1, 2, 4]
param ={'n_estimators':ntree,
        'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf}

gs_rf = GridSearchCV(estimator=RandomForestRegressor(oob_score = True,random_state=0), 
                     param_grid=param, n_jobs=-1, cv=5, scoring='r2')
gs_rf.fit(x,y)

print('best_params:', gs_rf.best_params_, gs_rf.best_score_)
```
best_params: {'max_depth': 50, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 201} 0.9005132352402694

```Python
opt_rf =RandomForestRegressor(n_estimators=201,max_depth=50,min_samples_leaf=2,min_samples_split=5,oob_score = True,random_state=0)
opt_rf=opt_rf.fit(x,y)
y_te_pred=gs_rf.predict(x_te)
print('test R2:', metrics.r2_score(y_te,y_te_pred))
print('Out of Bag:',opt_rf.oob_score_)
```
test R2: 0.9013272660802846
Out of Bag: 0.901400117074135

#### initial RF with validation. we can see there is a overfitting problem.
```Python
base_rf = RandomForestRegressor(n_estimators=40,oob_score = True,random_state=0)
base_rf.fit(x_tr, y_tr)
y_pred1 = base_rf.predict(x_va)
y_tr_pred=base_rf.predict(x_tr)
print('test MSE:', metrics.mean_squared_error(y_va,y_pred1))
print('train MSE:', metrics.mean_squared_error(y_tr,y_tr_pred))
print('Out of Bag:',base_rf.oob_score_)
```
test MSE: 0.0022654992256412516
train MSE: 0.00034827731433476874
Out of Bag: 0.8362582632788543

#### use validation and adjust parameters using grid_search
```Python
ntree = range(1,211,10)
max_depth=range(10,110,40)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
param ={'n_estimators':ntree,
        'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf}

gs_rfv = GridSearchCV(estimator=RandomForestRegressor(oob_score = True,random_state=0), 
                     param_grid=param, n_jobs=-1, cv=3, scoring='neg_mean_squared_error')
gs_rfv.fit(x_tr,y_tr)

print('best_params:', gs_rf.best_params_, gs_rf.best_score_)
```
best_params: {'max_depth': 50, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 201} -0.0022836226754578428

#### check validation MSE
```Python
opt_rf1=RandomForestRegressor(n_estimators=201,min_samples_leaf=4,min_samples_split=10,max_depth=50,oob_score = True,random_state=0)
opt_rf1.fit(x_tr, y_tr)
y_pred_opt= opt_rf1.predict(x_va)
y_tr_pred_opt=opt_rf1.predict(x_tr)

print('test MSE:', metrics.mean_squared_error(y_va,y_pred_opt))
print('train MSE:', metrics.mean_squared_error(y_tr,y_tr_pred_opt))
print('Out of Bag:',opt_rf.oob_score_)

```
test MSE: 0.002229613583784239
train MSE: 0.0007971369340714459
Out of Bag: 0.8491829684925485



### Appendix 6 - SVM

```Python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
```

```Python
X = train.drop(['sso_user_id','rfm_score', 'most_freq_store'], axis = 1)
y = train['rfm_score']

test = pd.read_csv('./team_data/new_test.csv')

X_val = test.drop(['sso_user_id','rfm_score', 'most_freq_store'], axis = 1)
y_val = test['rfm_score']

X = X.values
y = y.values
X_val = X_val.values

```
#### feature scaling
```Python
y= y.reshape(-1, 1)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
X_val = sc_X.fit_transform(X_val)
y = sc_y.fit_transform(y)
```
#### Fitting SVR to the dataset
```Python
X.shape, y.shape, X_val.shape
```
((20000, 40), (20000, 1), (2783, 40))
```Python
y= y.reshape(20000, )
```
#### Gridsearch
```Python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
```

```Python
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [0.01, 1e-3, 1e-4],'kernel': ['rbf']}

grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=2)
grid.fit(X, y)
```

```Python
print(grid.best_estimator_)
```
SVR(C=100, gamma=0.001)

```Python
best_parameters = {'gamma':0.001,"C":100}

regressor = SVR(kernel = 'rbf',C=best_parameters["C"], gamma=best_parameters["gamma"])
regressor.fit(X, y)
y_pred = regressor.predict(X_val)
y_pred = sc_y.inverse_transform(y_pred) 

metrics.r2_score(Y_val, y_pred)
```
0.8719051159255504


### Appendix 7 - Neural Network
```Python
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
```

```Python
from sklearn.model_selection import train_test_split
Y = df.loc[:, 'rfm_score']
X = df.drop(['rfm_score','sso_user_id','most_freq_store'], axis = 1)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
```

```Python
model = Sequential([Dense(32, activation='relu', input_shape=(40,)),Dense(32, activation='relu'),Dense(1, activation='sigmoid'),])
model.compile(optimizer='adam',loss='mse',metrics=['mae', 'mse'])
result = model.fit(X, Y, batch_size = 128, epochs = 100, validation_split = 0.2)
```

```Python
import matplotlib.pyplot as plt
print(result.history.keys())
# "Loss"
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
```

```Python
df_test = pd.read_csv('team_data/new_test.csv')
Y_test = df_test.loc[:, 'rfm_score']
X_test = df_test.drop(['rfm_score','sso_user_id','most_freq_store'], axis = 1)
pred_y = model.predict(X_test).flatten()
```

```Python
def mae_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(predicted[i] - actual[i])
    return sum_error / float(len(actual))
```

```Python
def mse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return mean_error
```

```Python
mae_metric(Y_test, pred_y)
```
0.02142082871220066

```Python
mse_metric(Y_test, pred_y)
```
0.000989031827716198
```Python
import sklearn.metrics
sklearn.metrics.r2_score(Y_test, pred_y, sample_weight=None, multioutput='uniform_average')
```
0.8657518000609311
