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
#### Linear Regression was Adopted to Better Facilitate Interpretation
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

