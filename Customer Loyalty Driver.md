# Customer Loyalty Consulting Project,Grocery-Retail
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
  + Less informative variables: Remove average sales percent from the following departments 
    * SHOWCASE MEAT
    * THIRD PARTY GIFT CARDS
    * SUSHI FOODS
    * SALAD BAR
    * PIZZARIA
    * PROMOTIONAL TICKETS SALES
    * SCHNUCKS COOKING SCHOOL
#####  Total of 0.2% data were removed due to the above criteria

### 4. Model selection

### 5. Results interpretation & business insights

### 6. Future scope and extension

