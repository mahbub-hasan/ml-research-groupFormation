##  Road Accident Forcasting
 
## Step 1

### Importing all necessary machine learning libraries

### Get the original data from csv file as dataframe

=> `pd.read_csv("data.csv")`

##find the shape of data
=>data.shape


#step 2

Clean the Data as requirement and keep only (Date    Serial Number	District where death(s) happened	Number of Accident	Number of Death	   Number of Injury)

Import clean data set  Give a new name data1.

After That find The data dtype  for change the catagorial data to numeric value


#Step 3

After that plot are print from the data set

First Plot- x=District where death(s) happened  y=Number of Accident

Second plot- First index_col='Date'  so that it can't be difine as a normal string  So here x=date and y= Number of Accident

Third Plot- x=date, y= Number of Death

Fouth Plot x= Date , y=Number of Injury

## Step 4

After that find the Augmented Dickey-Fuller, Probability valu,  Num of Lags, Num of observation used for ADF Regression and critical values conclution,critical values  from the dataset 
Coloum (Number of Accident, Number of Death and Number of Injury)

## Step 5

#figure out order for ARIMA model
from pmdarima import auto_arima

#ignore hramless warming
import warnings

#Step 6


#split data into training and testing

Traing data 1-1808
Testing Data 1808-2008

#train the model

#make prediction on test set

After That Find the mean of test data


After that find the error 
Error persentage is 3.73%

##Step 7
Fit the train data in model and predection done by test data

Finialy saw the 1 year road accident prediction from this model.
