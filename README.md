# Group Formation Using Maching Learning

## Step 1

### Importing all necessary machine learning libraries

### Get the original data from csv file as dataframe

=> `pd.read_csv("data.csv")`

## Step 2

Clean the Data as requirement and keep only (Student ID,	Course Code,	Credits,	Semester	Gender,	Marks Round) column

Import prerequisite course data from csv

```python

pd.read_csv("prerequisite.csv")

```

## Step 3

Now, From original data separate 30 students with specific “course code” and “Semester” for working with machine learning model with a new variable of thirty students.

```python

original_data[(original_data['Course Code'] == 'CSE2015') & (original_data['Semester'] == 'Fall-2017')].head(30)

```
 
After that, we arbitrarily cluster them into 10 groups with maximum 3 students and pass them to our machine learning group formation model.
 
In machine learning model it takes the data with a random cluster. It split the data for training a ML model like below with 20% text size.

```python

 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
```

## Step 4

Here, our optimizer simulated annealing takes training data and optimized it internally because it's a well known machine learning model for group formation made by ML expert people with several iterations.

```python

sa = SimulatedAnneal(clf, svc_params, T=10.0, T_min=0.001, alpha=0.75,
                           verbose=True, max_iter=1, n_trans=5, max_runtime=300,
                           cv=3, scoring='f1_macro', refit=True)
sa.fit(X_train, y_train)

```

The whole process goes in several iterations and  finally we got our best accuracy and the result with the best group of students.
