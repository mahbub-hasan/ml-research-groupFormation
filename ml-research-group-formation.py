
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pip install simulated_annealing

"""## Importing Data"""

original_data = pd.read_csv("drive/MyDrive/Research Methodology 2020/Dataset/GroupFormation Dataset - Sheet1.csv")
original_data['Student ID'] = original_data['Student ID'].str.replace('-','')
original_data

"""# **Find and Drop Duplicate Data** """

#Sorted Data
sorted_data = original_data.sort_values(by=['Student ID','Course Code','Semester','Marks Round'])

# Drop Duplicates data
fresh_data = sorted_data.drop_duplicates(subset=['Student ID','Course Code'],keep='last')

# Count Duplicate Data
fresh_data[fresh_data.duplicated(['Student ID', 'Course Code'])].count()

# now original_data is fresh data
original_data = fresh_data.drop(['Total','Grades'],axis=1)

# Drop rows which Marks Round is lower than 40
original_data =  original_data[original_data['Marks Round'] >= 40]
original_data

original_data.dtypes

original_data.info()

original_data.describe()

"""## Co-relations"""

# Co-relation of ogininal data
original_data.corr()


original_data.isna().sum()


"""#Prerequisite Course"""

seu_curriculum = pd.read_csv("drive/MyDrive/Research Methodology 2020/Dataset/SEU Curriculum - Sheet1.csv")
seu_curriculum['Prerequisite'] = seu_curriculum['Prerequisite'].str.split('\n').str[0]
seu_curriculum

prerequisite_course = seu_curriculum[['Course Code', 'Prerequisite']]
# prerequisite_course[prerequisite_course['Prerequisite'].isnull()]
prerequisite_course

# getting prerequisite course
def getPrerequisite(courseCode):
  return str(prerequisite_course.loc[prerequisite_course['Course Code'] == courseCode, 'Prerequisite'].iloc[0])
  # return prerequisite_course.get(courseCode)

getPrerequisite('ENG1001')

# Get All Prerequisite
def getAllPrerequisites(CourseCode):
  list = []
  prereq_course = getPrerequisite(CourseCode)
  while(prereq_course != 'nan'):
    list.append(prereq_course)
    # print(prereq_course)
    prereq_course = getPrerequisite(prereq_course)
  
  return list

getAllPrerequisites('MATH2015')

"""# Main WorkFlow"""

thirty_students = original_data[(original_data['Course Code'] == 'CSE2015') & (original_data['Semester'] == 'Fall-2017')]
thirty_students = thirty_students.head(30)
# Replacing Female to 0 and male to 1
thirty_students['Gender'].replace(['female','male'],[0,1],inplace=True)
thirty_students = thirty_students.reset_index(drop=True)
thirty_students

"""# Random Cluster"""

len = thirty_students.shape[0]
arr = []
for i in range(len):
  arr.append(i // 3)

thirty_students['Cluster'] = arr
thirty_students

len = thirty_students.shape[0]
arr = []

for i in range(len):
  arr.append(i // 3)

# suffle array
# np.random.seed(42)
np.random.shuffle(arr)

thirty_students['Cluster'] = arr
thirty_students

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

X = thirty_students[['Student ID','Credits','Gender','Marks Round']].values
y = thirty_students['Cluster'].values
  # Split the data into test and train sets                     
X_train, X_test, y_train, y_test = train_test_split(X, y)

# np.random.seed(42)
model = RandomForestClassifier(n_estimators=20).fit(X_train, y_train)
# model.fit(X_train, y_train)
model.score(X_test, y_test)

thirty_students.plot(x='Student ID', y='Cluster');

"""# Improving ML Model"""

def optimaizedGroupFormation(data, courseCode, semester):

  from sklearn import svm, datasets
  from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
  from simulated_annealing.optimize import SimulatedAnneal
  from sklearn.linear_model import SGDClassifier
  from sklearn.neighbors import NearestNeighbors
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
  from sklearn.cluster import KMeans
  import random
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.compose import ColumnTransformer

  best_score = 0.0
  best_data = pd.DataFrame()

  for num in range(5):
    # thirty_students = thirty_students.sort_values(by=['Marks Round'])
    # thirty_students = thirty_students.sample(frac=1).reset_index(drop=True)
    # thirty_students = thirty_students.sample(frac=1)

    len = data.shape[0]
    cluster = []

    for i in range(len):
      cluster.append(i // 3)


    data['Cluster'] = cluster



    X = data[['Student ID','Credits','Gender','Marks Round']].values
    y = data['Cluster'].values
    # Split the data into test and train sets                         
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    svc_params = {'C':np.logspace(-8, 10, 19, base=2),
                  'fit_intercept':[True, False]
                }
    # Using a linear SVM classifier             
    clf = svm.LinearSVC()
    # Initialize Simulated Annealing and fit
    sa = SimulatedAnneal(clf, svc_params, T=10.0, T_min=0.001, alpha=0.75,
                            verbose=True, max_iter=1, n_trans=5, max_runtime=300,
                            cv=3, scoring='f1_macro', refit=True)
    sa.fit(X_train, y_train)

    model = RandomForestRegressor().fit(X_train, y_train)
    current_score = model.score(X_test, y_test) * 100
    print(f"Model accuracy on test set: {current_score:.2f}%")
    print("")
    if(current_score > best_score):
      best_score = current_score
      best_data = data

  print(f"Best Score ========================> {best_score:.2f}%")
  final_data = best_data.drop(['Marks Round'],axis=1)
  final_data['Course Code'] = courseCode
  final_data['Semester'] = semester
  return final_data

data = thirty_students
courseCode = 'CSE4029'
semester = 'Fall-2020'
result = optimaizedGroupFormation(data, courseCode ,semester)
result

result.plot(x='Student ID', y='Cluster');


"""# Working with Testing Data"""

testing_data = pd.read_csv("drive/MyDrive/Research Methodology 2020/Dataset/GroupFormation Dataset - Testing Data.csv")
testing_data['Student ID'] = testing_data['Student ID'].str.replace('-','')
testing_data

getPrerequisite('CSE4029')

s = pd.DataFrame({'Student ID':[], 'Course Code':[], 'Credits':[], 'Semester':[], 'Gender':[], 'Marks Round': []})
c = original_data[(original_data['Student ID'] == '2170022') & (original_data['Course Code'] == 'CSE2015')]
s.append(c, ignore_index=True)
s

student_ids = testing_data['Student ID'].values
student_ids

pre_data = pd.DataFrame({'Student ID':[], 'Course Code':[], 'Credits':[], 'Semester':[], 'Gender':[], 'Marks Round': []})
for id in student_ids:
  if(not original_data[(original_data['Student ID'] == id) & (original_data['Course Code'] == 'CSE2015')].empty):
    row = original_data[(original_data['Student ID'] == id) & (original_data['Course Code'] == 'CSE2015')]
    pre_data.append(row)
pre_data

original_data[(original_data['Student ID'] == '2170023') & (original_data['Course Code'] == 'CSE2015')]

df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
df.append(df2)

