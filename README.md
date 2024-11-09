# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIJAY R
RegisterNumber:  212223240178
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![Screenshot 2024-10-16 071931](https://github.com/user-attachments/assets/915671d5-b2b2-406a-a5b7-b6e059e7b5cc)
![Screenshot 2024-10-16 071942](https://github.com/user-attachments/assets/99bbf63c-9d12-4279-912b-39004577ef42)
![Screenshot 2024-10-16 071954](https://github.com/user-attachments/assets/9070beb1-1ba2-484a-bbfa-810645daf1a4)
![Screenshot 2024-10-16 072007](https://github.com/user-attachments/assets/bc973564-bd56-4d0d-a222-9ae590798ec9)
![Screenshot 2024-10-16 072019](https://github.com/user-attachments/assets/84a69f81-0059-47d1-9bbe-0b950509bb25)
![Screenshot 2024-10-16 072702](https://github.com/user-attachments/assets/447b8302-5efe-46d3-88ca-008819725084)
![Screenshot 2024-10-16 072722](https://github.com/user-attachments/assets/ad6298e1-34fd-4e6f-a23a-e073c2f8dc01)
![Screenshot 2024-10-16 072733](https://github.com/user-attachments/assets/5ffa20e8-ebc0-45b4-acdb-5d8d7ea4adc2)
![Screenshot 2024-10-16 072755](https://github.com/user-attachments/assets/4c34bae3-c18e-4477-8dc1-0dd333642382)






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
