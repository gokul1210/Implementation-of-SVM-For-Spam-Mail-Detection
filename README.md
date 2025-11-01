# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Detect File Encoding: Use chardet to determine the dataset's encoding. 

2.Load Data: Read the dataset with pandas.read_csv using the detected encoding. 

3.Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum(). 

4.Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split. 

5.Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix. 

6.Train SVM Model: Fit an SVC model on the training data. 

7.Predict Labels: Predict test labels using the trained SVM model.

8.Evaluate Model: Calculate and display accuracy with metrics.accuracy_scor 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: GOUKL S
RegisterNumber:  212224230075
*/
```
```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result
```
```
import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
```
```
data.head()
```
```
data.info()
```
```
data.isnull().sum()
```
```
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train
```
```
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train
x_test
```
```
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:


<img width="638" height="53" alt="image" src="https://github.com/user-attachments/assets/ce3e6b44-849a-48f5-a4f7-0e5f76080e84" />

## data.head()
<img width="772" height="218" alt="image" src="https://github.com/user-attachments/assets/6491e7d4-7e9f-490c-9bce-5288d429999b" />

## data.info()
<img width="510" height="228" alt="image" src="https://github.com/user-attachments/assets/287c0f01-1e02-4c8c-867f-4796c6717fc0" />

## data.isnull().sum()
<img width="623" height="245" alt="image" src="https://github.com/user-attachments/assets/89049176-71a8-4140-b642-c62a57bf8198" />

## x_train
<img width="1221" height="133" alt="image" src="https://github.com/user-attachments/assets/1cc9000a-e48c-4b81-96d8-8a603292f282" />

## x_test
<img width="849" height="73" alt="image" src="https://github.com/user-attachments/assets/0054ff5e-1cca-4c22-96d8-45046f979255" />

## y_pred
<img width="1167" height="43" alt="image" src="https://github.com/user-attachments/assets/2d006383-a4de-465c-920e-a64f5f174abb" />

## accuracy
<img width="796" height="52" alt="image" src="https://github.com/user-attachments/assets/ea391306-c59b-48b4-a4ca-ad5f5f967f7f" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
