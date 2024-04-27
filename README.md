# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter Notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SREEKUMAR S 
RegisterNumber: 212223240157 
*/
```
```c
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## Result output:
![281978212-1be2e57f-2501-41c0-862a-19bd02626dc6](https://github.com/charumathiramesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120204455/28a5795a-2580-433a-9443-e2f07c687b5e)


## data.head():
![Screenshot 2024-04-27 105606](https://github.com/guru14789/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151705853/aab424be-38f0-47ad-9df2-1244b4966cdc)

## data.info():
![Screenshot 2024-04-27 105612](https://github.com/guru14789/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151705853/383da979-16ee-4c25-a35e-c9edb7b0d164)

## Y_prediction value:
![Screenshot 2024-04-27 105620](https://github.com/guru14789/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151705853/3d3d1c76-2a63-4e9c-a558-1b75216fdb57)

 ## Accuracy value:
![Screenshot 2024-04-27 105625](https://github.com/guru14789/Implementation-of-SVM-For-Spam-Mail-Detection/assets/151705853/30f0f147-e396-4911-8922-d41da666e7ad)


 



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using Python programming.
