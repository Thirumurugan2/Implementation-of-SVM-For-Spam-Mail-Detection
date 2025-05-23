# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```.py
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: THIRUMURUGAN R
RegisterNumber: 212223220118 
*/

import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data['v2'].values
y=data['v1'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train
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
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```

## Output:
## Encoding:
![image](https://github.com/user-attachments/assets/7152dd54-fda7-47c9-8e59-86103bc1932f)



## Head():
![image](https://github.com/user-attachments/assets/c5b5ffa4-ea38-4b70-b1a9-65a9cf8dc0b8)




## Info():
![image](https://github.com/user-attachments/assets/3a4a35f1-a541-4ca9-a147-6a9d8de75b46)

## isnull().sum():
![image](https://github.com/user-attachments/assets/32455f75-2836-4d15-9f94-e76a1b689a19)

## Prediction of y:
![image](https://github.com/user-attachments/assets/8e0f2c6a-6fe3-4029-8a81-f2c0444910f5)

## Accuracy:
![image](https://github.com/user-attachments/assets/54f57889-4ea2-4a41-89a0-7215c0dd7fc5)








## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
