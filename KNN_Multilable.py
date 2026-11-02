import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
import pickle

df=pd.read_csv('iris.csv')
df=df.drop(['Id'],axis=1)
print(df)
df['Species']=df['Species'].map({'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2}).astype(int)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
reg=KNeighborsClassifier()
reg.fit(X_train,y_train)

#Train Performance
y_train_predictions=reg.predict(X_train)
print(f"Train Confusion Matrix:{confusion_matrix(y_train,y_train_predictions)}")
print(f"Train Accuracy:{accuracy_score(y_train,y_train_predictions)}")
print(f"Train Classification Report:{classification_report(y_train,y_train_predictions)}")

#Test Performance
y_test_predictions=reg.predict(X_test)
print(f"Test Confusion Matrix:{confusion_matrix(y_test,y_test_predictions)}")
print(f"Test Accuracy:{accuracy_score(y_test,y_test_predictions)}")
print(f"Test Classification Report:{classification_report(y_test,y_test_predictions)}")

#Checking with individual values
big_leaf_length=2.3
big_leaf_width=2.1
small_leaf_length=1.1
small_leaf_width=1.3
outcome=reg.predict([[big_leaf_length,big_leaf_width,small_leaf_length,small_leaf_width]])
if outcome==0:
    print("Setosa")
elif outcome==1:
    print("Versicolor")
else:
    print(f"Virginica")

with open("Iris_Model.pkl","wb") as f:
    pickle.dump(reg,f)
with open("Iris_Model.pkl","rb") as p:
    m=pickle.load(p)

result=m.predict([[big_leaf_length,big_leaf_width,small_leaf_length,small_leaf_width]])
if result==0:
    print("Setosa")
elif outcome == 1:
    print("Versicolor")
else:
    print(f"Virginica")



