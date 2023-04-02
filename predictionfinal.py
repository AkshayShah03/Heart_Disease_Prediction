import pandas as pd
import numpy as np
data = pd.read_csv("/home/akshay/Documents/semester4/machine learning/CIA2_AkshayShah_21011101016/Heart_Disease_Prediction.csv")
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data.drop(columns = ['Age','Sex','FBS over 120','EKG results','Max HR','Exercise angina','ST depression','Slope of ST','Number of vessels fluro','Thallium'])
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42529)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
train_convert = {"Absence":0,"Presence":1}
y_train = y_train.replace(train_convert)
test_convert = {"Absence":0,"Presence":1}
y_test = y_test.replace(test_convert)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.fit_transform(X_test)
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
pred = rf.predict(X_test)
print(pred)
import pickle

pickle.dump(pred,open('predictionfinal.pkl','wb'))
