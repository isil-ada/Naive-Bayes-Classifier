import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

df = pd.read_csv("proje/airline_passenger_satisfaction.csv")

print("\n")
for i in df.columns : print(i)
print("\n")
print(df.head(3),"\n")

df = df.drop(columns=["ID"] , axis=1)

print(df.isnull().sum(),"\n")
print(df[["Arrival Delay"]].isnull().sum(),"\n")


imputer = SimpleImputer(missing_values=np.nan , strategy="mean")
df[["Arrival Delay"]] = imputer.fit_transform(df[["Arrival Delay"]])

print(df.isnull().sum(),"\n")
print(df.info(),"\n")


encoder = LabelEncoder()
for col in [0,2,3,4,22] : 
    df.iloc[:,col] = encoder.fit_transform(df.iloc[:,col])

df[["Gender","Customer Type","Type of Travel","Class","Satisfaction"]] = df[["Gender","Customer Type","Type of Travel","Class","Satisfaction"]].apply(pd.to_numeric)

print(df.head(3),"\n")
print(df.info(),"\n")


x = df.drop(columns=["Satisfaction"],axis=1)
y = df["Satisfaction"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

start_time_fit = time.perf_counter() 

gaussianModel = GaussianNB()
model = gaussianModel.fit(x_train,y_train)

end_time_fit = time.perf_counter() 
print(f"Trainning time of the model => {end_time_fit-start_time_fit} second \n")

print(f"Accuracy of the Model : {model.score(x_test,y_test)} \n")

start_time_pred = time.perf_counter()

y_pred = model.predict(x_test)

end_time_pred = time.perf_counter()
print(f"Prediction time of the model => {end_time_pred-start_time_pred} second \n")

matrix = confusion_matrix( y_pred=y_pred , y_true=y_test )
print(f"-Confusion Matrix- \n {matrix}")
print("\n length of y_test => " , len(y_test)) #confusion matrix i dogrulamak icin

plt.figure(figsize=(6.5,5))
sns.heatmap(matrix,fmt="d",cmap="Purples",annot=True,linewidths=2,linecolor="black")
plt.xlabel("Predicted Label",fontsize=17,fontweight="bold",fontfamily="Book Antiqua")
plt.ylabel("True Label",fontsize=17,fontweight="bold",fontfamily="Book Antiqua")
plt.title("CONFUSSION MATRIX",fontsize=20,fontweight="bold",fontfamily="Book Antiqua")
plt.show()

"""
test_sample = pd.DataFrame([[1,30,0,1,2,900,2,5,3,3,4,3,3,4,5,6,7,8,9,8,2,1]],columns=x.columns)

if model.predict(test_sample) == 1 :
    print(f"\n Result of the prediction : {model.predict(test_sample)} => satisfied \n")
else :
    print(f"\n Result of the prediction : {model.predict(test_sample)} => neutral or dissatisfied \n")"""