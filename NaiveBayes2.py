import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

class MyGaussianNB :
    def __init__(self):
        self.class_labels = None
        self.means = {}
        self.variances = {}
        self.priors = {}
    
    def fit(self,x,y) :
        self.class_labels = np.unique(y)
        x = x.astype(float)

        for label in self.class_labels :
            x_sample = x[y==label]
            self.means[label] = np.mean(x_sample,axis=0)
            self.variances[label] = np.var(x_sample,axis=0) + 1e-9 #to avoid division by zero error
            self.priors[label] = np.log(np.sum(y == label)/len(y))
    
    def predict(self,x) :
        x = x.astype(float)
        probabilities = []

        for label in self.class_labels :
            mean = self.means[label]
            variance = self.variances[label] 
            likelihood = -0.5*np.sum(np.log(2*np.pi*variance)) -0.5*np.sum((x-mean)**2/variance,axis=1)
            probabilities.append(likelihood+self.priors[label])

        return pd.Series(self.class_labels[np.argmax(np.array(probabilities),axis=0)])
    
    def score(self,x,y) :
        y_pred = self.predict(x)
        correct_predictions = (y_pred.to_numpy() == y.to_numpy())
        return np.sum(correct_predictions) / len(y)
    
#Load dataset
df = pd.read_csv("proje/airline_passenger_satisfaction.csv")

print("\n")
for i in df.columns : print(i)
print("\n")
print(df.head(3),"\n")

#Data preprocessing
df = df.drop(columns=["ID"] , axis=1)

#Handle missing values
print(df.isnull().sum(),"\n")
print(df[["Arrival Delay"]].isnull().sum(),"\n")
imputer = SimpleImputer(missing_values=np.nan , strategy="mean")
df[["Arrival Delay"]] = imputer.fit_transform(df[["Arrival Delay"]])
print(df.isnull().sum(),"\n")
print(df.info(),"\n")

#Label encoding
encoder = LabelEncoder()
for col in [0,2,3,4,22] : 
    df.iloc[:,col] = encoder.fit_transform(df.iloc[:,col])

df = df.apply(pd.to_numeric, errors="coerce")

print(df.head(3),"\n")
print(df.info(),"\n")

#Split dataset into training and testing sets
x = df.drop(columns=["Satisfaction"],axis=1)
y = df["Satisfaction"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Fit model
start_time_fit = time.perf_counter() 
gaussianModel = MyGaussianNB()
gaussianModel.fit(x_train,y_train)
end_time_fit = time.perf_counter() 
print(f"Trainning time of the model => {end_time_fit-start_time_fit} second \n")

#Calculate accuracy of the model
print(f"Accuracy of the Model : {gaussianModel.score(x_test,y_test)} \n")

#Prediction
start_time_pred = time.perf_counter()
y_pred = gaussianModel.predict(x_test)
end_time_pred = time.perf_counter()
print(f"Prediction time of the model => {end_time_pred-start_time_pred} second \n")

#Confusion matrix
matrix = confusion_matrix( y_pred=y_pred , y_true=y_test )
print(f"-Confusion Matrix- \n {matrix}")
print("\n length of y_test => " , len(y_test)) #confusion matrix i dogrulamak icin

plt.figure(figsize=(6.5,5))
sns.heatmap(matrix,fmt="d",cmap="Purples",annot=True,linewidths=2,linecolor="black")
plt.xlabel("Predicted Label",fontsize=17,fontweight="bold",fontfamily="Book Antiqua")
plt.ylabel("True Label",fontsize=17,fontweight="bold",fontfamily="Book Antiqua")
plt.title("CONFUSSION MATRIX",fontsize=20,fontweight="bold",fontfamily="Book Antiqua")
plt.show()

