import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

url= "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df=pd.read_csv(url)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=72)

X_train=X_train.to_numpy()
X_test=X_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

model=DecisionTreeClassifier()
model.fit(X_train,y_train)

print("the model is trained successfully")


prediction=model.predict(X_test)



accuracy=accuracy_score(y_test,prediction)

print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

#See the actual predictions vs true values
print("\n📋 First few predictions:")
for i in range(10):

    print(f"Actual: {y_test[i]}, Predicted: {prediction[i]}")
