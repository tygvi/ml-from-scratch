import pandas as pd 
import numpy as np
df=pd.read_csv("../../datasets/sna.csv")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

scaler=StandardScaler()
encoder=LabelEncoder()

df["Gender"] = encoder.fit_transform(df["Gender"])

X=df.iloc[:,0:3].values
y=df.iloc[:,-1].values
knn.fit(X_train,y_train)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(accuracy_score(y_test,y_pred))

from collections import Counter
class Knn:
    def __init__(self,k):
        self.n_neighbors=k
        self.X_train=None
        self.y_train=None
    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
    def predict(self,X_test):
        y_pred=[]
        for i in X_test:
            #calculating distance with each training point
            distances=[]
            for j in self.X_train:
                distances.append(self.calculate_distance(i,j))
            n_neighbors=sorted(list(enumerate(distances)),key=lambda x:x[1])[0:self.n_neighbors]
            label=self.majority_count(n_neighbors)
            y_pred.append(label)
        return np.array(y_pred)
    def calculate_distance(self,point_A,point_B):
        return np.linalg.norm(point_A-point_B)
    def majority_count(self,neighbors):
        votes=[]
        for i in neighbors:
           votes.append(self.y_train[i[0]])
        votes=Counter(votes)
        return votes.most_common()[0][0]

my_knn = Knn(k=3)
my_knn.fit(X_train, y_train)

# Predict
my_pred = my_knn.predict(X_test)

# Accuracy
print("My KNN Accuracy:", accuracy_score(y_test, my_pred))