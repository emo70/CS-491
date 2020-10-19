import pandas as pd

df = pd.read_csv("C:/Users/emeka/Drive/CS 491/InputData2.csv")
title = []
for i in range(10):
    f = "f" + str(i + 1)
    for j in range(18):
        title += [f + "v" + str(j + 1)]
for i in range(181, 197):
    title += ["MAV" + str(i - 180)]
for i in range(197, 357):
    title += ["MAVS" + str(i - 196)]
for i in range(357, 373):
    title += ["VAR" + str(i - 357)]
title += ["label", "label2"]
df.columns = title

from sklearn.model_selection import train_test_split
X = df[title[:-3]]
# title[-1] means whether that part is affected or unaffected
# title[-2] means whether that person is healthy or patient
Y = df[title[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))