import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import ExtraTreesClassifier

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")

train_data = pd.read_csv(TRAIN_DATA_PATH)
X_train, y_train = train_data.iloc[:,:-1], train_data.iloc[:,-1]

sc = StandardScaler()
X_tr = sc.fit_transform(X_train)

level_0 = list()
level_0.append(('RF', ExtraTreesClassifier(n_estimators=1000)))
level_0.append(('LR',LogisticRegression(max_iter=7000)))   
level_1 = LinearDiscriminantAnalysis()
model = StackingClassifier(estimators=level_0, final_estimator=level_1, cv=4)
model.fit(X_tr, y_train)

test_data = pd.read_csv(TEST_DATA_PATH)
X_te = sc.transform(test_data)
submission = model.predict(X_te)
submission = pd.DataFrame(submission)

submission.to_csv('submission.csv', header=['class'], index=False)

