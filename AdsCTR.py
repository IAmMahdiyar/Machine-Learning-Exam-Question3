# Download Train.csv and Test.csv from https://www.kaggle.com/c/avazu-ctr-prediction/data

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd

n_rows = 100000
df = pd.read_csv("train.csv", nrows=n_rows)

Y = df["click"].values
X = df.drop(["click", "id", "hour", "device_id", "device_ip"], axis=1).values

n_click = (X==1).sum()
n_not_click = (X == 0).sum()
print(n_click / (n_click + n_not_click) * 100, "Percent Have Clicked")

n_train = int(n_rows * 0.9)
x_train = X[ :n_train]
y_train = Y[ :n_train]
x_test = X[n_train: ]
y_test = Y[n_train: ]

enc = OneHotEncoder(handle_unknown='ignore')
x_train_enc = enc.fit_transform(x_train)
x_test_enc = enc.transform(x_test)

param = {'max_depth': [3, 10, None]}
decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)

grid_search = GridSearchCV(decision_tree, param, n_jobs=-1, cv=3, scoring='roc_auc')

print("Fitting...")
grid_search.fit(x_train_enc, y_train)

print("The best Settings Are:", grid_search.best_params_)

best = grid_search.best_estimator_
proba = best.predict_proba(x_test_enc)[:, 1]
score = roc_auc_score(y_test, proba)
print("Score is", score)