import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import preprocess as prep
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import Pipeline
import time
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

X_train, y_train, X_test, y_test = prep.X_train2, prep.y_train2, prep.X_test2, prep.y_test2

accs = []
recalls = []
precisions = []

lr=LogisticRegression(solver='lbfgs', max_iter=100000)

def LR(X_train, y_train, X_test, y_test): #send vectorized
    print(X_train.shape)
    print(y_train.shape)
    lr.fit(X_train,y_train.ravel())
    pred = lr.predict(X_test)
    score = lr.score(X_test, y_test)
    return pred, round(score,3)

def sfs_lr(m):
    global precision
    global recall
    global sfs1
    sfs1 = SFS(lr,
                   k_features=m,
                   forward=True,
                   floating=False,
                   scoring='accuracy',
               n_jobs = -1)
    # now we can work with our selected features
    #.fit(X_train.iloc[:, feat_cols], y_train)
    print("m = " + "{:f}".format(m))
    pipe = Pipeline([('sfs1', sfs1),
                              ('LR', lr)])
    start = time.time()
    pipe.fit(X_train, y_train.ravel())
    stop = time.time()
    print(f"Training time: {stop - start}s")
    print(sfs1.k_feature_idx_)
    start = time.time()
    cc = pipe.predict(X_test)
    acc = pipe.score(X_test, y_test)
    stop = time.time()
    print(f"Valid time: {stop - start}s")
    print(acc)
    accs.append(acc)
    recall = recall_score(y_test, cc, average=None)
    print(recall)
    recalls.append(recall)
    precision = precision_score(y_test, cc, average=None)
    print(precision)
    precisions.append(precision)
    col = pipe.named_steps['sfs1'].k_feature_names_
    print(col)
    return col, acc
pred, sc = LR(X_train, y_train, X_test, y_test)
print(sc)
best_acc = 0
for m in range(1,6):
    print(y_train.shape)
    col, acc = sfs_lr(m)
    if acc > best_acc:
        best_acc = acc
        feat_cols = col

print("Our best accuracy = " + "{:f}".format(best_acc))
print("Features selected by forward sequential selection was: "
      f"{feat_cols}")