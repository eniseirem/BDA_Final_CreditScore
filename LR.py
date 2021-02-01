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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from sklearn.metrics import plot_confusion_matrix


X_train, y_train, X_test, y_test = prep.X_train2, prep.y_train2, prep.X_test2, prep.y_test2
X_train2, y_train2, X_test2, y_test2 = prep.X_train3, prep.y_train3, prep.X_test3, prep.y_test3

print("fffffffffffffffffff")
print(X_train.shape)
print(X_train2.shape)
accs = []
recalls = []
precisions = []
feat_count = []

accs_tr = []
recalls_tr = []
precisions_tr = []
feat_count_tr = []

lr=LogisticRegression(solver='lbfgs', max_iter=100000)
def clear_lists():
    precisions.clear()
    recalls.clear()
    accs.clear()
    feat_count.clear()
def LR(X_train, y_train, X_test, y_test): #send vectorized
    print(X_train.shape)
    print(y_train.shape)
    lr.fit(X_train,y_train.ravel())
    pred = lr.predict(X_test)
    predtr = lr.predict(X_train)
    score = lr.score(X_test, y_test)
    scoretr = lr.score(X_train, y_train)
    return pred, predtr, round(score,3), round(scoretr,3)
def metric(X,Y,pred):
    print('Coefficient is:', lr.coef_)
    print('Intercept is:', lr.intercept_)

    num_data = X.shape[0]
    print('num_data::', num_data)  # number of records- 4
    Y_pred = pred
    mse = mean_squared_error(Y, Y_pred)
    rmse = math.sqrt(mse / num_data)
    rse = math.sqrt(mse / (num_data - 2))
    rsquare = lr.score(X, Y)
    mae = mean_absolute_error(Y, Y_pred)

    print('RSE=', rse)
    print('R-Square=', rsquare)
    print('rmse=', rmse)
    print('mae=', mae)

def heatmap_cm(rgr,X, y, acc, name):
    plot_confusion_matrix(rgr, X, y)
    plt.title('{0} Accuracy Score: {1}'.format(name,acc), size = 15)
    print("check")
    plt.show()
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
    X = pipe.named_steps['sfs1'].transform(X_train)
    acc1 = pipe.named_steps['sfs1'].k_score_
    acc1 = round(acc1,3)
    stop = time.time()
    print(f"Training time: {stop - start}s")
    tr = pipe.named_steps['LR'].predict(X)
    metric(X, y_train, tr)
    m = int(m)
    heatmap_cm(lr, X, y_train, acc1, "TRAIN FeatCount" + "{:f}".format(m))
    print(sfs1.k_feature_idx_)
    start = time.time()
    cc = pipe.predict(X_test)
    acc = pipe.score(X_test, y_test)
    stop = time.time()
    print(f"Valid time: {stop - start}s")
    X = pipe.named_steps['sfs1'].transform(X_test)
    metric(X, y_test, cc)
    print(round(acc,3))
    accs.append(round(acc,3))
    recall = recall_score(y_test, cc, average=None)
    print(recall)
    recalls.append(recall)
    precision = precision_score(y_test, cc, average=None)
    print(precision)
    precisions.append(precision)
    col = pipe.named_steps['sfs1'].k_feature_names_
    print(col)
    feat_count.append(m)
    acc = round(acc,2)
    heatmap_cm(lr, X, y_test, acc, "TEST FeatCount " + "{:f}".format(m))
    return col, acc
clear_lists()
pred, predtr, sc, sc_train = LR(X_train, y_train, X_test, y_test)
heatmap_cm(lr, X_train, y_train, sc, "Model 1 : LR Train")
metric(X_train, y_train, predtr)
heatmap_cm(lr, X_test, y_test, sc_train, "Model 1 : LR Test")
metric(X_test, y_test, pred)
clear_lists()
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

#%%%%

fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.plot(feat_count, recalls, label='Recall')
ax2.plot(feat_count, accs, label='Precision')
ax2.plot(feat_count, precisions, label='accuracy')
ax2.set_xlabel('Feature Count')
ax2.set_ylabel('Value')
ax2.legend(loc='center left')
fig2.show()
r_2=recalls
p_2= precisions
a_2 = accs
f_2 = feat_count

clear_lists()

X_train, y_train, X_test, y_test = X_train2, y_train2, X_test2, y_test2

pred, predtr, sc, sc_train = LR(X_train, y_train, X_test, y_test)
heatmap_cm(lr, X_train, y_train, sc, "Model 1 : LR Train")
metric(X_train, y_train, predtr)
heatmap_cm(lr, X_test, y_test, sc_train, "Model 1 : LR Test")
metric(X_test, y_test, pred)
clear_lists()
best_acc = 0
for m in range(1,10):
    print(y_train.shape)
    col, acc = sfs_lr(m)
    if acc > best_acc:
        best_acc = acc
        feat_cols = col

print("Our best accuracy = " + "{:f}".format(best_acc))
print("Features selected by forward sequential selection was: "
      f"{feat_cols}")

#%%%%

fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.plot(feat_count, recalls, label='Recall')
ax2.plot(feat_count, accs, label='Precision')
ax2.plot(feat_count, precisions, label='accuracy')
ax2.set_xlabel('Feature Count')
ax2.set_ylabel('Value')
ax2.legend(loc='center left')
fig2.show()
r_2=recalls
p_2= precisions
a_2 = accs
f_2 = feat_count
