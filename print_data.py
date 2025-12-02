from featureGenerator import *
from readToMatrix import *
import numpy as np
import re
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, roc_curve,confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sys



x1_ds2 = getMatrix("data/T4SE_train1502/result/negative/pssm_profile_uniref50")
x2_ds2 = getMatrix("data/T4SE_train1502/result/positive/pssm_profile_uniref50")
y = [-1 for i in range(x1.shape[0])]
y.extend([1 for i in range(x2.shape[0])])
y = np.array(y)
x = np.vstack((x1, x2))

test_x1 = getMatrix("data/T4SE_test180/result/negative/pssm_profile_uniref50")
test_x2 = getMatrix("data/T4SE_test180/result/positive/pssm_profile_uniref50")

test_x = np.vstack((test_x1, test_x2))
test_y = [-1 for i in range(test_x1.shape[0])]
test_y.extend([1 for i in range(test_x2.shape[0])])
test_y=np.array(test_y)
x_all=np.vstack((x,test_x))
x_all=Norm(x_all)
x = x_all[:1502, :]
test_x = x_all[1502:, :]

print(x)
print()
print(y)
print("ddd")
