import pandas as pd
import warnings

# warnings非表示
warnings.filterwarnings('ignore')

df = pd.read_csv('data/wdbc.data', header=None)
print("head()\n", df.head())
print("shape()", df.shape)

from sklearn.preprocessing import LabelEncoder

# 特徴量
X = df.loc[:, 2:].values
# クラスラベル
y = df.loc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)
print("classes", le.classes_)
print("transform", le.transform(['M', 'B']))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     test_size = 0.20,
                     stratify=y,
                     random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, solver='lbfgs'))

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)

print("Test Accuracy: %.3f" % pipe_lr.score(X_test, y_test))

# -----------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print("Fold: %2d, class dist.: %s, Acc: %.3f" % 
          (k+1, np.bincount(y_train[train]), score))

print("\nCV accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))

# -----------------------------------------------------------------------

from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv = 10,
                         n_jobs=1)
print("\nCV accuracy scores:\n %s" % scores)
print("CV accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))

# -----------------------------------------------------------------------

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty="l2",
                                           random_state=1,
                                           solver="lbfgs",
                                           max_iter=10000))

train_sizes, train_scores, test_scores = \
    learning_curve(estimator=pipe_lr,
                   X=X_train,
                   y=y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=10,
                   n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color="blue", marker="o",
         markersize=5, label="Training accuracy")

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color="green", linestyle="--",
         marker="s", markersize=5,
         label="Validation accuracy")

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel("Number of training examples")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.ylim([0.8, 1.03])
plt.tight_layout()

# plt.savefig("images/learning_curve.png", dpi=300)
plt.show()

# -----------------------------------------------------------------------

from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name='logisticregression__C',
    param_range=param_range,
    cv=10
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color="blue", marker="o",
         markersize=5, label="Training accuracy")

plt.fill_between(param_range,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(param_range, test_mean,
         color="green", linestyle="--",
         marker="s", markersize=5,
         label="Validation accuracy")

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xscale("log")
plt.legend(loc="lower right")
plt.xlabel("Parameter C")
plt.ylabel("Accuracy")
plt.ylim([0.8, 1.0])
plt.tight_layout()

# plt.savefig("images/validation_curve.png", dpi=300)
plt.show()

# -----------------------------------------------------------------------

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{"svc__C": param_range,
               "svc__kernel": ['linear']},
              {"svc__C": param_range,
               "svc__gamma": param_range,
               "svc__kernel": ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)

print("\nbest_score_", gs.best_score_)
print("best_params_", gs.best_params_)

clf = gs.best_estimator_
print("test accuracy: %.3f" % clf.score(X_test, y_test))

# -----------------------------------------------------------------------

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                        np.std(scores)))

from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                        np.std(scores)))

# -----------------------------------------------------------------------

from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print("\nconfmat\n", confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()

# plt.savefig("images/confusion_matrix.png", dpi=300)
plt.show()

# -----------------------------------------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score

print("\nPrecision: %3f" % precision_score(y_true=y_test, y_pred=y_pred))
print("Recall: %3f" % recall_score(y_true=y_test, y_pred=y_pred))
print("F1: %3f" % f1_score(y_true=y_test, y_pred=y_pred))

# -----------------------------------------------------------------------

from sklearn.metrics import make_scorer

scorer = make_scorer(f1_score, pos_label=0)

c_gamma_range = [0.01, 0.1, 1.0, 10.0]

param_grid = [{"svc__C": param_range,
               "svc__kernel": ['linear']},
              {"svc__C": param_range,
               "svc__gamma": param_range,
               "svc__kernel": ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)

print("\nbest_score_", gs.best_score_)
print("best_params_", gs.best_params_)

# -----------------------------------------------------------------------

from sklearn.metrics import roc_curve, auc
from distutils.version import LooseVersion as Version
from scipy import __version__ as scipy_version

if scipy_version >= Version('1.4.1'):
    from numpy import interp
else:
    from scipy import interp

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty='l2',
                                           random_state=1,
                                           solver='lbfgs',
                                           C=100.0))

pipe_svc = make_pipeline(StandardScaler(),
                         PCA(n_components=2),
                         SVC(random_state=1,
                             kernel='linear',
                             probability=True,
                             C=0.1))

# これ何？
X_train2 = X_train[:, [4, 14]]

cv = list(StratifiedKFold(n_splits=3).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],
                         y_train[train]).predict_proba(X_train2[test])
    
    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr,
             tpr,
             label='ROC fold %d (area = %0.2f)'
                   % (i+1, roc_auc))

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='Rondom guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='Perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc='lower right')

plt.tight_layout()
# plt.savefig('images/ROC_curve_LR.png', dpi=300)
# plt.savefig('images/ROC_curve_SVM.png', dpi=300)
plt.show()

# -----------------------------------------------------------------------

pipe_lr = pipe_lr.fit(X_train2, y_train)
y_pred2 = pipe_lr.predict(X_test[:, [4, 14]])

from sklearn.metrics import roc_auc_score, accuracy_score

print('ROC AUC: %.3f' % roc_auc_score(y_true=y_test, y_score=y_pred2))
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred2))

pre_scorer = make_scorer(score_func=precision_score,
                         pos_label=1,
                         greater_is_better=True,
                         average='micro')



X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

y_pred = np.zeros(y_imb.shape[0])
print(np.mean(y_pred == y_imb) * 100)

print(X_imb)

# -----------------------------------------------------------------------

from sklearn.utils import resample

print('Number of class 1 examples before:', X_imb[y_imb == 1].shape[0])

# y_imb == 1 / y_imb == 0　を入れ替えるとダウンサンプリングも可能
X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)
            
print('Number of class 1 examples after:', X_upsampled.shape[0])

X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))

y_pred = np.zeros(y_bal.shape[0])
print(np.mean(y_pred == y_bal) * 100)
