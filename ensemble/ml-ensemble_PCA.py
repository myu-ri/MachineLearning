import warnings
warnings.filterwarnings('ignore')

# ToDoリスト
# 閾値を変えてみる　6のところで分けてみるとか


# データダウンロード
import pandas as pd

df = pd.read_csv("winequality-white.csv", sep = ";", encoding = "utf-8")
print(df)

# 品質データごとにグループ分けして数を数える
count_data = df.groupby('quality')["quality"].count()
print(count_data)

# データセットの分割
from sklearn.model_selection import train_test_split

# 特徴量・ラベル
X, y = df.iloc[:, :11].values, df.iloc[:, 11].values

# if ~ class

# ラベルを付け直す


newlabel = []
for v in list(y):
    if v <= 5:
        newlabel += [0] 
    else:
        newlabel += [1]
y = newlabel

# 各クラスラベルの個数を確認
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
print("\nclasses_after", le.classes_)

import numpy as np
print("Number of Instances_after", np.bincount(y))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     test_size = 0.20,
                     stratify=y,
                     random_state=1)

# --------------------------------------------------

# PCAによる次元圧縮と描画
# データを標準化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)


# scikit-learnでの実装

"""
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA()

# 訓練データを使ってモデルを適合・変換
# 分散説明率の算出
X_train_pca = pca.fit_transform(X_train_std)
print('explained_variance_ratio_\n', pca.explained_variance_ratio_)


# 累積寄与率のプロット
# 分散説明率の累積和のプロット
plt.bar(range(1, 12), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 12), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.savefig('images-PCA/variance_explained_ratio.png', dpi=300)
plt.show()

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')

plt.savefig('images-PCA/PCA_subspace.png', dpi=300)
plt.show()

"""


# アンサンブルモデル
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator, 
                             ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'} (default='classlabel')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.

    """
    def __init__(self, classifiers, vote='classlabel', weights=None):

        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value
                                  in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Matrix of training examples.

        y : array-like, shape = [n_examples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError("vote must be 'probability' or 'classlabel'"
                             "; got (vote=%r)"
                             % self.vote)

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d classifiers'
                             % (len(self.weights), len(self.classifiers)))

        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Matrix of training examples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_examples]
            Predicted class labels.
            
        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote

            #  Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X)
                                      for clf in self.classifiers_]).T

            maj_vote = np.apply_along_axis(
                                      lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_examples, n_classes]
            Weighted average probability for each class per example.

        """
        probas = np.asarray([clf.predict_proba(X)
                             for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
            return out

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# clf1 = LR, clf2 = Tree, clf3 = KNN, clf4 = svm
clf1 = LogisticRegression(penalty='l2', 
                          C=0.001,
                          solver='lbfgs',
                          random_state=1)

clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)

clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')

clf4 = NuSVC(random_state=1, nu = 0.25)


pipe1 = Pipeline([['sc', StandardScaler()],
                  ['pca', PCA(n_components=2)],
                  ['clf', clf1]])

pipe3 = Pipeline([['sc', StandardScaler()],
                  ['pca', PCA(n_components=2)],
                  ['clf', clf3]])

pipe4 = Pipeline([['sc', StandardScaler()],
                  ['pca', PCA(n_components=2)],
                  ['clf', clf4]])

"""
pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])

pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])

pipe4 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf4]])
"""

# 変更箇所　3か所＋1か所
clf_labels = ['Logistic regression', 'Decision tree', 'KNN', "SVM"]
# clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

print('\n10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3, pipe4], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='accuracy')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

print("-------------------------------------------")

# Majority Vote
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3, pipe4])

clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, pipe4, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='accuracy')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

# ROC曲線
"""
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls \
        in zip(all_clf,
               clf_labels, colors, linestyles):

    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train,
                     y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                     y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,
             color=clr,
             linestyle=ls,
             label='%s (auc = %0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2)

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.savefig('images-PCA/ROC_curve.png', dpi=300)
plt.show()

"""



# チューニング可能なパラメータ
# print("\n", mv_clf.get_params())

# グリッドサーチ

"""
from sklearn.model_selection import GridSearchCV

# try tuning the depth of the decision tree and the 
# inverse regularization parameter C of logistic regression
params = {'decisiontreeclassifier__max_depth': [1, 2],
          'pipeline-1__clf__C': [0.001, 0.1, 100.0]}

grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    #iid=False,
                    scoring='roc_auc')
grid.fit(X_train, y_train)

# perform 10-fold cross-validation after completing grid search
# output various hyperparameter value combinations and 
# the average value of the ROC curve
for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_['mean_test_score'][r], 
             grid.cv_results_['std_test_score'][r] / 2.0, 
             grid.cv_results_['params'][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)

"""

