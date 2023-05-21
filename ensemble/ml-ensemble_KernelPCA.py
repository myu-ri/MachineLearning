import warnings
warnings.filterwarnings('ignore')

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

# ラベルを付け直す → 評価5・6・７を除外する

y_new = []
X_new = []

for i in range(len(y)):
    if y[i] < 5:
        y_new.append('0')
        X_new.append(X[i])
    elif y[i] > 7:
        y_new.append('1')
        X_new.append(X[i])

y = y_new
X = X_new


# 各クラスラベルの個数を確認
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
print("\nclasses_after", le.classes_)

import numpy as np
print("Number of Instances_after", np.bincount(y))


# データセット分割

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     test_size = 0.20,
                     stratify=y,
                     random_state=1)

# --------------------------------------------------

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
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# clf1 = LR, clf2 = Tree, clf3 = KNN, clf4 = SVM
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

clf4 = SVC(kernel='rbf',
           C=1.0,
           random_state=1,
           gamma=0.20)

pipe1 = Pipeline([['sc', StandardScaler()],
                  ['kpca', KernelPCA(n_components=6,
                                     kernel='rbf')],
                  ['clf', clf1]])

pipe3 = Pipeline([['sc', StandardScaler()],
                  ['kpca', KernelPCA(n_components=6,
                                     kernel='rbf')],
                  ['clf', clf3]])

pipe4 = Pipeline([['sc', StandardScaler()],
                  ['kpca', KernelPCA(n_components=6,
                                     kernel='rbf')],
                  ['clf', clf4]])

clf_labels = ['Logistic regression', 'Decision tree', 'KNN', "SVM"]

print('\n10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3, pipe4], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
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
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

# チューニング可能なパラメータ
# print("\n", mv_clf.get_params())

# グリッドサーチ

from sklearn.model_selection import GridSearchCV

# try tuning the depth of the decision tree and the 
# inverse regularization parameter C of logistic regression
params = {'decisiontreeclassifier__max_depth': [1, 2],
          'pipeline-1__clf__C': [0.001, 0.1, 100.0],
          'pipeline-3__clf__C': [0.001, 0.1, 100.0],
          'pipeline-3__clf__gamma': [0.001, 0.1, 100.0]}

grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    #iid=False,
                    scoring='accuracy')
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

