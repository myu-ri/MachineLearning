# Dealing with missing data

## Identifying missing values in tabular data
from calendar import c
from re import M, S
import pandas as pd
from io import StringIO
import sys
# make sample data
csv_data = \
'''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
# if you are using Python 2.7
# you need to convert the string to unicode:
if (sys.version_info < (3, 0)):
    csv_data = unicode(csv_data)
# load sample data
df = pd.read_csv(StringIO(csv_data))
print(df)

# count missing values in each feature
print('df.isnull()\n', df.isnull().sum())

# acces the underlying Numpy array
# via the 'values' attribute
print('df.values\n', df.values)

## Eliminating training examples or features with missing values
# remove rows/columns that contai missing values
print("axis=0\n", df.dropna(axis=0))
print("axis=1\n", df.dropna(axis=1))

print("how = 'all'\n", df.dropna(how='all'))
print("tresh=4\n", df.dropna(thresh=4))

print("subset=['C']\n", df.dropna(subset=['C']))


## Imputing missing values
#again: our original array
print('df.values\n', df.values)

# impute missing values via the column mean
from sklearn.impute import SimpleImputer
import numpy as np

# generate in instance for impute missing values with mean
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
# apply data
imr = imr.fit(df.values)
# execute imputation
imputed_data = imr.transform(df.values)
print("inputed_data\n", imputed_data)

print(df.fillna(df.mean()))

# Handling categorical data
## Nominal and ordinal features
import pandas as pd

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']

print("T-shirts\n", df)

## Mapping ordinal features
# generate a dictionary the maps T-shirt size to integers
size_mapping = {'XL':3,
                'L':2,
                'M':1}
# convert t-shirt size to integer
df['size'] = df['size'].map(size_mapping)
print("convert_int\n", df)

# If you want to return the integer velue
# to its ordinal string representation
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(df['size'].map(inv_size_mapping))

## Encoding class labels
import numpy as np
# create a mapping dictionary to convert class labels from strings to integers
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
# convert classlabels from strings to integer
df['classlabel'] = df['classlabel'].map(class_mapping)
print("convert_int\n", df)

## reverse the classlabel mapping
# create a dictionary to map integer and class label
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

## Label encoding with sklearn's LabelEncoder
from sklearn.preprocessing import LabelEncoder
# create an instance of labels encoder
class_le = LabelEncoder()

y = class_le.fit_transform(df['classlabel'].values)
print("y", y)

# reverse mapping
print("reverse mapping", class_le.inverse_transform(y))

## Performing one-hot encoding on nominal features
# extract color of t-shirt, size and price
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print("LE\n", X)

from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
print("OHE\n", color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())

# in case you want to selectively transform
# an array column consisting of multiple features
from sklearn.compose import ColumnTransformer
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]),
                              ('noting', 'passthrough', [1, 2])])
print(c_transf.fit_transform(X).astype(float))

# ohe via pandas
print(pd.get_dummies(df[['price', 'color', 'size']]))

print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))

# create an ohe
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([('onehot', color_ohe, [0]),
                              ('nothing', 'passthrough', [1, 2])])
print(c_transf.fit_transform(X).astype(float))


print("---------------------------------------------------")


# UCI machine learning repository
# df_wine = pd.read_csv('wine.data', header=None)
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
# show class labels
print('Class lavels', np.unique(df_wine['Class label']))

print('head\n', df_wine.head())

##split dataset
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# split dataset
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     test_size = 0.3,
                     random_state=0,
                     stratify=y)

## Bringing features onto the same scale
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
# scale the training data
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Avisual example:
ex = np.array([0, 1, 2, 3, 4, 5])

# standardize
print('standardized:', (ex -ex.mean()) / ex.std())

# normalize
print('normalized', (ex - ex.min()) / (ex.max() - ex.min()))

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

print('mean in X_train: ', X_train.mean())
print('variance in X_train: ', X_train.var())
print('mean in X_train_std: ', X_train_std.mean())
print('variance in X_train_std: ', X_train_std.var())

print('mean in X_test: ', X_test.mean())
print('variance in X_test: ', X_test.var())
print('mean in X_test_std: ', X_test_std.mean())
print('variance in X_test_std: ', X_test_std.var())

# Selecting meaningful features
# L1正則化
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')

# Applied to the standardized Wine data
lr.fit(X_train_std, y_train)
print('Training accuracy', lr.score(X_train_std, y_train))
print('Test accuracy', lr.score(X_test_std, y_test))

print(lr.intercept_)
print(lr.coef_)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

weights, params = [], []

for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear',
                            multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color = color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])

plt.ylabel('Weight coefficient')
plt.xlabel('C')

plt.xscale('log')
plt.legend(loc="upper left")
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)

# plt.savefig('images/regularization_pass_in_L1.png', dpi=300,
#             bbox_inches='tight', pad_inches=0.2)
plt.show()

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, 
                             test_size = self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self
    
    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')

plt.grid()
plt.tight_layout()

# plt.savefig('images/visualized_accuracy_of_kNN_with_SBS.png', dpi=300)
plt.show()

k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

knn.fit(X_train_std, y_train)
print('Training accuracy', knn.score(X_train_std, y_train))
print('Test accuracy', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy', knn.score(X_test_std[:, k3], y_test))

from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()

# plt.savefig('images/feature_importance_by_random_forest.png', dpi=300)
plt.show()

from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)

print('Number of features that meet this threshold criterion:',
      X_selected.shape[1])

for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

exit()