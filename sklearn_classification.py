from sklearn import datasets
import numpy as np
import warnings

# warnings非表示
warnings.filterwarnings('ignore')

# 2. sklearn perceptron
# データセットの読み込み
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class Labels:', np.unique(y))

# 訓練・テストデータセットに分類
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1, stratify = y)

print('Labels count in y:', np.bincount(y))
print('Labels count in y_train:', np.bincount(y_train))
print('Labels count in y_test:', np.bincount(y_test))

# 特徴量のスケーリング
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# パーセプトロンの訓練
from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

# predict method を使用した予測
y_pred = ppn.predict(X_test_std)

print('Misclassfied example: %d' % (y_test != y_pred).sum())

# accuracy_score(metrics)
from sklearn.metrics import accuracy_score

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

# Perceptron decision_regions
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from distutils.version import LooseVersion

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 0].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    
    # highlight test examples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        if LooseVersion(matplotlib.__version__) < LooseVersion('0.3.4'):
            plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black',
                        alpha=1.0, linewidth=1, marker='o', s=100, label='test set')
        else:
            plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolor='black',
                        alpha=1.0, linewidth=1, marker='o', s=100, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))

plt.xlabel('Petal Length [std]')
plt.ylabel('Petal Width [std]')
plt.legend(loc='upper left')
plt.tight_layout()

# plt.savefig('images/decision_regions.png', dpi=300)
plt.show()

# 3. ロジスティック回帰を用いたクラス確立予測モデルの構築
# シグモイド曲線の表示と特性
import matplotlib.pyplot as plt
import numpy as np

# シグモイド関数の定義
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.yticks([0.0, 0.5, 1.0])

ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()

# plt.savefig('images/sigmoid.png', dpi=300)
plt.show()

# ロジスティクス関数の重みの学習
def cost_1(z):
    return - np.log(sigmoid(z))

def cost_0(z):
    return - np.log(1 - sigmoid(z))

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])

plt.ylabel('J(w)')
plt.xlabel('$\phi (z)$')
plt.legend(loc='best')
plt.tight_layout()

# plt.savefig('images/costs.png', dpi=300)
plt.show()

# ADALINE実装をロジスティック回帰のアルゴリズムに変換
class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# ロジスティック回帰の実装を確認
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)

plot_decision_regions(X=X_train_01_subset,
                      y=y_train_01_subset,
                      classifier=lrgd)

plt.xlabel('Petal Length [std]')
plt.ylabel('Petal Width [std]')

plt.legend(loc='upper left')

plt.tight_layout()

# plt.savefig('images/decision_regions.png', dpi=300)
plt.show()

# sklearn logistic training
from sklearn.linear_model import LogisticRegression

# Cの値が変更可能
lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))

plt.xlabel('Petal Length [std]')
plt.ylabel('Petal Width [std]')

plt.legend(loc='upper left')

plt.tight_layout()

# plt.savefig('images/decision_regions_sklearn.png', dpi=300)
plt.show()

# sklearnを使い、クラスの所属関係の確立を求める
print(lr.predict_proba(X_test_std[:3, :]))
print(lr.predict_proba(X_test_std[:3, :]).sum(axis=1))

print(lr.predict_proba(X_test_std[:3, :]).argmax(axis=1))
print(lr.predict(X_test_std[:3, :]))
print(lr.predict(X_test_std[0, :].reshape(1, -1)))

# 正則化による過学習への対処
weights, params = [], []

for c in np.arange(-5, 5):
    lr = LogisticRegression(C = 10.**c, random_state=1,
                            solver = 'lbfgs',
                            multi_class='ovr') 
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

weights = np.array(weights)

plt.plot(params, weights[:, 0], label = 'Petal Length')
plt.plot(params, weights[:, 1], linestyle='--', label = 'Petal Width')

plt.xlabel('C')
plt.ylabel('weight coefficient')

plt.legend(loc='upper left')
plt.xscale('log')

# plt.savefig('images/regularrization_parameters.png', dpi=300)
plt.show()


# 4. サポートベクターマシンによる最大マージン分類
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))

plt.xlabel('Petal Length [std]')
plt.ylabel('Petal Width [std]')

plt.legend(loc='upper left')

plt.tight_layout()

# plt.savefig('images/decision_regions.png', dpi=300)
plt.show()

# 5. カーネルSVMを用いた非線形問題の求解
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)

y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c = 'b', marker='X',
            label = '1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c = 'r', marker='s',
            label = '-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()

# plt.savefig('images/XOR_Data.png', dpi=300)
plt.show()

# カーネルトリックを用いた高次元空間での分離平面特定
# カーネルSVMの訓練を行う

svm = SVC(kernel='rbf', C=10.0, random_state=1, gamma=0.10)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()

# plt.savefig('images/Kernel_trick.png', dpi=300)
plt.show()

svm = SVC(kernel='rbf', C=1.0, random_state=1, gamma=0.20)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))

plt.xlabel('Petal Length [std]')
plt.ylabel('Petal Width [std]')

plt.legend(loc='upper left')

plt.tight_layout()

# plt.savefig('images/decision_regions.png', dpi=300)
plt.show()

svm = SVC(kernel='rbf', C=1.0, random_state=1, gamma=100.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))

plt.xlabel('Petal Length [std]')
plt.ylabel('Petal Width [std]')

plt.legend(loc='upper left')

plt.tight_layout()

# plt.savefig('images/KernelSVM_Iris.png', dpi=300)
plt.show()

# 6. 決定木学習
# 情報利得の最大化
import matplotlib.pyplot as plt
import numpy as pd

def gini(p):
    return p * (1 - p) + (1 - p) * (1-(1-p))

def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)

for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                          ['Entropy', 'Entropy(scaled)',
                           'Gini impurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')

plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('impurity index')

# plt.savefig('images/impurity.png', dpi=300, bbox_inches='tight')
plt.show()

# 決定木の構築
# 特徴空間を矩形に分割することで複雑な決定境界を構築
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(criterion='gini',
                                    max_depth=4,
                                    random_state=1)

tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined,
                      classifier=tree_model, test_idx=range(105, 150))

plt.xlabel('Petal Length [std]')
plt.ylabel('Petal Width [std]')

plt.legend(loc='upper left')

plt.tight_layout()

# plt.savefig('DecisionTree_images_Base/decision_regions.png', dpi=300)
plt.show()

# 訓練後の決定木モデルを可視化
from sklearn import tree

tree.plot_tree(tree_model)
# plt.savefig('images/visualized_decision_tree.png', dpi=300)
plt.show()

# 可視化をさらに改善
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree_model,
                           filled=True,
                           class_names=['Setosa',
                                        'Versicolor',
                                        'Verginica'],
                           feature_names=['Petal Length',
                                          'Petal Width'],
                           out_file=None)
graph = graph_from_dot_data(dot_data)
# graph.write_png('images/tree.png')

# ランダムフォレスト分類器の構築
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('Petal Length [std]')
plt.ylabel('Petal Width [std]')

plt.legend(loc='upper left')

plt.tight_layout()

# plt.savefig('images/decision_regions_combined.png', dpi=300)
plt.show()

# 7. k（最）近傍法
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5,
                           p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('Petal Length [std]')
plt.ylabel('Petal Width [std]')

plt.legend(loc='upper left')

plt.tight_layout()

# plt.savefig('images/decision_regions.png', dpi=300)
plt.show()