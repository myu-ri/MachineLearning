import warnings
warnings.simplefilter('ignore')

# ml-source_25-28_s2.py

# Exploring the Housing dataset
## Loading the Housing dataset into a data frame
import pandas as pd

df = pd.read_csv('./data/housing.data.txt', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

print(df.head())


## Visualizing the important characteristics of a dataset
# if you don't have mlxtend library, install it with "pip install mlxtend"
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

scatterplotmatrix(df[cols].values, figsize=(10, 8), 
                  names=cols, alpha=0.5)
plt.tight_layout()

# plt.savefig('images/scatterplot_matrix.png', dpi=300)
plt.show()


# Aggregated graph by correlation matrix
import numpy as np
from mlxtend.plotting import heatmap

cm = np.corrcoef(df[cols].values.T) # calculate Pearson's r
hm = heatmap(cm, row_names=cols, column_names=cols)

# plt.savefig('images/correlation_heatmap.png', dpi=300)
plt.show()

# --------------------------------------------------------------

# ml-source_25-28_s3.py

# Implementing an ordinary least squares linear regression model
## Solving regression for regression parameters with gradient descent
import numpy as np

# basic linear regression model (based on AdalineGD class)
class LinearRegressionGD(object):

    # initialize
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta                                # learning rate
        self.n_iter = n_iter                          # iterations

    # execute training
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])            # initialize weights
        self.cost_ = []                               # initialize cost function value

        for i in range(self.n_iter):
            output = self.net_input(X)                # calculate output of activation function
            errors = (y - output)                     # calculate error
            self.w_[1:] += self.eta * X.T.dot(errors) # update weight w_1 and later
            self.w_[0] += self.eta * errors.sum()     # update weight w_0
            cost = (errors**2).sum() / 2.0            # calculate cost function
            self.cost_.append(cost)                   # store the value of cost function
        return self

    # calculate total input
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # calculate predicted value
    def predict(self, X):
        return self.net_input(X)


# train a model to predict the objective variable MEDV 
# using the explanatory variable RM
X = df[['RM']].values
y = df['MEDV'].values

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionGD()
lr.fit(X_std, y_std)


# line graph plot of number of epochs versus cost
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
#plt.tight_layout()

# plt.savefig('images/epochs_vs_cost.png', dpi=300)
plt.show()


# a simple helper function that plots 
# a scatterplot of the training data and adds a regression line
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 


# plot number of rooms against house price
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')

# plt.savefig('images/rooms_against_price.png', dpi=300)
plt.show()


# calculate predictions for the outcome variable at the original scale [Price in $1000s]
from distutils.version import LooseVersion
import sklearn

num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)

if LooseVersion(sklearn.__version__) >= LooseVersion('0.23.0'):
    print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std[:, np.newaxis]).flatten())
else:
    print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))


# There is no need to update the intercept weights 
# if you are dealing with standardized variables, 
# as the y-axis intercept will always be 0. 
# Output the weights to see if this is the case.
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])


## Estimating the coefficient of a regression model via scikit-learn
from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)


lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')

# plt.savefig('images/rooms_against_price_via_sklearn.png', dpi=300)
plt.show()

# --------------------------------------------------------------

# ml-source_25-28_s4.py

# Fitting a robust regression model using RANSAC
from sklearn.linear_model import RANSACRegressor

# instantiate RANSAC model
ransac = RANSACRegressor(LinearRegression(), 
                         max_trials=100, 
                         min_samples=50, 
                         loss='absolute_loss', 
                         residual_threshold=5.0, 
                         random_state=0)

ransac.fit(X, y)


inlier_mask = ransac.inlier_mask_                     # get boolean value representing inlier
outlier_mask = np.logical_not(inlier_mask)            # get boolean value representing outlier

line_X = np.arange(3, 10, 1)                          # create an integer value from 3 to 9
line_y_ransac = ransac.predict(line_X[:, np.newaxis]) # calculate predicted value

# plot inliers
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white', 
            marker='o', label='Inliers')
# plot outliers
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white', 
            marker='s', label='Outliers')
# plot predicted values
plt.plot(line_X, line_y_ransac, color='black', lw=2)   
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')

# plt.savefig('images/rooms_against_price_via_sklearn_with_RANSAC.png', dpi=300)
plt.show()

# check the results of the linear regression line fit
print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)

# --------------------------------------------------------------

# ml-source_25-28_s5.py

# Evaluating the performance of linear regression models
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

"""
#import numpy as np
#import scipy as sp
#
#ary = np.array(range(100000))
"""

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('images/residual_plot.png', dpi=300)
plt.show()


# evaluation of linear regression models w/MSE&R^2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

# --------------------------------------------------------------

# ml-source_25-28_s6.py

# Using regularized methods for regression
# with Lasso regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
print(lasso.coef_)

print('LASSO MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('LASSO R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
print(ridge.coef_)

print('RIDGE MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('RIDGE R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5) 
elanet.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
print(elanet.coef_)

print('ELASTIC NET MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('ELASTIC NET R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))



"""
## initialization for each regularized method

### Ridge regression
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)

### LASSO regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)

### Elastic Net regression
from sklearn.linear_model import ElasticNet
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5) 
"""

# --------------------------------------------------------------

# ml-source_25-28_s7.py

# Turning a linear regression model into a curve - polynomial regression

X = np.array([258.0, 270.0, 294.0, 
              320.0, 342.0, 368.0, 
              396.0, 446.0, 480.0, 586.0])\
             [:, np.newaxis]

y = np.array([236.4, 234.4, 252.8, 
              298.6, 314.2, 342.2, 
              360.8, 368.0, 391.2,
              390.8])

from sklearn.preprocessing import PolynomialFeatures
# instantiate a linear regression (least squares) model class
lr = LinearRegression()
pr = LinearRegression()
# instantiate a class of quadratic polynomial features
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

# fit linear features
lr.fit(X, y)
# make a column vector with np.newaxis
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
# calculate predicted value
y_lin_fit = lr.predict(X_fit)

# fit quadratic features
pr.fit(X_quad, y)
# calculate the value of y with the quadratic formula
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# plot results for the scatterplot, the linear regression model, 
# and the polynomial regression model
plt.scatter(X, y, label='Training points')
plt.plot(X_fit, y_lin_fit, label='Linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='Quadratic fit')
plt.xlabel('Explanatory variable')
plt.ylabel('Predicted or known target values')
plt.legend(loc='upper left')

plt.tight_layout()

# plt.savefig('images/linear_fit_vs_quadratic_fit.png', dpi=300)
plt.show()


# calculate mean squared error (MSE) and R^2 as metrics
y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
print('Training MSE linear: %.3f, quadratic: %.3f' % (
        mean_squared_error(y, y_lin_pred),
        mean_squared_error(y, y_quad_pred)))
print('Training R^2 linear: %.3f, quadratic: %.3f' % (
        r2_score(y, y_lin_pred),
        r2_score(y, y_quad_pred)))


## Modeling nonlinear relationships in the Housing Dataset


X = df[['LSTAT']].values
y = df['MEDV'].values
regr = LinearRegression()

# create quadratic and cubic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# fit features, predict, and calculate coefficients
# 1次式
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

# fit quadratic features, predict, and calculate coefficients
# 2次式
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

# fit cubic features, predict, and calculate coefficients
# 3次式
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# plot results of each model
plt.scatter(X, y, label='Training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, 
         label='Linear (d=1), $R^2=%.2f$' % linear_r2, 
         color='blue', 
         lw=2, 
         linestyle=':')
plt.plot(X_fit, y_quad_fit, 
         label='Quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red', 
         lw=2,
         linestyle='-')
plt.plot(X_fit, y_cubic_fit, 
         label='Cubic (d=3), $R^2=%.2f$' % cubic_r2,
         color='green', 
         lw=2, 
         linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')

# plt.savefig('images/fit_to_LSTAT_MDEV_d123.png', dpi=300)
plt.show()


### Transforming the dataset

X = df[['LSTAT']].values
y = df['MEDV'].values

# transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# fit features
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# plot results using projected data
plt.scatter(X_log, y_sqrt, label='Training points', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='Linear (d=1), $R^2=%.2f$' % linear_r2, 
         color='blue', 
         lw=2)

plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='lower left')

plt.tight_layout()

# plt.savefig('images/fit_to_LSTAT_MDEV_transformed.png', dpi=300)
plt.show()


# Dealing with nonlinear relationships using random forests
## Decision tree regression
from sklearn.tree import DecisionTreeRegressor

X = df[['LSTAT']].values
y = df['MEDV'].values

# instantiate the decision tree regression model class: 
# specify the depth of the decision tree with max_depth
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

# argsort returns sorted indices, flatten returns a one-dimensional array
sort_idx = X.flatten().argsort()

# create scatterplots and regression lines with the lin_regplot function
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')

# plt.savefig('images/decision_tree_regression.png', dpi=300)
plt.show()


## Random forest regression
X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)

from sklearn.ensemble import RandomForestRegressor

# instantiate a class for random forest regression
forest = RandomForestRegressor(n_estimators=1000, 
                               criterion='mse', 
                               random_state=1, 
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

# output MSE (Mean Squared Error)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
# output R^2 (coefficient of determination)
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# plot predicted values and residuals
plt.scatter(y_train_pred,           # x-value of graph (predicted value)
            y_train_pred - y_train, # y-value (difference between predicted and trained values)
            c='steelblue',          # plot color
            edgecolor='white',      # plot line color
            marker='o',             # marker type
            s=35,                   # marker size
            alpha=0.9,              # transparency
            label='Training data')  # label letter
plt.scatter(y_test_pred,  
            y_test_pred - y_test, 
            c='limegreen',
            edgecolor='white',
            marker='s', 
            s=35,
            alpha=0.9,
            label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10, 50])
plt.tight_layout()

# plt.savefig('images/predicted_values_and_residuals_by_random_forest.png', dpi=300)
plt.show()
