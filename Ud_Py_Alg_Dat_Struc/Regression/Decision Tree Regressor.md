# Decision Tree Regressor
```
from sklearn.tree import DecisionTreeRegressor
X = df[['LSTAT']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
plt.figure(figsize=(10,8))
plt.scatter(X[sort_idx], y[sort_idx])
plt.plot(X[sort_idx], tree.predict(X[sort_idx]), color='k')

plt.xlabel('LSTAT')
plt.ylabel('MEDV');

Using max_depth of 5 led to overfitting. Let's try 2 instead.

tree = DecisionTreeRegressor(max_depth=2)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
plt.figure(figsize=(10,8))
plt.scatter(X[sort_idx], y[sort_idx])
plt.plot(X[sort_idx], tree.predict(X[sort_idx]), color='k')

plt.xlabel('LSTAT')
plt.ylabel('MEDV');
```
# TOO SEE MORE GO TO BELOW FILE
## In this link at Nonlinear_Relationships.ipynb in https://github.com/Kkumar-20/Certificate_Job/blob/main/Machine_Learning_Cert/Codes/Regression/Nonlinear_Relationships.ipynb
