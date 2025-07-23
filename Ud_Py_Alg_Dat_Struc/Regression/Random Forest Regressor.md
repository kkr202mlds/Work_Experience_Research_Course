# Random Forest Regressor
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
X = df.values
y = df['MEDV'].values                                             random_state=42)

NameError: name 'X' is not defined
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=500, criterion='mse',
                               random_state=42, n_jobs=-1)
forest.fit(X_train, y_train)
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                      oob_score=False, random_state=42, verbose=0,
                      warm_start=False)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print("MSE train: {0:.4f}, test: {1:.4f}".\
      format(mean_squared_error(y_train, y_train_pred),
             mean_squared_error(y_test, y_test_pred)))
MSE train: 1.8700, test: 9.3921

print("R^2 train: {0:.4f}, test: {1:.4f}".\
      format(r2_score(y_train, y_train_pred),
             r2_score(y_test, y_test_pred)))
R^2 train: 0.9787, test: 0.8740
```
## According to Random Forest
```
forest.feature_importances_
array([0.03213849, 0.00166784, 0.00677542, 0.00131141, 0.01508521,
       0.4038535 , 0.01414013, 0.06280815, 0.00467201, 0.0120068 ,
       0.01833147, 0.0127986 , 0.41441098])
result = pd.DataFrame(forest.feature_importances_, df.columns)
result.columns = ['feature']
result.sort_values(by='feature', ascending=False).plot(kind='bar');
```
# TOO SEE MORE GO TO BELOW FILE
## Nonlinear_Relationships.ipynb
