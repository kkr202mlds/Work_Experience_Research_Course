# Ada Boost Regressor
```
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                        n_estimators=500, random_state=42)
ada.fit(X_train, y_train)
AdaBoostRegressor(base_estimator=DecisionTreeRegressor(criterion='mse',
                                                       max_depth=4,
                                                       max_features=None,
                                                       max_leaf_nodes=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       min_samples_leaf=1,
                                                       min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0,
                                                       presort=False,
                                                       random_state=None,
                                                       splitter='best'),
                  learning_rate=1.0, loss='linear', n_estimators=500,
                  random_state=42)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
print("MSE train: {0:.4f}, test: {1:.4f}".\
      format(mean_squared_error(y_train, y_train_pred),
             mean_squared_error(y_test, y_test_pred)))
MSE train: 4.6605, test: 13.4949
print("R^2 train: {0:.4f}, test: {1:.4f}".\
      format(r2_score(y_train, y_train_pred),
             r2_score(y_test, y_test_pred)))
R^2 train: 0.9470, test: 0.8189
Revisiting Feature Importance
13 features.

Are they all equally important?

Which features are more important?

Can scikit-learn help us with this?

According to AdaBoost
ada.feature_importances_
array([0.03104826, 0.00243815, 0.01083651, 0.00085548, 0.0371141 ,
       0.23589575, 0.00958776, 0.13743614, 0.01893706, 0.02188852,
       0.04185043, 0.02193283, 0.43017901])
df.columns
Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT'],
      dtype='object')
result = pd.DataFrame(ada.feature_importances_, df.columns)
result.columns = ['feature']
result.sort_values(by='feature', ascending=False)
feature
LSTAT	0.430179
RM	0.235896
DIS	0.137436
PTRATIO	0.041850
NOX	0.037114
CRIM	0.031048
B	0.021933
TAX	0.021889
RAD	0.018937
INDUS	0.010837
AGE	0.009588
ZN	0.002438
CHAS	0.000855
result.sort_values(by='feature', ascending=False).plot(kind='bar');
```
# TOO SEE MORE GO TO BELOW FILE
## Nonlinear_Relationships.ipynb
