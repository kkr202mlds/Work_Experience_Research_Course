# Robust Regression
Outlier Demo: http://digitalfirst.bfwpub.com/stats_applet/stats_applet_5_correg.html
```
df.head()
CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV
0	0.01	18.00	2.31	0	0.54	6.58	65.20	4.09	1	296.00	15.30	396.90	4.98	24.00
1	0.03	0.00	7.07	0	0.47	6.42	78.90	4.97	2	242.00	17.80	396.90	9.14	21.60
2	0.03	0.00	7.07	0	0.47	7.18	61.10	4.97	2	242.00	17.80	392.83	4.03	34.70
3	0.03	0.00	2.18	0	0.46	7.00	45.80	6.06	3	222.00	18.70	394.63	2.94	33.40
4	0.07	0.00	2.18	0	0.46	7.15	54.20	6.06	3	222.00	18.70	396.90	5.33	36.20
```
## RANdom SAmple Consensus (RANSAC) Algorithm
link = http://scikit-learn.org/stable/modules/linear_model.html#ransac-regression

Each iteration performs the following steps:

Select min_samples random samples from the original data and check whether the set of data is valid (see is_data_valid).

Fit a model to the random subset (base_estimator.fit) and check whether the estimated model is valid (see is_model_valid).

Classify all data as inliers or outliers by calculating the residuals to the estimated model (base_estimator.predict(X) - y) - all data samples with absolute residuals smaller than the residual_threshold are considered as inliers.

Save fitted model as best model if number of inlier samples is maximal. In case the current estimated model has the same number of inliers, it is only considered as the best model if it has better score.
```
X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor()
ransac.fit(X, y)
RANSACRegressor()
```
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
```
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
np.arange(3, 10, 1)
array([3, 4, 5, 6, 7, 8, 9])
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))
sns.set(style='darkgrid', context='notebook')
plt.figure(figsize=(12,8));
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.legend(loc='upper left')
plt.show()
```
```
ransac.estimator_.coef_
array([8.25634575])
ransac.estimator_.intercept_
-28.841305570723048
X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(0, 40, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))
sns.set(style='darkgrid', context='notebook')
plt.figure(figsize=(12,8));
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.legend(loc='upper right')
plt.show()
```
# TOO SEE MORE GO TO BELOW FILE
## Boston_Housing_Price_Prediction.ipynb
