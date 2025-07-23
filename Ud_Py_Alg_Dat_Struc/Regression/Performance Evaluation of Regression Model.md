# Performance Evaluation of Regression Model
```
from sklearn.model_selection import train_test_split
df.head()
CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV
0	0.01	18.00	2.31	0	0.54	6.58	65.20	4.09	1	296.00	15.30	396.90	4.98	24.00
1	0.03	0.00	7.07	0	0.47	6.42	78.90	4.97	2	242.00	17.80	396.90	9.14	21.60
2	0.03	0.00	7.07	0	0.47	7.18	61.10	4.97	2	242.00	17.80	392.83	4.03	34.70
3	0.03	0.00	2.18	0	0.46	7.00	45.80	6.06	3	222.00	18.70	394.63	2.94	33.40
4	0.07	0.00	2.18	0	0.46	7.15	54.20	6.06	3	222.00	18.70	396.90	5.33	36.20
#X = df['LSTAT'].values.reshape(-1,1)
X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
```
## Method 1: Residual Analysis
```
plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='k')
plt.xlim([-10, 50])
plt.show()
```
## Method 2: Mean Squared Error (MSE)
```
The average value of the Sums of Squared Error cost function

Useful for comparing different regression models

For tuning parameters via a grid search and cross-validation

from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, y_train_pred)
19.326470203585725
mean_squared_error(y_test, y_test_pred)
33.44897999767653
```
## Method 3: Coefficient of Determination, 
```
SSE: Sum of squared errors

SST: Total sum of squares

from sklearn.metrics import r2_score
r2_score(y_train, y_train_pred)
0.7730135569264234
r2_score(y_test, y_test_pred)
0.5892223849182507
```
# What does a Near Perfect Model Looks like?
```
generate_random = np.random.RandomState(0)
x = 10 * generate_random.rand(1000)
y = 3 * x + np.random.randn(1000)
plt.figure(figsize = (10, 8))
plt.scatter(x, y);
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X_train.reshape(-1, 1), y_train)

y_train_pred = model.predict(X_train.reshape(-1, 1))
y_test_pred = model.predict(X_test.reshape(-1, 1))
Method 1: Residual Analysis
plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-3, xmax=33, lw=2, color='k')
plt.xlim([-5, 35])
plt.ylim([-25, 15])
plt.show()

Method 2: Mean Squared Error (MSE)
mean_squared_error(y_train, y_train_pred)
1.0397119909455284
mean_squared_error(y_test, y_test_pred)
0.9893267030825136
Method 3: Coefficient of Determination, 
r2_score(y_train, y_train_pred)
0.9864565043160682
r2_score(y_test, y_test_pred)
0.9875301836511529
```
# TOO SEE MORE GO TO BELOW FILE
## Boston_Housing_Price_Prediction.ipynb
- In this link at Downloading and Parsing Statsbomb Data In line 15 https://github.com/Kkumar-20/Data_Scientist_Intern/blob/main/Football_Analytics_with_Data_Revolution.ipynb
- In this link at Performance Measures in https://github.com/Kkumar-20/Certificate_Job/blob/main/Machine_Learning_Cert/Codes/Classification/MNIST(openml).ipynb
