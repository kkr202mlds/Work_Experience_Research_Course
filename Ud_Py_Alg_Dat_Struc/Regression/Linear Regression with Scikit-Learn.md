```
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.regplot()
sns.jointplot()
```
# Linear Regression with Scikit-Learn
```
df.head()
CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV
0	0.01	18.00	2.31	0	0.54	6.58	65.20	4.09	1	296.00	15.30	396.90	4.98	24.00
1	0.03	0.00	7.07	0	0.47	6.42	78.90	4.97	2	242.00	17.80	396.90	9.14	21.60
2	0.03	0.00	7.07	0	0.47	7.18	61.10	4.97	2	242.00	17.80	392.83	4.03	34.70
3	0.03	0.00	2.18	0	0.46	7.00	45.80	6.06	3	222.00	18.70	394.63	2.94	33.40
4	0.07	0.00	2.18	0	0.46	7.15	54.20	6.06	3	222.00	18.70	396.90	5.33	36.20
X = df['RM'].values.reshape(-1,1)
df['RM'].values
X
df['RM'].values.reshape(-1,1).shape
(506, 1)
y = df['MEDV'].values
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
model.coef_
array([9.10210898])
model.intercept_
-34.67062077643857
plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show();
/opt/anaconda3/lib/python3.8/site-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(

sns.jointplot(x='RM', y='MEDV', data=df, kind='reg', height=8);
plt.show();

model.predict(np.array([7]).reshape(1,-1))
array([29.04414209])
```
Below is from Jacob T. VanderPlas text, Python Data Science Handbook: Essential Tools for Working with Data

Basics of the API:

Most commonly, the steps in using the Scikit-Learn estimator API are as follows (we will step through a handful of detailed examples in the sections that follow):

Choose a class of model by importing the appropriate estimator class from Scikit- Learn.
Choose model hyperparameters by instantiating this class with desired values.
Arrange data into a features matrix and target vector.
Fit the model to your data by calling the fit() method of the model instance.
Apply the model to new data:
For supervised learning, often we predict labels for unknown data using the predict() method.
For unsupervised learning, we often transform or infer properties of the data using the transform() or predict() method.
```
# Step 1: Selecting a model

# Step 2: Instantiation
ml_2 = LinearRegression()

# Step 3: Arrange data
X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values

# Step 4: Model fitting
ml_2.fit(X, y)

# Step 5: Predict
ml_2.predict(np.array([15]).reshape(1,-1))
array([20.30310057])
plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show();

sns.jointplot(x='LSTAT', y='MEDV', data=df, kind='reg', height=8);
plt.show();
```
# TOO SEE MORE GO TO BELOW FILE
## Boston_Housing_Price_Prediction.ipynb
