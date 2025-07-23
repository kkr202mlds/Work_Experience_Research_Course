# Confusion Matrix
```
from sklearn.model_selection import cross_val_predic
y_train_pred = cross_val_predict(clf, X_train, y_train_0, cv=3)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_0, y_train_pred)

import pandas as pd
pd.DataFrame(confusion_matrix(y_train_0, y_train_pred))

pd.DataFrame(confusion_matrix(y_train_0, y_train_pred),
             columns=pd.MultiIndex.from_product([['Prediction'], ["Negative", "Positive"]]),
             index=pd.MultiIndex.from_product([["Actual"], ["Negative", "Positive"]]))
```
