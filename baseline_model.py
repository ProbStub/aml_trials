# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.svm import LinearSVC
from imblearn.under_sampling import NearMiss
from imblearn import over_sampling as os
from imblearn.pipeline import make_pipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix, precision_score,auc,roc_auc_score,roc_curve,recall_score
from sklearn.preprocessing import StandardScaler

random.seed(50)

# Importing the dataset
#dataset = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
trx_dataset = pd.read_csv('data/input/synthetic_transactions.csv')
alert_dataset = pd.read_csv('data/input/synthetic_alerts.csv')

trx_dataset['Timestamp'] = pd.to_datetime(trx_dataset['Timestamp'])
alert_dataset['Date'] = pd.to_datetime(alert_dataset['Date'])
dataset = pd.merge(trx_dataset, alert_dataset, on='AlertID', how='left')

#dataset.drop('nameOrig', axis=1, inplace=True)
#dataset.drop('nameDest', axis=1, inplace=True)
#dataset.drop('isFlaggedFraud', axis=1, inplace=True)

dataset['Outcome'] = dataset['Outcome'].replace({'Report': 1, 'Dismiss': 0})
dataset.drop('Date', axis=1, inplace=True)
dataset.drop('AlertID', axis=1, inplace=True)
dataset['Timestamp'] = pd.to_numeric(pd.to_datetime(dataset['Timestamp']))

# Note sampling only possible if all elements of trx in a single record/row
sample_dataframe = dataset.sample(n=100000)
X = sample_dataframe.iloc[:, :-1].values
y = sample_dataframe.iloc[:, 4].values
# Unsampled execution is very slow due to O(n^3) of some SVM operations
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 4].values

# Encoding categorical data
# Define the columns to be one-hot encoded
onehot_columns = [1, 2]

# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), onehot_columns) # drop="first" to avoid Dummy
    ],
    remainder='passthrough'  # Keep the remaining columns unchanged
)

# Apply the ColumnTransformer to the data
X = preprocessor.fit_transform(X)


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set

# Apply the sampling
ada = ADASYN()
X_resampled, y_resampled = ada.fit_resample(X_train, y_train)
count = np.unique(y_resampled, return_counts=True)

# Create a pipeline
pipeline4 = make_pipeline(ADASYN(),LinearSVC(random_state=1))
pipeline4.fit(X_train, y_train)
print(count)

# Classify and report the results
print(classification_report_imbalanced(y_test, pipeline4.predict(X_test)))

# Making the Confusion Matrix
cm = confusion_matrix(y_val, pipeline4.predict(X_val))
roc = roc_auc_score(y_val, pipeline4.predict(X_val))
fpr, tpr, thresholds = roc_curve(y_val, pipeline4.predict(X_val))
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('data/output/roc_curve.png')
