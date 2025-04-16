import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Step 1: Load and preprocess the dataset

# Assuming the dataset is in a CSV file named 'dataset.csv'

df = pd.read_csv('enhanced_dataset.csv')

# Check for missing values

missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Drop rows with missing values

df.dropna(inplace=True)

# Convert categorical variables to numerical using one-hot encoding

df = pd.get_dummies(df, columns=['categorical_variable'])

# Step 2: Split the dataset into training and testing sets


X = df.drop('target_variable', axis=1)
y = df['target_variable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the logistic regression model


model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the model's performance


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)

# Step 5: Generate ROC curve and AUC score


fpr, tpr, _ = roc_curve(y_test, y_pred)

auc_score = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()