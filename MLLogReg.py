from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np

df=pd.read_csv("out.csv")
print(df.head())
X = df.drop('Username', axis=1)  # features
             # target/class
y= df['Username']
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# parameters = {'C':[1, 5,10, 20, 50,75]}
# log_reg_model = LogisticRegression(max_iter=50000,penalty='l1',multi_class='ovr',class_weight='balanced',solver='liblinear')
# cv = GridSearchCV(log_reg_model, parameters)
# cv.fit(X_train, y_train)
# print(cv.best_params_)
# exit()
# Evaluate the models with the best C value (you would determine this using cross-validation)
best_C = 50# Example, you should find the optimal C
l1_model_best = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, random_state=42)
l1_model_best.fit(X_train_scaled, y_train)
y_pred1=l1_model_best.predict(X_test_scaled)
print(classification_report(y_test, y_pred1))
conf_matrix = confusion_matrix(y_test, y_pred1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=l1_model_best.classes_, yticklabels=l1_model_best.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

best_C = 75
l2_model_best = LogisticRegression(penalty='l2', C=best_C, random_state=42)
l2_model_best.fit(X_train_scaled, y_train)
y_pred2=l1_model_best.predict(X_test_scaled)
print(classification_report(y_test, y_pred2))
conf_matrix = confusion_matrix(y_test, y_pred2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=l2_model_best.classes_, yticklabels=l2_model_best.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
