import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load Wine dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target


# Basic statistics
print(X.describe())

# Relationships visualization between features using pairplot
sns.pairplot(X)
plt.show()

# Streamlit UI to input feature values
st.title("Wine Classification App")
st.write("Enter the wine features to predict the wine type")


# Histogram display
X.hist(figsize=(12, 8))
plt.show()

# Display dataset on Streanlit
st.title("Wine Classification with Logistic Regression and Decision Tree")
st.write("### Wine Dataset Overview")
st.write("The dataset contains information about different wines.")


# Data training 80%-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check for missing values
print(X.isnull().sum())

# Logistic Regression Model
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

# Decision Tree Classifier Model
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)

# Model Evaluation
st.write("### Model Evaluation Results")

# Logistic Regression Accuracy and Report
st.write("#### Logistic Regression")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_logreg)}")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred_logreg))

# Decision Tree Accuracy and Report
st.write("#### Decision Tree Classifier")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_dtree)}")
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred_dtree))

# Plotting confusion matrix using Streamlit
st.write("### Confusion Matrix")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix for Logistic Regression
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
sns.heatmap(cm_logreg, annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title('Confusion Matrix: Logistic Regression')

# Confusion Matrix for Decision Tree
cm_dtree = confusion_matrix(y_test, y_pred_dtree)
sns.heatmap(cm_dtree, annot=True, fmt="d", cmap="Blues", ax=ax[1])
ax[1].set_title('Confusion Matrix: Decision Tree')

st.pyplot(fig)

# Display Decision Tree Plot
st.write("### Decision Tree Visualization")
fig2 = plt.figure(figsize=(15, 10))

plot_tree(dtree, feature_names=data.feature_names, class_names=data.target_names, filled=True)

st.pyplot(fig2)
