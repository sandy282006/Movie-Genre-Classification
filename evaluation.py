from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd

# Load the test data and model
X_test = pd.read_csv('data/features.csv')
y_test = pd.read_csv('data/target.csv')
model = joblib.load('models/random_forest_model.pkl')

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))