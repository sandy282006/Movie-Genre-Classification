from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

# Load features and target
X = pd.read_csv('data/features.csv')
y = pd.read_csv('data/target.csv')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Save the best model
import joblib
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'models/random_forest_model.pkl')