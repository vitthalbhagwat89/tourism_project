import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
import xgboost as xgb

# Load splits (created by previous cell)
X_train = pd.read_csv('Xtrain.csv')
X_test = pd.read_csv('Xtest.csv')
y_train = pd.read_csv('ytrain.csv').squeeze()
y_test = pd.read_csv('ytest.csv').squeeze()

# Identify columns
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
  ('num', StandardScaler(), numeric_cols),
  ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# Initialize XGBoost Classifier without the deprecated use_label_encoder parameter
clf = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

pipe = Pipeline([
  ('pre', preprocessor),
  ('clf', clf)
])

param_grid = {
  'clf__n_estimators': [50, 100],
  'clf__max_depth': [3, 5]
}

gs = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1)

mlflow.set_experiment('tourism_project_experiment')
with mlflow.start_run(run_name='xgb_grid_workflow'):
  gs.fit(X_train, y_train)
  best = gs.best_estimator_
  print('Best params:', gs.best_params_)
  ypred = best.predict(X_test)
  acc = accuracy_score(y_test, ypred)
  print('\nTest accuracy:', acc)
  print('\nClassification report:\n', classification_report(y_test, ypred))

  # Log params and metrics
  mlflow.log_params(gs.best_params_)
  mlflow.log_metric('test_accuracy', float(acc))

  # Log the trained model with MLflow sklearn flavor
  mlflow.sklearn.log_model(best, artifact_path='model')

# Save the pipeline for deployment
os.makedirs('deployment', exist_ok=True)
joblib.dump(best, 'deployment/model.joblib')
print('\nSaved trained model to deployment/model.joblib')
print('Model training and registration complete.')
