from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                            r2_score, accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score, 
                            confusion_matrix)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                            GradientBoostingRegressor, GradientBoostingClassifier)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import time
import joblib

class ModelTrainer:
    def __init__(self, X, y, problem_type):
        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.models = self._initialize_models()
        self.results = {}
        
    def _initialize_models(self):
        """Initialize models based on problem type"""
        if self.problem_type == 'regression':
            return {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'Support Vector Machine': SVR(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor()
            }
        else:
            return {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Support Vector Machine': SVC(probability=True),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'XGBoost': XGBClassifier()
            }
    
    def train_models(self, selected_models, test_size=0.2, random_state=42, tune=False):
        """Train and evaluate selected models"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        for model_name in selected_models:
            start_time = time.time()
            model = self.models[model_name]
            
            if tune:
                model = self._hyperparameter_tuning(model, model_name, X_train, y_train)
            
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            
            metrics = self._calculate_metrics(y_test, y_pred)
            
            self.results[model_name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'train_time': train_time,
                'feature_importance': self._get_feature_importance(model, model_name)
            }
        
        return self.results
    
    def _calculate_metrics(self, y_test, y_pred):
        """Calculate evaluation metrics based on problem type"""
        if self.problem_type == 'regression':
            return {
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred)
            }
        else:
            return {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1': f1_score(y_test, y_pred, average='weighted'),
                'AUC': roc_auc_score(y_test, y_pred) if len(np.unique(self.y)) == 2 else None
            }
    
    def _get_feature_importance(self, model, model_name):
        """Get feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            return dict(zip(self.X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                return dict(zip(self.X.columns, model.coef_))
            else:
                return {f'Class_{i}': dict(zip(self.X.columns, coef)) 
                       for i, coef in enumerate(model.coef_)}
        return None
    
    def _hyperparameter_tuning(self, model, model_name, X_train, y_train):
        """Basic hyperparameter tuning"""
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20]
            },
            'XGBoost': {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            },
            'Support Vector Machine': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name], cv=3, n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        
        return model
    
    def save_model(self, model_name, filepath):
        """Save trained model to file"""
        if model_name in self.results:
            joblib.dump(self.results[model_name]['model'], filepath)
            return True
        return False
    
    def load_model(self, filepath):
        """Load trained model from file"""
        return joblib.load(filepath)