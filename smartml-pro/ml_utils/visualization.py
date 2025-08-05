import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

class ModelVisualizer:
    @staticmethod
    def plot_metrics_comparison(results):
        """Create interactive comparison of model metrics"""
        metrics_df = pd.DataFrame({
            model: data['metrics'] for model, data in results.items()
        }).T.reset_index().rename(columns={'index': 'Model'})
        
        # Add training time to metrics
        for model in results:
            metrics_df.loc[metrics_df['Model'] == model, 'Training Time'] = results[model]['train_time']
        
        melted_df = metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Value')
        
        fig = px.bar(melted_df, x='Model', y='Value', color='Metric',
                     barmode='group', title='Model Performance Comparison',
                     hover_data=['Value'])
        fig.update_layout(legend_title_text='Metrics')
        return fig
    
    @staticmethod
    def plot_predictions_vs_actual(y_test, y_pred, model_name):
        """Scatter plot of predicted vs actual values"""
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title=f'{model_name}: Predicted vs Actual')
        fig.add_shape(type='line', line=dict(dash='dash'),
                     x0=y_test.min(), y0=y_test.min(),
                     x1=y_test.max(), y1=y_test.max())
        return fig
    
    @staticmethod
    def plot_residuals(y_test, y_pred, model_name):
        """Plot residuals for regression models"""
        residuals = y_test - y_pred
        fig = px.scatter(x=y_pred, y=residuals,
                        labels={'x': 'Predicted', 'y': 'Residuals'},
                        title=f'{model_name}: Residual Plot')
        fig.add_shape(type='line', line=dict(dash='dash'),
                     x0=y_pred.min(), y0=0,
                     x1=y_pred.max(), y1=0)
        return fig
    
    @staticmethod
    def plot_feature_importance(feature_importance, model_name):
        """Plot feature importance if available"""
        if feature_importance is None:
            return None
            
        if isinstance(feature_importance, dict):
            # Single set of feature importances
            fi_df = pd.DataFrame({
                'Feature': list(feature_importance.keys()),
                'Importance': list(feature_importance.values())
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(fi_df.head(10), x='Importance', y='Feature',
                        title=f'{model_name}: Top Feature Importance')
        else:
            # Multiple sets (e.g., for multi-class)
            fig = make_subplots(rows=1, cols=len(feature_importance),
                              subplot_titles=[f'Class {i}' for i in range(len(feature_importance))])
            
            for i, (cls, imp) in enumerate(feature_importance.items()):
                fi_df = pd.DataFrame({
                    'Feature': list(imp.keys()),
                    'Importance': list(imp.values())
                }).sort_values('Importance', ascending=False).head(10)
                
                fig.add_trace(
                    go.Bar(x=fi_df['Importance'], y=fi_df['Feature'], orientation='h'),
                    row=1, col=i+1
                )
            
            fig.update_layout(title_text=f'{model_name}: Feature Importance by Class')
        
        return fig
    
    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, model_name):
        """Plot confusion matrix for classification"""
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       title=f'{model_name}: Confusion Matrix')
        return fig
    
    @staticmethod
    def plot_roc_curve(y_test, y_pred_proba, model_name):
        """Plot ROC curve for binary classification"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                               mode='lines',
                               name=f'ROC curve (area = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                               mode='lines',
                               line=dict(dash='dash'),
                               name='Random'))
        
        fig.update_layout(title=f'{model_name}: ROC Curve',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate')
        return fig