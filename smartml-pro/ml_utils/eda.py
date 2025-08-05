import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
import numpy as np

_lock = RendererAgg.lock

class EDAAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def plot_distributions(self, target=None):
        """Plot distributions of numeric features"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col == target:
                continue
                
            fig = px.histogram(self.df, x=col, color=target, 
                              marginal='box', title=f'Distribution of {col}')
            fig.update_layout(bargap=0.1)
            yield fig
    
    def plot_correlation(self):
        """Plot correlation heatmap"""
        corr = self.df.select_dtypes(include=['number']).corr()
        fig = px.imshow(corr, text_auto=True, aspect='auto',
                       title='Feature Correlation Heatmap')
        yield fig
    
    def plot_pairplot(self, target=None, sample_size=500):
        """Plot pairplot (sampled for performance)"""
        if len(self.df) > sample_size:
            df_sample = self.df.sample(sample_size)
        else:
            df_sample = self.df
            
        numeric_cols = df_sample.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 8:  # Limit number of columns for performance
            numeric_cols = numeric_cols[:8]
            
        if target and target in numeric_cols:
            color_col = target
        else:
            color_col = None
            
        with _lock:
            fig = sns.pairplot(df_sample[numeric_cols], hue=color_col)
            plt.suptitle('Pairplot of Numeric Features', y=1.02)
            yield fig
    
    def plot_target_distribution(self, target):
        """Plot distribution of target variable"""
        if self.df[target].nunique() > 20:  # Continuous
            fig = px.histogram(self.df, x=target, title=f'Distribution of Target: {target}')
        else:  # Categorical
            fig = px.bar(self.df[target].value_counts(), 
                        title=f'Distribution of Target: {target}')
        yield fig
    
    def plot_scatter(self, x_col, y_col, color_col=None):
        """Interactive scatter plot"""
        fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col,
                        hover_data=self.df.columns, 
                        title=f'{y_col} vs {x_col}')
        yield fig