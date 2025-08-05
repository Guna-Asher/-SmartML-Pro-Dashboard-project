import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from ml_utils.data_processing import DataProcessor
from ml_utils.eda import EDAAnalyzer
from ml_utils.modeling import ModelTrainer
from ml_utils.visualization import ModelVisualizer
import joblib
import base64

# Page configuration
st.set_page_config(
    page_title="SmartML Pro Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

# Helper functions
def get_table_download_link(df, filename="predictions.csv"):
    """Generates a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Predictions</a>'

# Main app
def main():
    st.title("🧠 SmartML Pro Dashboard")
    st.markdown("""
    A comprehensive machine learning dashboard for data analysis, model training, and evaluation.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", [
        "Data Upload", 
        "Data Cleaning", 
        "Exploratory Analysis", 
        "Model Training", 
        "Model Evaluation",
        "Make Predictions"
    ])
    
    # Page 1: Data Upload
    if app_mode == "Data Upload":
        st.header("📁 Dataset Upload")
        
        uploaded_file = st.file_uploader(
            "Drag and drop your dataset (CSV or Excel)", 
            type=['csv', 'xlsx']
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                
                st.success("Dataset loaded successfully!")
                
                # Initialize data processor
                st.session_state.processor = DataProcessor(st.session_state.df)
                
                # Show data preview
                st.subheader("Data Preview")
                st.write(f"Shape: {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(st.session_state.df.head())
                with col2:
                    st.dataframe(st.session_state.processor.get_summary())
                
                # Target selection
                st.session_state.target = st.selectbox(
                    "Select target column", 
                    st.session_state.df.columns
                )
                
                # Problem type detection
                unique_target_values = st.session_state.df[st.session_state.target].nunique()
                if unique_target_values < 10 or st.session_state.df[st.session_state.target].dtype == 'object':
                    st.session_state.problem_type = 'classification'
                    st.info(f"Problem type detected: Classification ({unique_target_values} classes)")
                else:
                    st.session_state.problem_type = 'regression'
                    st.info("Problem type detected: Regression")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Page 2: Data Cleaning
    elif app_mode == "Data Cleaning" and st.session_state.df is not None:
        st.header("🧹 Data Cleaning Tools")
        
        # Missing values handling
        st.subheader("Missing Values Handling")
        missing_summary = st.session_state.df.isnull().sum()
        if missing_summary.sum() > 0:
            st.warning(f"Found {missing_summary.sum()} missing values across {len(missing_summary[missing_summary > 0])} columns")
            st.write(missing_summary[missing_summary > 0])
            
            col1, col2 = st.columns(2)
            with col1:
                strategy = st.selectbox(
                    "Select strategy for handling missing values",
                    ['mean', 'median', 'mode', 'drop']
                )
            with col2:
                columns_to_clean = st.multiselect(
                    "Select columns to clean (leave empty for all numeric columns)",
                    st.session_state.df.columns,
                    default=list(missing_summary[missing_summary > 0].index)
                )
            
            if st.button("Apply Missing Values Handling"):
                st.session_state.df = st.session_state.processor.handle_missing_values(
                    strategy=strategy,
                    columns=columns_to_clean if columns_to_clean else None
                )
                st.success("Missing values handled!")
                st.dataframe(st.session_state.processor.get_summary())
        else:
            st.success("No missing values found in the dataset!")
        
        # Column operations
        st.subheader("Column Operations")
        cols_to_drop = st.multiselect(
            "Select columns to drop",
            [col for col in st.session_state.df.columns if col != st.session_state.target]
        )
        
        if st.button("Drop Selected Columns"):
            st.session_state.df = st.session_state.processor.drop_columns(cols_to_drop)
            st.success(f"Dropped {len(cols_to_drop)} columns")
            st.dataframe(st.session_state.processor.get_summary())
        
        # Feature engineering
        st.subheader("Feature Engineering")
        
        col1, col2 = st.columns(2)
        with col1:
            normalize = st.checkbox("Normalize/Standardize numeric features")
            if normalize:
                norm_method = st.radio(
                    "Normalization method",
                    ['standard', 'minmax']
                )
        with col2:
            encode_cats = st.checkbox("Encode categorical features")
        
        if st.button("Apply Feature Engineering"):
            if normalize:
                st.session_state.df = st.session_state.processor.normalize_data(
                    method=norm_method
                )
            if encode_cats:
                st.session_state.df = st.session_state.processor.encode_categorical()
            
            st.success("Feature engineering applied!")
            st.dataframe(st.session_state.processor.get_summary())
    
    # Page 3: Exploratory Analysis
    elif app_mode == "Exploratory Analysis" and st.session_state.df is not None:
        st.header("📊 Exploratory Data Analysis")
        eda = EDAAnalyzer(st.session_state.df)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Distributions", 
            "Correlations", 
            "Target Analysis", 
            "Feature Relationships"
        ])
        
        with tab1:
            st.subheader("Feature Distributions")
            for fig in eda.plot_distributions(st.session_state.target):
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Feature Correlations")
            for fig in eda.plot_correlation():
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Pairplot (Sample)")
            for fig in eda.plot_pairplot(st.session_state.target):
                st.pyplot(fig)
        
        with tab3:
            st.subheader("Target Variable Analysis")
            for fig in eda.plot_target_distribution(st.session_state.target):
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Feature Relationships")
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox(
                    "X-axis feature",
                    [col for col in st.session_state.df.columns if col != st.session_state.target]
                )
            with col2:
                y_col = st.selectbox(
                    "Y-axis feature",
                    [col for col in st.session_state.df.columns if col != x_col and col != st.session_state.target]
                )
            
            color_col = st.selectbox(
                "Color by",
                [None, st.session_state.target] + 
                [col for col in st.session_state.df.columns if col != st.session_state.target and col != x_col and col != y_col]
            )
            
            for fig in eda.plot_scatter(x_col, y_col, color_col):
                st.plotly_chart(fig, use_container_width=True)
    
    # Page 4: Model Training
    elif app_mode == "Model Training" and st.session_state.df is not None and st.session_state.target is not None:
        st.header("🤖 Model Training")
        
        # Prepare data
        X = st.session_state.df.drop(columns=[st.session_state.target])
        y = st.session_state.df[st.session_state.target]
        
        # Model selection
        st.subheader("Model Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider(
                "Test set size (%)",
                10, 40, 20
            )
            
            # Allow problem type override
            problem_type = st.radio(
                "Problem type",
                ['regression', 'classification'],
                index=0 if st.session_state.problem_type == 'regression' else 1
            )
            
            st.session_state.problem_type = problem_type
        
        with col2:
            tune_models = st.checkbox(
                "Enable basic hyperparameter tuning (slower)",
                value=False
            )
        
        # Get available models based on problem type
        trainer = ModelTrainer(X, y, st.session_state.problem_type)
        available_models = list(trainer.models.keys())
        
        selected_models = st.multiselect(
            "Select models to train",
            available_models,
            default=available_models[:3]
        )
        
        if st.button("Train Selected Models"):
            if not selected_models:
                st.warning("Please select at least one model")
            else:
                with st.spinner("Training models..."):
                    st.session_state.results = trainer.train_models(
                        selected_models,
                        test_size=test_size/100,
                        tune=tune_models
                    )
                
                st.success("Model training completed!")
                st.experimental_rerun()
        
        # Show training results if available
        if st.session_state.results:
            st.subheader("Training Results")
            
            # Metrics comparison
            fig = ModelVisualizer.plot_metrics_comparison(st.session_state.results)
            st.plotly_chart(fig, use_container_width=True)
            
            # Individual model details
            for model_name, result in st.session_state.results.items():
                with st.expander(f"{model_name} Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Performance Metrics")
                        metrics_df = pd.DataFrame(result['metrics'], index=[model_name])
                        st.dataframe(metrics_df.style.highlight_max(axis=0))
                        
                        st.write(f"Training time: {result['train_time']:.2f} seconds")
                    
                    with col2:
                        st.subheader("Visualizations")
                        
                        # Predicted vs Actual
                        fig = ModelVisualizer.plot_predictions_vs_actual(
                            y[result['predictions'].index],  # Get test set y values
                            result['predictions'],
                            model_name
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Residuals (for regression)
                        if st.session_state.problem_type == 'regression':
                            fig = ModelVisualizer.plot_residuals(
                                y[result['predictions'].index],
                                result['predictions'],
                                model_name
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature importance
                        if result['feature_importance']:
                            fig = ModelVisualizer.plot_feature_importance(
                                result['feature_importance'],
                                model_name
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
    
    # Page 5: Model Evaluation
    elif app_mode == "Model Evaluation" and st.session_state.results:
        st.header("📈 Model Evaluation")
        
        # Model comparison
        st.subheader("Model Comparison")
        
        # Metrics comparison
        fig = ModelVisualizer.plot_metrics_comparison(st.session_state.results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrices for classification
        if st.session_state.problem_type == 'classification':
            st.subheader("Confusion Matrices")
            cols = st.columns(2)
            col_idx = 0
            
            for model_name, result in st.session_state.results.items():
                y_test = st.session_state.df[st.session_state.target][result['predictions'].index]
                fig = ModelVisualizer.plot_confusion_matrix(
                    y_test,
                    result['predictions'],
                    model_name
                )
                cols[col_idx].plotly_chart(fig, use_container_width=True)
                col_idx = (col_idx + 1) % 2
        
        # Feature importance comparison
        st.subheader("Feature Importance Comparison")
        importance_data = []
        
        for model_name, result in st.session_state.results.items():
            if result['feature_importance']:
                if isinstance(result['feature_importance'], dict):
                    # Single importance set
                    for feature, importance in result['feature_importance'].items():
                        importance_data.append({
                            'Model': model_name,
                            'Feature': feature,
                            'Importance': importance
                        })
                else:
                    # Multiple importance sets (multi-class)
                    for cls, imp_dict in result['feature_importance'].items():
                        for feature, importance in imp_dict.items():
                            importance_data.append({
                                'Model': f"{model_name} (Class {cls})",
                                'Feature': feature,
                                'Importance': importance
                            })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            
            # Get top features across all models
            top_features = importance_df.groupby('Feature')['Importance'].mean().nlargest(10).index
            
            fig = px.bar(
                importance_df[importance_df['Feature'].isin(top_features)],
                x='Importance', y='Feature', color='Model',
                barmode='group', title='Top Feature Importance Across Models'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for selected models")
    
    # Page 6: Make Predictions
    elif app_mode == "Make Predictions" and st.session_state.results:
        st.header("🔮 Make Predictions")
        
        # Option 1: Use test set predictions
        st.subheader("Existing Predictions")
        if st.checkbox("Show test set predictions"):
            for model_name, result in st.session_state.results.items():
                y_test = st.session_state.df[st.session_state.target][result['predictions'].index]
                pred_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': result['predictions']
                }, index=y_test.index)
                
                st.write(f"{model_name} Predictions:")
                st.dataframe(pred_df)
                
                # Download link
                st.markdown(get_table_download_link(pred_df, f"{model_name}_predictions.csv"), unsafe_allow_html=True)
        
        # Option 2: Upload new data
        st.subheader("New Data Predictions")
        new_data_file = st.file_uploader(
            "Upload new data for predictions (same features as training data)",
            type=['csv', 'xlsx']
        )
        
        if new_data_file:
            try:
                if new_data_file.name.endswith('.csv'):
                    new_df = pd.read_csv(new_data_file)
                else:
                    new_df = pd.read_excel(new_data_file)
                
                st.success("New data loaded successfully!")
                st.dataframe(new_df.head())
                
                # Select model
                selected_model = st.selectbox(
                    "Select model for predictions",
                    list(st.session_state.results.keys())
                )
                
                if st.button("Generate Predictions"):
                    model = st.session_state.results[selected_model]['model']
                    predictions = model.predict(new_df)
                    
                    result_df = new_df.copy()
                    result_df['Prediction'] = predictions
                    
                    st.subheader("Prediction Results")
                    st.dataframe(result_df)
                    
                    # Download link
                    st.markdown(
                        get_table_download_link(result_df, f"{selected_model}_new_predictions.csv"), 
                        unsafe_allow_html=True
                    )
            
            except Exception as e:
                st.error(f"Error processing new data: {str(e)}")
    
    # Handle cases where data isn't loaded
    else:
        if st.session_state.df is None and app_mode != "Data Upload":
            st.warning("Please upload a dataset first on the Data Upload page")
        elif st.session_state.target is None and app_mode not in ["Data Upload", "Data Cleaning"]:
            st.warning("Please select a target column")
        elif app_mode == "Model Training" and not st.session_state.results:
            st.info("Select models and click 'Train Selected Models' to see results")
        elif app_mode in ["Model Evaluation", "Make Predictions"] and not st.session_state.results:
            st.warning("Please train models first on the Model Training page")

if __name__ == "__main__":
    main()