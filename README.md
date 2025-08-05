# 🧠 SmartML Pro Dashboard

A comprehensive, user-friendly machine learning dashboard built with Streamlit that provides end-to-end data analysis, model training, and evaluation capabilities for both classification and regression tasks.

## 🌟 Features

### 📊 **Data Management**
- **Drag & Drop Upload**: Support for CSV and Excel files
- **Data Preview**: Instant dataset overview with shape and summary statistics
- **Data Cleaning**: Automated handling of missing values, outliers, and data types
- **Feature Engineering**: Normalization, encoding, and feature selection tools

### 🔍 **Exploratory Data Analysis (EDA)**
- **Interactive Visualizations**: Distribution plots, correlation matrices, and pairplots
- **Target Analysis**: In-depth analysis of target variable distributions
- **Feature Relationships**: Scatter plots and relationship analysis
- **Statistical Summaries**: Automated statistical insights

### 🤖 **Model Training**
- **Multi-Model Support**: Train multiple algorithms simultaneously
- **AutoML Capabilities**: Automatic problem type detection (classification/regression)
- **Hyperparameter Tuning**: Optional grid search for optimal parameters
- **Cross-Validation**: Built-in k-fold cross-validation
- **Supported Algorithms**:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Support Vector Machines
  - Logistic Regression
  - Linear/Elastic Net Regression
  - And more...

### 📈 **Model Evaluation**
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, RMSE, MAE, R²
- **Visual Comparisons**: Side-by-side model performance charts
- **Confusion Matrices**: For classification tasks
- **Feature Importance**: SHAP values and permutation importance
- **Residual Analysis**: For regression tasks

### 🔮 **Prediction Interface**
- **Batch Predictions**: Upload new data for predictions
- **Model Selection**: Choose from trained models
- **Download Results**: Export predictions as CSV
- **Real-time Updates**: Live prediction updates

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smartml-pro.git
cd smartml-pro
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

### Using the Dashboard

1. **Upload Data**: Go to "Data Upload" and drag your CSV/Excel file
2. **Clean Data**: Use "Data Cleaning" to handle missing values and outliers
3. **Explore**: Check "Exploratory Analysis" for insights
4. **Train Models**: Select models and train on "Model Training" page
5. **Evaluate**: Compare models on "Model Evaluation"
6. **Predict**: Use "Make Predictions" for new data

## 📋 Requirements

- Python 3.7+
- Streamlit 1.22.0+
- Pandas 1.5.0+
- Scikit-learn 1.2.0+
- Plotly 5.11.0+
- XGBoost 1.7.3+
- SHAP 0.41.0+

## 🏗️ Project Structure

```
smartml-pro/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── ml_utils/            # Machine learning utilities
│   ├── data_processing.py    # Data preprocessing
│   ├── eda.py               # Exploratory data analysis
│   ├── modeling.py          # Model training
│   └── visualization.py     # Plotting utilities
├── utils/               # Utility functions
│   └── logger.py        # Logging configuration
├── models/              # Saved models directory
├── data/               # Data storage
└── logs/               # Application logs
```

## ⚙️ Configuration

The application can be configured through environment variables or by modifying `config.py`:

```python
# Example environment variables
export DEBUG=True
export MAX_DATASET_SIZE_MB=200
export TEST_SIZE=0.3
export CV_FOLDS=10
```

## 🎯 Use Cases

- **Business Analytics**: Customer churn prediction, sales forecasting
- **Healthcare**: Disease diagnosis, patient outcome prediction
- **Finance**: Credit scoring, fraud detection
- **Marketing**: Campaign effectiveness, lead scoring
- **Education**: Student performance prediction
- **Manufacturing**: Quality control, predictive maintenance

## 🔧 Advanced Usage

### Custom Model Integration

To add your own models, extend the `ModelTrainer` class in `ml_utils/modeling.py`:

```python
from ml_utils.modeling import ModelTrainer

class CustomTrainer(ModelTrainer):
    def add_custom_model(self, model_name, model_instance):
        self.models[model_name] = model_instance
```

### API Integration

The dashboard can be extended to work with APIs:

```python
# Example API endpoint integration
import requests

def fetch_data_from_api(url):
    response = requests.get(url)
    return pd.DataFrame(response.json())
```

## 📊 Performance Tips

- **Large Datasets**: Use sampling for datasets > 100MB
- **Memory Management**: Enable caching for repeated operations
- **Model Training**: Use smaller test sizes for faster training
- **Visualization**: Limit features for pairplots on large datasets

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Scikit-learn community for machine learning algorithms
- Plotly for interactive visualizations
- XGBoost developers for gradient boosting
- SHAP team for model interpretability

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/smartml-pro/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/smartml-pro/discussions)
- **Email**: support@smartml-pro.com

## 🔄 Changelog

### Version 2.0.0 (Current)
- Complete dashboard redesign
- Added XGBoost support
- Enhanced visualization capabilities
- Improved model evaluation metrics
- Added SHAP feature importance

### Version 1.0.0
- Initial release with basic ML functionality
- Support for common algorithms
- Basic EDA and visualization

---

**Made with ❤️ by the SmartML Pro Team**
