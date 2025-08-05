import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
    def handle_missing_values(self, strategy='mean', columns=None):
        """Handle missing values in specified columns"""
        if columns is None:
            columns = self.numeric_cols
            
        if strategy == 'drop':
            self.df = self.df.dropna(subset=columns)
        else:
            imp = SimpleImputer(strategy=strategy)
            self.df[columns] = imp.fit_transform(self.df[columns])
        return self.df
    
    def encode_categorical(self, columns=None):
        """Encode categorical variables"""
        if columns is None:
            columns = self.categorical_cols
            
        for col in columns:
            if self.df[col].nunique() <= 10:  # One-hot encode if few categories
                self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
            else:  # Label encode if many categories
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
        return self.df
    
    def normalize_data(self, method='standard', columns=None):
        """Normalize/standardize data"""
        if columns is None:
            columns = self.numeric_cols
            
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self.df
    
    def drop_columns(self, columns):
        """Drop specified columns"""
        self.df = self.df.drop(columns=columns)
        return self.df
    
    def get_summary(self):
        """Get data summary including null values"""
        summary = pd.DataFrame({
            'Column': self.df.columns,
            'Type': self.df.dtypes,
            'Missing Values': self.df.isnull().sum(),
            'Unique Values': self.df.nunique()
        })
        return summary