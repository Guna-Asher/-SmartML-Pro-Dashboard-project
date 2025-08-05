"""
Configuration management for SmartML Pro
"""
import os
from typing import Dict, Any

class Config:
    """Configuration settings for SmartML Pro"""
    
    # General settings
    APP_NAME = "SmartML Pro"
    VERSION = "2.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Data processing settings
    MAX_DATASET_SIZE_MB = int(os.getenv("MAX_DATASET_SIZE_MB", "100"))
    MAX_FEATURES = int(os.getenv("MAX_FEATURES", "1000"))
    SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "10000"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
    
    # Model settings
    TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
    CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
    N_JOBS = int(os.getenv("N_JOBS", "-1"))
    
    # Performance settings
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "True").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File paths
    MODELS_DIR = os.getenv("MODELS_DIR", "./models")
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    LOGS_DIR = os.getenv("LOGS_DIR", "./logs")
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return {
            "app_name": cls.APP_NAME,
            "version": cls.VERSION,
            "debug": cls.DEBUG,
            "max_dataset_size_mb": cls.MAX_DATASET_SIZE_MB,
            "max_features": cls.MAX_FEATURES,
            "sample_size": cls.SAMPLE_SIZE,
            "test_size": cls.TEST_SIZE,
            "cv_folds": cls.CV_FOLDS,
            "n_jobs": cls.N_JOBS,
            "enable_caching": cls.ENABLE_CACHING,
            "models_dir": cls.MODELS_DIR,
            "data_dir": cls.DATA_DIR,
            "logs_dir": cls.LOGS_DIR
        }
