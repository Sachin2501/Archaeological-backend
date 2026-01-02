import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the application"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')
    
    # File upload settings
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', 'processed')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'tif', 'tiff', 'jpg', 'jpeg', 'png', 'geotiff'}
    
    # Model paths
    MODEL_FOLDER = os.getenv('MODEL_FOLDER', 'models')
    SEGMENTATION_MODEL_PATH = os.getenv('SEGMENTATION_MODEL_PATH', 'models/segmentation.h5')
    DETECTION_MODEL_PATH = os.getenv('DETECTION_MODEL_PATH', 'models/detection.pt')
    EROSION_MODEL_PATH = os.getenv('EROSION_MODEL_PATH', 'models/erosion.pkl')
    
    # Database settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///site_mapping.db')
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:8000').split(',')
    
    # Image processing settings
    DEFAULT_IMAGE_SIZE = (512, 512)
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))
    
    # Model training settings
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    EPOCHS = int(os.getenv('EPOCHS', 50))
    
    # Analysis settings
    EROSION_RISK_THRESHOLDS = {
        'low': 0.3,
        'medium': 0.6,
        'high': 0.8
    }
    
    # API settings
    API_PREFIX = '/api'
    API_VERSION = 'v1'
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
        
        # Create necessary directories
        for folder in [Config.UPLOAD_FOLDER, Config.PROCESSED_FOLDER, Config.MODEL_FOLDER]:
            os.makedirs(folder, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_URL = 'sqlite:///:memory:'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration by name"""
    if config_name is None:
        config_name = os.getenv('FLASK_CONFIG', 'default')
    return config.get(config_name, config['default'])