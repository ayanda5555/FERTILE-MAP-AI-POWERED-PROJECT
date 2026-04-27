import os

class Config:
    # Security key (replace in production!)
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')

    # Uploads folder
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')

    # Max upload size (16 MB)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  

    # Database file path (points to Backend/database/soil_app.db)
    DATABASE = os.path.join(os.path.dirname(__file__), 'database', 'soil_app.db')

    # Allowed file extensions for uploads
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}