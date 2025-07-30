import os

# Base directory for the project (where main.py is)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the directory where models are stored
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Define paths to your models
CNN_CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, 'glaucoma_cnn_balanced.h5')
UNET_SEGMENTATION_MODEL_PATH = os.path.join(MODELS_DIR, 'unet_glaucoma_multiclass_retrained.h5')
SEVERITY_MODEL_PATH = os.path.join(MODELS_DIR, 'severity_rule_model.pkl')

# Image dimensions required by the models
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Glaucoma classification threshold (for sigmoid output)
GLAUCOMA_THRESHOLD = 0.5

# Segmentation class labels (useful for clarity)
SEG_BACKGROUND = 0
SEG_OPTIC_DISC = 1
SEG_OPTIC_CUP = 2

# Logging configuration (optional, but good practice)
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "glaucoma_pipeline": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}