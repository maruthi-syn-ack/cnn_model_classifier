import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib # Changed from pickle
import logging
from typing import Optional, Any

# Import configurations
from .config import CNN_CLASSIFIER_MODEL_PATH, UNET_SEGMENTATION_MODEL_PATH, SEVERITY_MODEL_PATH
# Import SeverityClassifier for joblib model loading
from .ml_core.severity_classifier import SeverityClassifier

logger = logging.getLogger("glaucoma_pipeline")

# Global variables to store loaded models (will be populated on startup)
_cnn_classifier_model: Any = None
_unet_segmentation_model: Any = None
_severity_rule_model: Any = None

def get_cnn_classifier_model():
    """Dependency to provide the CNN classifier model."""
    if _cnn_classifier_model is None:
        logger.error("CNN Classifier model is not loaded!")
    return _cnn_classifier_model

def get_unet_segmentation_model():
    """Dependency to provide the UNet segmentation model."""
    if _unet_segmentation_model is None:
        logger.error("UNet Segmentation model is not loaded!")
    return _unet_segmentation_model

def get_severity_rule_model():
    """Dependency to provide the severity rule model."""
    if _severity_rule_model is None:
        logger.warning("Severity Rule Model is not loaded or not found.")
    return _severity_rule_model

async def load_all_ml_models():
    """
    Loads all ML models into global variables.
    This function is intended to be called once during application startup.
    """
    global _cnn_classifier_model, _unet_segmentation_model, _severity_rule_model
    logger.info("Starting to load ML models...")

    # Load CNN Classifier
    try:
        _cnn_classifier_model = load_model(CNN_CLASSIFIER_MODEL_PATH)
        logger.info(f"CNN Classifier loaded successfully from: {CNN_CLASSIFIER_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading CNN Classifier from {CNN_CLASSIFIER_MODEL_PATH}: {e}")
        _cnn_classifier_model = None

    # Load UNet Segmentation Model
    try:
        _unet_segmentation_model = load_model(UNET_SEGMENTATION_MODEL_PATH)
        logger.info(f"UNet Segmentation model loaded successfully from: {UNET_SEGMENTATION_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading UNet Segmentation model from {UNET_SEGMENTATION_MODEL_PATH}: {e}")
        _unet_segmentation_model = None

    # Load Severity Rule Model
    try:
        # Changed to joblib.load
        _severity_rule_model = joblib.load(SEVERITY_MODEL_PATH)
        logger.info(f"Severity Rule Model loaded successfully from: {SEVERITY_MODEL_PATH}")
    except FileNotFoundError:
        logger.warning(f"Severity Rule Model not found at {SEVERITY_MODEL_PATH}. It will not be available.")
        _severity_rule_model = None
    except Exception as e:
        logger.error(f"Error loading Severity Rule Model from {SEVERITY_MODEL_PATH}: {e}")
        _severity_rule_model = None

    logger.info("All model loading attempts completed.")