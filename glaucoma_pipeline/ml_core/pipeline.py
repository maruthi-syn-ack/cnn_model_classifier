import numpy as np
import cv2
import base64
from io import BytesIO
from typing import Tuple, Dict, Any, Optional, Union
import joblib

# Import configurations
from ..config import IMG_WIDTH, IMG_HEIGHT, GLAUCOMA_THRESHOLD, SEG_OPTIC_DISC, SEG_OPTIC_CUP
# Import schemas (though not strictly necessary here, good for type hinting)
from ..schemas import PredictionResponse

# These will be injected via dependencies in FastAPI, so we define their types here
KerasModel = Any # Type hint for a loaded Keras model (tf.keras.Model)
PickleModel = Any # Type hint for a loaded joblib/pickle model (any Python object with a .predict method)

# ==============================================================================
# SeverityClassifier class moved to ml_core.severity_classifier module
# Import it from there for joblib model loading
# ==============================================================================


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Loads and preprocesses a single image from bytes.
    Steps: Decode, Resize, Apply CLAHE, Convert to RGB, Normalize, Add batch dimension.

    Args:
        image_bytes (bytes): The raw bytes of the image file.

    Returns:
        np.ndarray: The preprocessed image ready for model inference, with batch dimension (1, H, W, C).

    Raises:
        ValueError: If the image bytes cannot be decoded.
    """
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image bytes. Ensure it's a valid image format.")

    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Apply CLAHE
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    clahe_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR) # Image is currently BGR

    # =========================================================================
    # CRITICAL FIX: Convert from BGR (OpenCV default) to RGB (TensorFlow/Keras common expectation)
    # This aligns the inference preprocessing with the training preprocessing (as implied by imshow).
    # =========================================================================
    clahe_img_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)

    # Normalize to 0-1 range
    normalized_img = clahe_img_rgb / 255.0 # Ensure normalization is applied to the RGB version

    return np.expand_dims(normalized_img, axis=0) # Add batch dimension

def predict_glaucoma_classification(preprocessed_image: np.ndarray, cnn_classifier_model: KerasModel) -> Tuple[int, float]:
    """
    Predicts whether an image shows signs of glaucoma.
    """
    prediction = cnn_classifier_model.predict(preprocessed_image)
    confidence_score = float(prediction[0][0])
    # is_glaucoma = int(confidence_score <= GLAUCOMA_THRESHOLD)
    is_glaucoma = int(confidence_score > GLAUCOMA_THRESHOLD) # This will be 1 if True, 0 if False

    return is_glaucoma, confidence_score

def predict_segmentation(preprocessed_image: np.ndarray, unet_segmentation_model: KerasModel) -> np.ndarray:
    """
    Segments the optic disc and optic cup from the image.
    Applies post-processing steps (morphological closing).

    Args:
        preprocessed_image (np.ndarray): The preprocessed image (1, H, W, C).
        unet_segmentation_model (KerasModel): The loaded U-Net segmentation model.

    Returns:
        np.ndarray: The post-processed segmented mask (H, W), with pixel values
                    corresponding to SEG_BACKGROUND, SEG_OPTIC_DISC, SEG_OPTIC_CUP.
    """
    raw_mask_prediction = unet_segmentation_model.predict(preprocessed_image)
    predicted_mask = np.argmax(raw_mask_prediction, axis=-1)[0] # Remove batch dimension

    # Morphological closing
    kernel = np.ones((5,5), np.uint8) # Kernel size can be tuned
    closed_mask = cv2.morphologyEx(predicted_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # TODO: Integrate your specific 'remove_small_objects' logic here if available from notebook
    final_segmented_mask = closed_mask

    return final_segmented_mask

def calculate_cdr_percent(segmented_mask: np.ndarray) -> float:
    """
    Calculates the Cup-to-Disc Ratio (CDR) percentage from the segmented mask.
    This is a simplified calculation based on pixel count, assuming circular areas.
    Your training code uses `cdr_percent` directly, implying it's already a percentage.
    We'll assume 'cdr_percent' in your model means (cup_area / disc_area) * 100.
    """
    optic_disc_pixels = np.sum(segmented_mask == SEG_OPTIC_DISC)
    optic_cup_pixels = np.sum(segmented_mask == SEG_OPTIC_CUP)

    if optic_disc_pixels == 0:
        # If no optic disc is found, CDR calculation is problematic.
        # Returning 100.0 will likely push it to 'Severe' in your rule.
        # Consider if a different error handling or 'N/A' is more appropriate for your use case.
        return 100.0
    if optic_cup_pixels == 0:
        # If no optic cup is found, CDR is 0, implying healthy/mild
        return 0.0

    cdr_percentage = (optic_cup_pixels / optic_disc_pixels) * 100.0

    return cdr_percentage

def predict_severity(glaucoma_status: int, segmented_mask: np.ndarray, severity_rule_model: Optional[PickleModel]) -> Union[str, float]:
    """
    Predicts glaucoma severity based on the segmented mask and the rule-based model.

    Args:
        glaucoma_status (int): 1 if glaucoma, 0 if healthy.
        segmented_mask (np.ndarray): The segmented mask (H, W).
        severity_rule_model (Optional[PickleModel]): The loaded severity rule model or None.

    Returns:
        Union[str, float]: The predicted severity (string), or a placeholder message.
    """
    if severity_rule_model is None:
        return "Severity model not loaded or defined."

    if glaucoma_status == 0:
        return "N/A (Image classified as Healthy)."

    # Calculate cdr_percent from the segmented mask
    cdr_percent = calculate_cdr_percent(segmented_mask)

    # Use the loaded rule-based model to predict severity
    # Reshape scalar to 2D array as expected by scikit-learn models
    cdr_input = np.array([[cdr_percent]])  # Shape: (1, 1)
    severity_label = severity_rule_model.predict(cdr_input)[0]
    
    # Ensure the result is always a string
    if isinstance(severity_label, (np.number, float, int)):
        # Convert numeric predictions to string labels
        cdr_value = float(severity_label)
        if cdr_value < 50:
            return "Mild"
        elif 50 <= cdr_value < 80:
            return "Moderate"
        else:
            return "Severe"
    
    return str(severity_label)


def create_mask_visualization(segmented_mask: np.ndarray) -> str:
    """
    Converts a segmented mask (0, 1, 2 values) into a color-coded PNG image,
    base64 encoded for JSON response.

    Args:
        segmented_mask (np.ndarray): The segmented mask (H, W).

    Returns:
        str: Base64 encoded PNG string of the visualized mask.
    """
    mask_display = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    # Green for Optic Disc
    mask_display[segmented_mask == SEG_OPTIC_DISC] = [0, 255, 0]
    # Red for Optic Cup
    mask_display[segmented_mask == SEG_OPTIC_CUP] = [255, 0, 0]

    is_success, buffer = cv2.imencode(".png", mask_display)
    if not is_success:
        raise ValueError("Could not encode segmented mask to PNG.")
    return base64.b64encode(buffer).decode("utf-8")

def get_segmentation_image_array(segmented_mask: np.ndarray) -> np.ndarray:
    """
    Converts a segmented mask (0, 1, 2 values) into a color-coded image array
    for direct visualization, similar to the training notebook.

    Args:
        segmented_mask (np.ndarray): The segmented mask (H, W).

    Returns:
        np.ndarray: Color-coded image array (H, W, 3) with RGB colors.
    """
    mask_display = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    # Yellow for Optic Disc (matching training notebook)
    mask_display[segmented_mask == SEG_OPTIC_DISC] = [255, 255, 0]
    # Red for Optic Cup (matching training notebook)
    mask_display[segmented_mask == SEG_OPTIC_CUP] = [255, 0, 0]
    
    return mask_display

# --- Full Pipeline Orchestrator ---
async def run_full_pipeline(
    image_bytes: bytes,
    cnn_classifier_model: KerasModel,
    unet_segmentation_model: KerasModel,
    severity_rule_model: Optional[PickleModel]
) -> PredictionResponse:
    """
    Executes the full machine learning pipeline for a given image,
    including conditional steps for segmentation and severity prediction.

    Args:
        image_bytes (bytes): The raw bytes of the input image.
        cnn_classifier_model (KerasModel): The loaded CNN classifier model.
        unet_segmentation_model (KerasModel): The loaded U-Net segmentation model.
        severity_rule_model (Optional[PickleModel]): The loaded severity rule model or None.

    Returns:
        PredictionResponse: A Pydantic model containing all prediction results.

    Raises:
        ValueError: If image preprocessing fails.
    """
    filename = "uploaded_image.jpg" # This will be overwritten by the actual filename in the FastAPI endpoint
    glaucoma_status = "Unknown"
    glaucoma_confidence_score = None
    segmented_mask_b64 = None
    severity_prediction_result: Union[str, float, None] = "N/A"

    preprocessed_img = preprocess_image(image_bytes)

    # --- Step 1: Glaucoma Classification ---
    is_glaucoma, glaucoma_score = predict_glaucoma_classification(preprocessed_img, cnn_classifier_model)
    glaucoma_status = "Glaucoma" if is_glaucoma == 1 else "Healthy"
    glaucoma_confidence_score = glaucoma_score

    # --- Step 2: Conditional Optic Disc and Cup Segmentation ---
    if is_glaucoma == 1: # Only segment if classified as Glaucoma
        print("Image classified as Glaucoma. Proceeding with segmentation.")
        try:
            segmented_mask = predict_segmentation(preprocessed_img, unet_segmentation_model)
            segmented_mask_b64 = create_mask_visualization(segmented_mask)

            # --- Step 3: Conditional Severity Prediction ---
            # Only if glaucoma is detected AND segmentation was successful
            severity_prediction_result = predict_severity(is_glaucoma, segmented_mask, severity_rule_model)

        except Exception as e:
            segmented_mask_b64 = f"Error during segmentation or severity calculation: {str(e)}"
            severity_prediction_result = "Skipped due to upstream error."
            print(f"Skipping segmentation/severity due to error: {e}")

    else: # Image classified as Healthy
        segmented_mask_b64 = "Segmentation skipped (Image classified as Healthy)."
        severity_prediction_result = "N/A (Image classified as Healthy)."
        print("Image classified as Healthy. Skipping segmentation and severity prediction.")

    return PredictionResponse(
        filename=filename,
        glaucoma_prediction=glaucoma_status,
        glaucoma_confidence_score=glaucoma_confidence_score,
        segmented_mask_png_base64=segmented_mask_b64,
        severity_prediction=severity_prediction_result
    )