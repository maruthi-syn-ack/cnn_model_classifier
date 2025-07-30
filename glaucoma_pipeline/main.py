from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import JSONResponse
import logging
import logging.config
from typing import Optional ,Any

# Import components from our structured project
from .config import LOGGING_CONFIG
from .schemas import PredictionResponse
from .ml_core.pipeline import run_full_pipeline
from .dependencies import load_all_ml_models, get_cnn_classifier_model, get_unet_segmentation_model, get_severity_rule_model

# Configure logging using the dictionary from config.py
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("glaucoma_pipeline") # Get our specific logger

# Initialize FastAPI app
app = FastAPI(
    title="Glaucoma Diagnosis ML Pipeline API",
    description="An API for classifying glaucoma, segmenting optic disc/cup, and predicting severity.",
    version="1.0.0",
    docs_url="/docs",       # Enable Swagger UI at /docs
    redoc_url="/redoc"      # Enable ReDoc at /redoc
)

# --- FastAPI Lifespan Events ---
# This decorator ensures `load_all_ml_models` is called once at app startup
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup event: Loading models...")
    await load_all_ml_models() # Call the async function from dependencies

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown event.")
    # You could add cleanup logic here if needed (e.g., closing connections)

# --- API Endpoints ---

@app.get("/", summary="Root endpoint for API status")
async def read_root():
    """
    Returns a welcome message and basic API status.
    """
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Glaucoma Diagnosis ML Pipeline API! Use /predict to send images."}

@app.post(
    "/predict",
    response_model=PredictionResponse, # Ensure response adheres to our schema
    summary="Process a retinal image through the glaucoma diagnosis pipeline",
    status_code=status.HTTP_200_OK
)
async def predict_image(
    file: UploadFile = File(..., description="Upload a retinal image (PNG, JPG, JPEG)."),
    cnn_model: Any = Depends(get_cnn_classifier_model),
    unet_model: Any = Depends(get_unet_segmentation_model),
    severity_model: Optional[Any] = Depends(get_severity_rule_model)
):
    """
    Accepts an image file and processes it through the ML pipeline to predict
    glaucoma status, segment the optic disc and cup, and predict severity (conditionally).

    - **file**: The retinal image file to be processed.
    - **Returns**: A JSON object containing the prediction results.
    """
    logger.info(f"Received prediction request for file: {file.filename}")

    # Basic content type validation
    if not file.content_type or not file.content_type.startswith('image/'):
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image type (e.g., image/jpeg, image/png)."
        )

    # Check if essential models are loaded
    if cnn_model is None or unet_model is None:
        logger.error("Required ML models are not loaded. Cannot proceed with prediction.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning models are not loaded. Please check server logs."
        )

    try:
        image_bytes = await file.read()

        # Run the full pipeline using the imported function
        # Pass the actual filename for the response
        results = await run_full_pipeline(image_bytes, cnn_model, unet_model, severity_model)
        results.filename = file.filename # Set the filename from the uploaded file

        logger.info(f"Prediction successful for {file.filename}. Glaucoma: {results.glaucoma_prediction}")
        return results

    except ValueError as ve:
        logger.error(f"Image processing error for {file.filename}: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except HTTPException: # Re-raise FastAPI's HTTPExceptions
        raise
    except Exception as e:
        logger.critical(f"An unexpected error occurred during prediction for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction. Please try again or contact support. Error: {e}"
        )