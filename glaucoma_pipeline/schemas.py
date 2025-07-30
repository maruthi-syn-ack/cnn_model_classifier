from pydantic import BaseModel, Field
from typing import Optional, Union

class PredictionResponse(BaseModel):
    """
    Schema for the response returned by the /predict API endpoint.
    """
    filename: str = Field(..., example="retina_image.jpg", description="Name of the uploaded file.")
    glaucoma_prediction: str = Field(..., example="Glaucoma", description="Predicted glaucoma status (Healthy or Glaucoma).")
    glaucoma_confidence_score: Optional[float] = Field(
        None, example=0.8765, description="Confidence score of the glaucoma prediction (0.0 to 1.0)."
    )
    segmented_mask_png_base64: Optional[str] = Field(
        None,
        example="iVBORw0KGgoAAAANSUhEUgAAAPAAAADwCAYAAAA+VemSAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAA",
        description="Base64 encoded PNG image of the segmented optic disc and cup. Null if segmentation skipped."
    )
    severity_prediction: Union[str, float, None] = Field(
        None,
        example="Mild",
        description="Predicted glaucoma severity. 'N/A' or 'Skipped' if not applicable or an error occurred."
    )

    class Config:
        schema_extra = {
            "example": {
                "filename": "patient_001.jpg",
                "glaucoma_prediction": "Glaucoma",
                "glaucoma_confidence_score": 0.92,
                "segmented_mask_png_base64": "iVBORw0KGgoAAAANSUhEUgAAAPAAAADwCAYAAAA+VemSAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAA",
                "severity_prediction": "Moderate"
            }
        }