import streamlit as st
import requests
from PIL import Image
import io
import base64 # Required to decode the base64 image from the API response

# --- Configuration ---
# Make sure this matches where your FastAPI app is running!
FASTAPI_BACKEND_URL = "http://127.0.0.1:8000/predict" # Corrected URL and endpoint

# --- Page setup ---
st.set_page_config(page_title="Glaucoma Disease Prediction", layout="centered")
st.title("👁️ Glaucoma Disease Prediction System")
st.markdown("Upload a retinal image to get predictions for glaucoma, optic disc/cup segmentation, and severity.")

# --- Upload input ---
uploaded_file = st.file_uploader("Choose an Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded original image
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Uploaded Image", use_column_width=True)

    # Note: We do NOT resize the image here.
    # The FastAPI backend handles all preprocessing (resizing, CLAHE, etc.).
    # We send the raw image bytes to the backend.

    # Predict button
    if st.button("Analyze Image"):
        st.info("Processing image... Please wait.")
        try:
            # Convert uploaded file to bytes for API request
            img_bytes = uploaded_file.getvalue()

            # Prepare files for multipart/form-data request
            # The key 'file' must match the parameter name in your FastAPI endpoint:
            # async def predict_image(file: UploadFile = File(...))
            files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}

            # Make the POST request to your FastAPI backend
            response = requests.post(FASTAPI_BACKEND_URL, files=files)

            # --- Handle response ---
            if response.status_code == 200:
                data = response.json()
                st.subheader("📊 Prediction Results")

                # Display Glaucoma Prediction
                glaucoma_status = data.get("glaucoma_prediction")
                glaucoma_score = data.get("glaucoma_confidence_score")

                if glaucoma_status == "Healthy":
                    st.success(f"**Glaucoma Status:** {glaucoma_status} (Confidence: {glaucoma_score:.2f})")
                    st.write("No signs of glaucoma detected in this image.")
                else:
                    st.error(f"**Glaucoma Status:** {glaucoma_status} (Confidence: {glaucoma_score:.2f})")
                    st.warning("Potential signs of glaucoma detected. Further evaluation recommended.")

                st.markdown("---")

                # Display Segmentation Mask
                segmented_mask_b64 = data.get("segmented_mask_png_base64")
                
                if segmented_mask_b64 and isinstance(segmented_mask_b64, str) and not segmented_mask_b64.startswith("Segmentation") and not segmented_mask_b64.startswith("Error"):
                    try:
                        # Decode base64 string back to bytes
                        decoded_img_bytes = base64.b64decode(segmented_mask_b64)
                        # Open as PIL Image
                        segmented_image = Image.open(io.BytesIO(decoded_img_bytes))
                        st.subheader("Segmented Optic Disc & Cup")
                        st.image(segmented_image, caption="Optic Disc (Green), Optic Cup (Red)", use_column_width=True)
                    except Exception as e:
                        st.warning(f"Could not display segmentation mask: {e}")
                        st.text(segmented_mask_b64)
                elif segmented_mask_b64 and isinstance(segmented_mask_b64, str):
                    # It's an error/status message from backend
                    st.subheader("Segmented Optic Disc & Cup")
                    st.info(f"Segmentation Status: {segmented_mask_b64}")
                else:
                    st.subheader("Segmented Optic Disc & Cup")
                    st.info("Segmentation not available (e.g., no glaucoma detected or error).")

                st.markdown("---")

                # Display Severity Prediction
                severity = data.get("severity_prediction")
                st.subheader("Severity Prediction")
                if severity and severity != "N/A (Image classified as Healthy)." and not severity.startswith("Skipped"):
                    st.success(f"**Severity Level:** {severity}")
                else:
                    st.info(f"**Severity Level:** {severity}") # Display the N/A or skipped message

            elif response.status_code == 400:
                st.error(f"Error: Invalid input image. Details: {response.json().get('detail', 'No details provided.')}")
            elif response.status_code == 503:
                st.error(f"Error: Backend service unavailable. Details: {response.json().get('detail', 'Models might not be loaded.')}")
            else:
                st.error(f"An unexpected error occurred with the backend: HTTP {response.status_code}")
                st.json(response.json()) # Display full JSON response for debugging

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Make sure it's running at "
                     f"`{FASTAPI_BACKEND_URL.replace('/predict', '')}`.")
            st.warning("Run the backend with: `uvicorn glaucoma_pipeline.main:app --host 0.0.0.0 --port 8000 --reload`")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Developed with ❤️ by Megha ")