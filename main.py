from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import logging
import os
import io
import requests
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from inference import load_model, inference, cdr_evaluation, get_image_from_url, detection_inference
from cup_utils import create_comprehensive_subplot

logging.basicConfig(level= logging.INFO)
logger = logging.getLogger(__name__)

# Global variable 
class Settings(BaseSettings):
    model_weight_cup: str
    model_weight_disk: str
    inference_image: str
    n_segments: int = 190
    compactness: int = 30
    threshold: float = 90.0
    filter_size: int = 11

    model_config = SettingsConfigDict(
        env_file= ".env",
        env_file_encoding= "utf-8"
    )

# Global variables for models and settings value
setting = Settings()
model_cup = None
model_disk = None
reference_image = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models on startup"""
    global model_cup, model_disk, reference_image

    if setting.model_weight_cup and os.path.exists(setting.model_weight_cup):
        model_cup = load_model(model_weight_path= setting.model_weight_cup)
        logger.info("Model load complete")
    else:
        logger.error(f"Model load failed as there is no path: {setting.model_weight_cup}")
        raise RuntimeError(f"Cup model weight not found: {setting.model_weight_cup}")

    if setting.model_weight_disk and os.path.exists(setting.model_weight_disk):
        model_disk = YOLO(setting.model_weight_disk)
        logger.info("Model load complete")
    else:
        logger.error(f"Model load failed as there is no path: {setting.model_weight_disk}")
        raise RuntimeError(f"Disk model weight not found: {setting.model_weight_disk}")
    
    if setting.inference_image and os.path.exists(setting.inference_image):
        reference_image = cv2.imread(setting.inference_image)
        logger.info("Reference Image load complete")
    else:
        logger.error(f"Reference Image load failed as there is no path: {setting.inference_image}")
        raise RuntimeError(f"Reference Image not found: {setting.inference_image}")

    yield
    # shutdown cleanup
    model_cup = None
    model_disk = None
    reference_image = None

app = FastAPI(lifespan= lifespan)

def predict_pipeline(image: np.ndarray) -> StreamingResponse:
    # Optical_disk extraction
    optical_disk = None
    detections = detection_inference(detect_model= model_disk,
                                     image= image)
    if not detections:
        raise HTTPException(status_code= 404, detail= "Optical disk not found")
    
    optical_disks = [d for d in detections['detection'] if d['name'] == "optical_disk"]
    # Get detection with highest confidence
    best_detection = max(optical_disks, key=lambda x: x['confidence'])
    xmin, ymin = int(best_detection["box"]['x1']), int(best_detection["box"]['y1'])
    xmax, ymax = int(best_detection["box"]['x2']), int(best_detection["box"]['y2'])

    optical_disk = detections["cropped_image"][ymin:ymax, xmin:xmax]
    
    # Get the value of segment result 
    segment_result, optical_disk_params, _, cdr_report, ist_rules, optical_disk_binary_map = inference(
        ref_image = reference_image,
        image= optical_disk,
        model= model_cup,
        n_segments= setting.n_segments,
        compactness= setting.compactness,
        threshold= setting.threshold,
        filter_size= setting.filter_size
    )
    
    # Generate report 
    report_result = cdr_evaluation(
        cdr_report= cdr_report,
        ist_report= ist_rules
    )
    prediction, red_flags, yellow_flags, green_flags = report_result

    # Generate figure 
    fig = create_comprehensive_subplot(
        original_image= optical_disk,
        smoothed_decision_map= segment_result["smoothed_decision_map"],
        binary_mask= segment_result["final_mask"],
        disk_binary_mask = optical_disk_binary_map,
        ellipse_params= segment_result["ellipse_params"],
        disk_ellipse= optical_disk_params,
        cdr_metrics= cdr_report,
        ist_report = ist_rules,
        save_path= None
    )
    buf = io.BytesIO()
    fig.savefig(buf, format = "png")
    buf.seek(0)
    # Close the fig to save memory
    plt.close(fig)

    metadata = {
        "status": "success",
        "cdr_report": {k: round(float(cdr_report[k]), 2) 
                       for k in ["vertical_cdr", "horizontal_cdr", "area_cdr", "vertical_rdr", "horizontal_rdr", "area_rdr", "cup_disk_center_difference"]},
        "prediction": prediction,
        "Red_area": red_flags,
        "Yellow_area": yellow_flags,
        "Green_area": green_flags
    }
    
    return StreamingResponse(content= buf, media_type= "image/png", headers= json.dumps(metadata))


@app.post("/predict")
async def predict(image_url: str = Query(..., description= "Give an color fundus URL")):
    if model_cup is None or model_disk is None:
        raise HTTPException(status_code= 500, detail= "model not loaded")
    
    # Fetch image URL
    response = requests.get(image_url, timeout= 10)
    response.raise_for_status()
    image = get_image_from_url(image_content= response.content)

    return predict_pipeline(image= image)


@app.post("/predict-upload")
async def predict_upload(file: UploadFile = File(..., description="Upload a color fundus image")):
    """Predict from uploaded image file"""
    if model_cup is None or model_disk is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
        )
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        return predict_pipeline(image= image)
    except HTTPException as e:
        raise HTTPException(status_code=400, detail="Image processing error")
    except Exception as e:
        logger.error(f"Error in predict_upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health():
    is_ready = model_cup is not None and model_disk is not None and reference_image is not None
    status_code = 200 if is_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if is_ready else "degraded",
            "model_cup_loaded": "model_cup is healthy",
            "model_disk_loaded": "model_disk is healthy",
            "reference_image_loaded": "reference_image is healthy",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port = 8109, reload= True)