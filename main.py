from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import logging
import os
import io
import requests
import torch
import json
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from dotenv import load_dotenv

from inference import load_model, inference, cdr_evaluation, get_image_from_url, detection_inference
from cup_utils import create_comprehensive_subplot

load_dotenv(".env")
app =FastAPI()

logging.basicConfig(level= logging.INFO)
logger = logging.getLogger(__name__)

# Global variable 
model_weight_cup = os.getenv("MODEL_WEIGHT_CUP")
model_weight_disk = os.getenv("MODEL_WEIGHT_DISK")
ref_image_path = os.getenv("INFERENCE_IMAGE")
n_segments_value = int(os.getenv("N_SEGMENTS"))
compactness_value = int(os.getenv("COMPACTNESS"))
threshold_value = float(os.getenv("THRESHOLD"))
filter_size_value = int(os.getenv("FILTER_SIZE"))

# Global variables for models
model_cup = None
model_disk = None
reference_image = None

@app.on_event("startup")
async def startup_event():
    """Load the models on startup"""
    global model_cup, model_disk, reference_image

    try:
        if model_weight_cup and os.path.exists(model_weight_cup):
            model_cup = load_model(model_weight_path= model_weight_cup)
            logger.info("Model load complete")
        else:
            logger.error(f"Model load failed as there is no path: {model_weight_cup}")

        if model_weight_disk and os.path.exists(model_weight_disk):
            model_disk = YOLO(model_weight_disk)
            logger.info("Model load complete")
        else:
            logger.error(f"Model load failed as there is no path: {model_weight_disk}")
        
        if ref_image_path and os.path.exists(ref_image_path):
            reference_image = cv2.imread(ref_image_path)
            logger.info("Reference Image load complete")
        else:
            logger.error(f"Reference Image load failed as there is no path: {ref_image_path}")
    except Exception as e:
        logger.error(f"Error loading the models: {e}")

@app.post("/predict")
async def predict(image_url: str = Query(..., description= "Give an color fundus URL")):
    if model_cup is None or model_disk is None:
        raise HTTPException(status_code= 500, detail= "model not loaded")
    
    # Fetch image URL
    response = requests.get(image_url, timeout= 10)
    response.raise_for_status()
    image = get_image_from_url(image_content= response.content)

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
    segment_result, optical_disk_params, optical_cup_params, cdr_report, ist_rules, optical_disk_binary_map = inference(
        ref_image = reference_image,
        image= optical_disk,
        model= model_cup,
        n_segments= n_segments_value,
        compactness= compactness_value,
        threshold= threshold_value,
        filter_size= filter_size_value
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
    plt.savefig(buf, format = "png")
    buf.seek(0)
    image_bytes = buf.read()
    buf = io.BytesIO(image_bytes)
    # Close the fig to save memory
    plt.close(fig)

    # Transfer to metadata
    vertical_cdr = round(float(cdr_report["vertical_cdr"]), 2)
    horizontal_cdr = round(float(cdr_report["horizontal_cdr"]), 2)
    area_cdr = round(float(cdr_report["area_cdr"]), 2)
    vertical_rdr = round(float(cdr_report["vertical_rdr"]), 2)
    horizontal_rdr = round(float(cdr_report["horizontal_rdr"]), 2)
    area_rdr = round(float(cdr_report["area_rdr"]), 2)
    cup_disk_center_difference = round(float(cdr_report["cup_disk_center_difference"]), 2)

    metadata = {
        "status": "success",
        "cdr_report": {
            "vertical_cdr": vertical_cdr,
            "horizontal_cdr": horizontal_cdr,
            "area_cdr": area_cdr,
            "vertical_rdr": vertical_rdr,
            "horizontal_rdr": horizontal_rdr,
            "area_rdr": area_rdr,
            "Cup_Disk_center_diff": cup_disk_center_difference
        },
        "prediction": prediction,
        "Red_area": red_flags,
        "Yellow_area": yellow_flags,
        "Green_area": green_flags
    }
    metadata_json = json.dumps(metadata) 
    print(type(metadata_json))
    headers= {"Result": metadata_json}
    
    # Delete the result
    del segment_result, optical_disk_params, optical_cup_params, cdr_report, ist_rules, optical_disk_binary_map
    return StreamingResponse(content= buf, media_type= "image/png", headers= headers) 


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
        segment_result, optical_disk_params, optical_cup_params, cdr_report, ist_rules, optical_disk_binary_map = inference(
            ref_image = reference_image,
            image= optical_disk,
            model= model_cup,
            n_segments= n_segments_value,
            compactness= compactness_value,
            threshold= threshold_value,
            filter_size= filter_size_value
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
        plt.savefig(buf, format = "png")
        buf.seek(0)
        image_bytes = buf.read()
        buf = io.BytesIO(image_bytes)
        # Close the fig to save memory
        plt.close(fig)

        # Transfer to metadata
        vertical_cdr = round(float(cdr_report["vertical_cdr"]), 2)
        horizontal_cdr = round(float(cdr_report["horizontal_cdr"]), 2)
        area_cdr = round(float(cdr_report["area_cdr"]), 2)
        vertical_rdr = round(float(cdr_report["vertical_rdr"]), 2)
        horizontal_rdr = round(float(cdr_report["horizontal_rdr"]), 2)
        area_rdr = round(float(cdr_report["area_rdr"]), 2)
        cup_disk_center_difference = round(float(cdr_report["cup_disk_center_difference"]), 2)

        metadata = {
            "status": "success",
            "cdr_report": {
                "vertical_cdr": vertical_cdr,
                "horizontal_cdr": horizontal_cdr,
                "area_cdr": area_cdr,
                "vertical_rdr": vertical_rdr,
                "horizontal_rdr": horizontal_rdr,
                "area_rdr": area_rdr,
                "Cup_Disk_center_diff": cup_disk_center_difference
            },
            "prediction": prediction,
            "Red_area": red_flags,
            "Yellow_area": yellow_flags,
            "Green_area": green_flags
        }
        metadata_json = json.dumps(metadata) 
        print(type(metadata_json))
        headers= {"Result": metadata_json}
        
        # Delete the result
        del segment_result, optical_disk_params, optical_cup_params, cdr_report, ist_rules, optical_disk_binary_map
        return StreamingResponse(content= buf, media_type= "image/png", headers= headers) 

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port = 8109, reload= True)