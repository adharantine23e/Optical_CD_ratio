import os
from typing import Optional, Dict
import cv2
import json
import joblib
from pathlib import Path
from tqdm import tqdm
import numpy as np
import io
import matplotlib.pyplot as plt

from feature_selection import feature_extraction
from cup_utils import extract_optical_cup, segment_optic_disc_cup, caculate_cdr_metrics, IST_rule
from disk_utils import detect_optical_disk_ellipse_v2, extract_optical_disk_v1

def  get_image_from_url(image_content: bytes) -> np.ndarray:
    image_stream = io.BytesIO(image_content)
    image = cv2.imdecode(np.frombuffer(image_stream.getvalue(), np.uint8), 1)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_model(model_weight_path: str):
    svm_model = joblib.load(model_weight_path)
    return svm_model

def detection_inference(detect_model, image: np.ndarray, conf: float = 0.1,
                        iou: float = 0.45,
                        imgsz: int = 1280):
    copied_img = image.copy()
    results = detect_model(image, conf= conf, iou= iou, imgsz = imgsz)
    od_macular = results[0].to_json()
    detections = json.loads(od_macular)
    if detections:
        return {"detection": detections, "cropped_image": copied_img}
    else:
        return {}

def inference(ref_image: np.ndarray,
             image: np.ndarray,
             model,
             n_segments: int = 50,
             compactness: int = 30,
             filter_size: int = 5,
             threshold: float = 80,
             apply_ellipse: bool = True) -> Dict[str, any]:
    """
    Perform inference on a single fundus image with comprehensive visualization.
    
    Args:
        image: np.array from cv2.imread
        model: trained SVM model (.pkl)
        n_segments: Number of SLIC superpixels
        compactness: SLIC compactness parameter
        filter_size: Size of mean filter for smoothing
        threshold: Decision threshold for binary segmentation
        apply_ellipse: Whether to apply ellipse fitting post-processing
        
    Returns:
        Dictionary containing all segmentation results and visualization
    """
    
    # Extract features
    features, slic_segments = feature_extraction(
        image= image,
        n_segments= n_segments,
        compactness= compactness)
    
    # Segmentation
    segment_result = segment_optic_disc_cup(
        features= features,
        slic_segments= slic_segments,
        svm_model= model,
        filter_size= filter_size,
        threshold= threshold,
        apply_ellipse= apply_ellipse)
    
    # Detect optical disk params
    optical_disk_params, optical_disk_binary_map = detect_optical_disk_ellipse_v2(ref_image= ref_image,
                                                                                     input_image= image)
    # Detect optical cup params
    optical_cup_params = extract_optical_cup(binary_mask= segment_result["final_mask"])

    # Extract the report from the segment
    cdr_report = caculate_cdr_metrics(cup_ellipse= optical_cup_params,
                                      disk_ellipse= optical_disk_params,
                                      cup_mask= segment_result["final_mask"])
    ist_report = IST_rule(cup_ellipse= optical_cup_params,
                         disk_ellipse= optical_disk_params)
    return segment_result, optical_disk_params, optical_cup_params, cdr_report, ist_report, optical_disk_binary_map
    
def cdr_evaluation(cdr_report: Dict[str, any],
                   ist_report: Dict[str, any]):
    # The rule is based on this https://www.sciencedirect.com/science/article/pii/S0039625722001163
    vcdr = cdr_report["vertical_cdr"]
    acdr = cdr_report["area_cdr"]
    ist_satisfiled = ist_report["ist_satisfied"]

    red_area = []
    yellow_area = []
    green_area = []
    # Vertical CDR assesment
    if vcdr <= 0.5:
        green_area.append(f"Normal Vertical Cup-to-Disc Ratio: ({vcdr:.2f})")
    elif 0.5 < vcdr < 0.7:
        yellow_area.append(f"Elevated Vertical Cup-to-Disc Ratio: {vcdr:.2f}")
    else:
        red_area.append(f"High Vertical Cup-to-Disc Ratio: {vcdr:.2f}")
    
    # Area CDR assesment
    if acdr <= 0.3:
        green_area.append(f"Normal Area Cup-to-Disc Ratio: {acdr:.2f}")
    elif 0.3 < acdr <= 0.5:
        yellow_area.append(f"Elevated Area Cup-to-Disc Ratio: {acdr:.2f}")
    else:
        red_area.append(f"High Area Cup-to-Disc Ratio: {acdr:.2f}")
    
    # IST rules
    if ist_satisfiled:
        green_area.append(f"IST (inferior-Superior-Temporial) rule is satisfied")
    else:
        red_area.append(f"IST (inferior-Superior-Temporial) rule is not satisfied")
    
    # Classification
    if len(red_area) >= 2 or (len(red_area) >= 1 and len(yellow_area) >= 2):
        return "Glaucoma Suspicious", red_area, yellow_area, green_area
    elif len(red_area) >=1 or len(yellow_area) >= 2:
        return "Borderline Glaucoma", red_area, yellow_area, green_area
    else:
        return "Normal", red_area, yellow_area, green_area
    