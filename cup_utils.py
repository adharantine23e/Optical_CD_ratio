import os
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.filters import threshold_otsu, threshold_local, threshold_yen
from sklearn.svm import SVC
from scipy.ndimage import uniform_filter
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional

############################## LOSS CALCULATION ##############################
def get_decision_value(model: SVC, features: np.ndarray) -> np.ndarray:
    decision_value = model.decision_function(features)
    return decision_value

def map_decision_value(decision_value: np.ndarray, slic_segments: np.ndarray) -> np.ndarray:
    height = slic_segments.shape[0]
    width = slic_segments.shape[1]
    pixel_decision_map = np.zeros((height, width), dtype=np.float32)
    for segID, segVal in enumerate(np.unique(slic_segments)):
        mask = (slic_segments == segVal)
        pixel_decision_map[mask] = decision_value[segID]
    return pixel_decision_map

def mean_filter(decision_map: np.ndarray, filter_size: int = 3) -> np.ndarray:
    smoothed_map = uniform_filter(decision_map, size=filter_size, mode='reflect')    
    return smoothed_map

def binary_segment(decision_map: np.ndarray, 
                   high_value_threshold: float = 0.55,
                   proportion_cutoff: float = 0.07,
                   threshold: float= 90) -> np.ndarray:
    values = decision_map.flatten()
    
    # Normalize to 0-1 range for comparison
    normalized = (values - values.min()) / (values.max() - values.min())
    
    # Count proportion of high values
    high_proportion = np.sum(normalized > high_value_threshold) / len(values)
    print("High proportion: ", high_proportion)
    if high_proportion > proportion_cutoff:
        # Many high values on right → use percentile
        threshold_value = np.percentile(values, threshold)
        method = "percentile"
    else:
        # Few high values → use Otsu
        threshold_value = threshold_otsu(decision_map)
        method = "otsu"
        
    # threshold_value = threshold_otsu(decision_map)
    binary_map = (decision_map > threshold_value).astype(np.uint8)
    print(f"High values: {high_proportion:.1%}, Method: {method}, Threshold: {threshold_value:.2f}")
    return binary_map

############################## POST PROCESSING ##############################

def segment_optic_disc_cup(
                           features: np.ndarray,
                           slic_segments: np.ndarray,
                           svm_model: SVC,
                           filter_size: int = 5,
                           threshold: float = 80,
                           apply_ellipse: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete pipeline for optic disc/cup segmentation.
    
    Args:
        image_path: Path to fundus image
        features: Extracted features for superpixels
        slic_segments: SLIC segmentation map
        svm_model: Trained SVM model
        filter_size: Size of mean filter (default: 5)
        threshold: Decision threshold (default: 0.0)
        
    Returns:
        Tuple of (binary_mask, smoothed_decision_map, raw_decision_map)
    """
    # 1. Get decision values from SVM for each superpixel
    decision_values = get_decision_value(svm_model, features)
    
    # 2. Map decision values to all pixels
    pixel_decision_map = map_decision_value(decision_values, slic_segments)
    
    # 3. Apply mean filter for smoothing
    smoothed_decision_map = mean_filter(pixel_decision_map, filter_size)
    # 4. Get binary segmentation
    binary_mask = binary_segment(smoothed_decision_map, threshold= threshold)

    #5. Post processing
    final_mask, info = postprocess_segmentation(binary_mask, apply_ellipse)
    
    return {
        'final_mask': final_mask,
        'raw_mask': binary_mask,
        'largest_component': info.get('largest_component'),
        'ellipse_params': info.get('ellipse_params'),
        'smoothed_decision_map': smoothed_decision_map,
        'raw_decision_map': pixel_decision_map
    }

def get_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract the largest connected component from binary mask.
    
    Args:
        binary_mask: Binary segmentation mask (height, width)
        
    Returns:
        Binary mask with only largest connected component
    """
    # Label connected components
    labeled_mask, num_components = label(binary_mask)
    
    if num_components == 0:
        return binary_mask
    
    # Find the largest component
    component_sizes = np.bincount(labeled_mask.ravel())
    # Ignore background (label 0)
    component_sizes[0] = 0
    
    largest_component_label = component_sizes.argmax()
    
    # Create mask with only largest component
    largest_component_mask = (labeled_mask == largest_component_label).astype(np.uint8)
    return largest_component_mask

def fit_ellipse_to_mask(binary_mask: np.ndarray) -> tuple:
    """
    Fit an ellipse to the binary mask boundary.
    
    Args:
        binary_mask: Binary mask (height, width)
        
    Returns:
        Tuple of (ellipse_mask, ellipse_params)
        - ellipse_mask: Binary mask of fitted ellipse
        - ellipse_params: ((center_x, center_y), (width, height), angle)
    """
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return binary_mask, None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit ellipse (need at least 5 points)
    if len(largest_contour) < 5:
        return binary_mask, None
    
    # Fit ellipse: returns ((center_x, center_y), (width, height), angle)
    ellipse_params = cv2.fitEllipse(largest_contour)
    
    # Create ellipse mask
    ellipse_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse_params, color=1, thickness=-1)  # Filled ellipse
    
    return ellipse_mask, ellipse_params

def postprocess_segmentation(binary_mask: np.ndarray, 
                            apply_ellipse_fitting: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Post-process segmentation mask as described in the paper:
    1. Extract largest connected component
    2. Fit ellipse to the boundary
    
    Args:
        binary_mask: Raw binary segmentation mask
        apply_ellipse_fitting: Whether to apply ellipse fitting
        
    Returns:
        Tuple of (final_mask, info_dict)
    """
    info = {}
    
    # Step 1: Get largest connected component
    largest_component = get_largest_connected_component(binary_mask)
    info['largest_component'] = largest_component
    
    if not apply_ellipse_fitting:
        return largest_component, info
    
    # Step 2: Fit ellipse
    ellipse_mask, ellipse_params = fit_ellipse_to_mask(largest_component)
    info['ellipse_params'] = ellipse_params
    info['raw_boundary'] = largest_component
    
    if ellipse_params is None:
        print("Warning: Could not fit ellipse, returning largest component")
        return largest_component, info
    
    return ellipse_mask, info

def draw_ellipse_boundary(image: np.ndarray, 
                         ellipse_params: tuple,
                         color: tuple = (0, 255, 0),
                         thickness: int = 2) -> np.ndarray:
    """
    Draw ellipse boundary on image.
    
    Args:
        image: Input image
        ellipse_params: Ellipse parameters from cv2.fitEllipse
        color: Color for drawing (B, G, R)
        thickness: Line thickness
        
    Returns:
        Image with ellipse drawn
    """
    result = image.copy()
    if ellipse_params is not None:
        cv2.ellipse(result, ellipse_params, color, thickness)
    return result

def extract_optical_cup(binary_mask: np.ndarray) -> Tuple:

    contour, _ = cv2.findContours(binary_mask.astype(np.uint8),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    # Get the largest contour
    contour = max(contour, key = cv2.contourArea)
    # Fit an ellipse to the contour
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        # Extract params
        (center_x, center_y), (width, height), angle = ellipse
        return ((int(center_x), int(center_y)), (int(width), int(height)), int(angle))
    else:
        print("Not enough points to fit an ellipse")
        return ((None, None), (None, None), None) 
    

def caculate_cdr_metrics(cup_ellipse: Tuple,
                         disk_ellipse: Tuple,
                         cup_mask: Optional[np.ndarray]) -> Dict:
    
    # Extract cup params
    cup_center, cup_size, cup_angle = cup_ellipse
    cup_width = cup_size[0]
    cup_height = cup_size[1]

    # Extract disk params   
    disk_center, disk_size, disk_angle = disk_ellipse
    disk_width = disk_size[0]
    disk_height = disk_size[1]
    
    # CDR ratios
    vertical_CDR = cup_height / disk_height
    horizontal_CDR = cup_width / disk_width

    # RDR ratios
    vertical_RDR = 1.0 - vertical_CDR
    horizontal_RDR = 1.0 - horizontal_CDR

    # Area-based calculations
    # Area-based calculations
    disk_area_ellipse = np.pi * (disk_width / 2) * (disk_height / 2)
    cup_area_ellipse = np.pi * (cup_width / 2) * (cup_height / 2)
    
    area_CDR = cup_area_ellipse / disk_area_ellipse
    area_RDR = 1.0 - area_CDR

    return {
        "vertical_cdr": vertical_CDR,
        "horizontal_cdr": horizontal_CDR,
        "area_cdr": area_CDR,

        "vertical_rdr": vertical_RDR,
        "horizontal_rdr": horizontal_RDR,
        "area_rdr": area_RDR,

        "cup_area": cup_area_ellipse,
        "disk_area": disk_area_ellipse,
        "cup_disk_center_difference": np.sqrt((cup_center[0] - disk_center[0]) **2 + (cup_center[1] - disk_center[1])**2)
    }

def IST_rule(cup_ellipse: Tuple,
             disk_ellipse: Tuple[int],
             num_angle: int = 360) -> Dict:
    # We use an alternate of ISNT rule since based on this paper https://pmc.ncbi.nlm.nih.gov/articles/PMC5705386/
    # IST rule is better

    # Extract params
    (cup_center_x, cup_center_y), (cup_width, cup_height), cup_angle = cup_ellipse
    cup_a = cup_width / 2
    cup_b = cup_height / 2
    cup_angle_rad = np.deg2rad(cup_angle)

    (disk_center_x, disk_center_y), (disk_width, disk_height), disk_angle = disk_ellipse
    disk_a = disk_width / 2
    disk_b = disk_height / 2
    disk_angle_rad = np.deg2rad(disk_angle)

    
    angles = np.linspace(0, 2*np.pi, num_angle)
    rim_widths = []

    for theta in angles:
        # Calculate disk boundary point at angle theta
        theta_disk_rotated = theta - disk_angle_rad

        # Point on disk ellipse boundary
        disk_x = disk_center_x + disk_a * np.cos(theta_disk_rotated) * np.cos(disk_angle_rad) - disk_b * np.sin(theta_disk_rotated) * np.sin(disk_angle_rad)
        disk_y = disk_center_y + disk_a * np.cos(theta_disk_rotated) * np.sin(disk_angle_rad) + disk_b * np.sin(theta_disk_rotated) * np.cos(disk_angle_rad) 

        # Calculate distance from disk center to disk boundary
        disk_radius_angle = np.sqrt((disk_x - disk_center_x) ** 2 + (disk_y - disk_center_y) ** 2)

        # Calculate ray distance
        ray_dx = disk_x - disk_center_x
        ray_dy = disk_y - disk_center_y
        ray_length = np.sqrt(ray_dx**2 + ray_dy**2)    
        
        if ray_length > 0:
            ray_dx /= ray_length
            ray_dy /= ray_length
        else:
            rim_widths.append(0)
            continue
        
        # Transform to cup ellipse coordinate
        dx = disk_center_x - cup_center_x
        dy = disk_center_y - cup_center_y

        # Rotate to align with cup ellipse axes
        cos_a = np.cos(-cup_angle_rad)
        sin_a = np.sin(-cup_angle_rad)
        
        dx_rotated = dx * cos_a - dy * sin_a
        dy_rotated = dx * sin_a + dy * cos_a
        
        ray_dx_rotated = ray_dx * cos_a - ray_dy * sin_a
        ray_dy_rotated = ray_dx * sin_a + ray_dy * cos_a

        A = (ray_dx_rotated / cup_a)**2 + (ray_dy_rotated / cup_b)**2
        B = 2 * ((dx_rotated * ray_dx_rotated) / cup_a**2 + 
                 (dy_rotated * ray_dy_rotated) / cup_b**2)
        C = (dx_rotated / cup_a)**2 + (dy_rotated / cup_b)**2 - 1

        discriminant = B**2 - 4*A*C
        if discriminant >= 0 and A > 0:
            # Two solutions - take the positive one closer to disk center
            t1 = (-B + np.sqrt(discriminant)) / (2*A)
            t2 = (-B - np.sqrt(discriminant)) / (2*A)
            
            # Choose the intersection on the ray direction (positive t)
            valid_t = [t for t in [t1, t2] if t > 0]
            if valid_t:
                cup_distance = min(valid_t)
            else:
                cup_distance = 0
        else:
            cup_distance = 0
        
        # Calculate rim width
        rim_width = disk_radius_angle - cup_distance
        rim_widths.append(max(0, rim_width))
    
    # Convert to np.array
    rim_widths = np.array(rim_widths)
    
    # IST rule: Inferior > Superior > Temporal (0 degree = right/ temporal)
    superior_idx = int(num_angle * 270 /360) #  Top
    inferior_idx = int(num_angle * 90 / 360) # Bottom
    temporal_idx = 0 # Right
    
    return {
        "mean_rim_widths": np.mean(rim_widths),
        "inferior_rim": rim_widths[inferior_idx],
        "superior_rim": rim_widths[superior_idx],
        "temporal_rim": rim_widths[temporal_idx],
        "ist_satisfied": (
            rim_widths[inferior_idx] > rim_widths[superior_idx] > rim_widths[temporal_idx]
        )        
    }

def create_enhanced_visualization(
    original_image: np.ndarray,
    segmentation_mask: np.ndarray,
    ellipse_params: Optional[tuple] = None,
) -> np.ndarray:
    """
    Create visualization with only the optical cup boundary (no fill).
    
    Args:
        original_image: Original RGB image
        segmentation_mask: Binary segmentation mask
        ellipse_params: Optional ellipse parameters for boundary
        
    Returns:
        Visualization image with only boundary lines
    """
    result = original_image.copy()
    
    # Find contours of the segmentation mask
    contours, _ = cv2.findContours(segmentation_mask.astype(np.uint8), 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw only the contours (boundary) in green
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # Draw ellipse boundary if provided (in red)
    if ellipse_params is not None:
        result = draw_ellipse_boundary(result, ellipse_params, 
                                      color=(255, 0, 0), thickness=2)
    
    return result

def create_comprehensive_subplot(
    original_image: np.ndarray,
    smoothed_decision_map: np.ndarray,
    binary_mask: np.ndarray,
    disk_binary_mask: np.ndarray,
    ellipse_params: Optional[tuple] = None,
    disk_ellipse: Optional[tuple] = None,
    cdr_metrics: Optional[Dict] = None,
    ist_report: Optional[Dict] = None,
    save_path: Optional[str] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create a 2x2 subplot showing all stages of the segmentation process.
    
    Args:
        original_image: Original BGR image
        smoothed_decision_map: Smoothed decision values
        binary_mask: Binary segmentation mask
        ellipse_params: Ellipse parameters for optical cup
        od_circle: Optical disk circle parameters (cx, cy, radius)
        save_path: Path to save the figure
        dpi: DPI for saving figure
        
    Returns:
        matplotlib Figure object
    """
    if ellipse_params is not None and disk_ellipse is not None and cdr_metrics is not None:
        cup_center, cup_size, cup_angle = ellipse_params
        disk_center, disk_size, disk_angle = disk_ellipse
        cup_x, cup_y = int(cup_center[0]), int(cup_center[1])
        disk_x, disk_y = int(disk_center[0]), int(disk_center[1])
        cup_width, cup_height = cup_size
        disk_width, disk_height = disk_size
        # Convert angles to radians
        cup_angle_rad = np.deg2rad(cup_angle)
        disk_angle_rad = np.deg2rad(disk_angle)

    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    fig.suptitle('Optical Cup Segmentation Pipeline', fontsize=16, fontweight='bold')
    
    # 1. Original Image
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. Smoothed Decision Map
    im = axes[0, 1].imshow(smoothed_decision_map, cmap='jet')
    axes[0, 1].set_title('Smoothed Decision Map', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 3. Binary Mask
    # axes[1, 0].hist(smoothed_decision_map.flatten(), bins = 30, color = "skyblue", edgecolor = "black")
    # axes[1, 0].set_xlabel("Smooth decision value")
    # axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].imshow(binary_mask, cmap='gray')
    axes[1, 0].set_title('Cup Binary Mask', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    
    axes[1, 1].imshow(disk_binary_mask, cmap='gray')
    axes[1, 1].set_title('Disk Binary Mask', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    original_rgb_cp = original_rgb.copy()
    if disk_ellipse is not None:
        cv2.ellipse(original_rgb_cp, disk_ellipse, (255, 255, 0), 2)
        cv2.ellipse(original_rgb_cp, ellipse_params, (0, 255, 0), 2)
    # Helper function to get point on ellipse at angle
    def get_ellipse_point(cx, cy, a, b, angle_rad, theta):
        """Get point on ellipse at angle theta"""
        theta_rotated = theta - angle_rad
        x = cx + a * np.cos(theta_rotated) * np.cos(angle_rad) - \
            b * np.sin(theta_rotated) * np.sin(angle_rad)
        y = cy + a * np.cos(theta_rotated) * np.sin(angle_rad) + \
            b * np.sin(theta_rotated) * np.cos(angle_rad)
        return int(x), int(y)
    
    # Define key angles (Temporal=0°, Superior=90°, Inferior=270°)
    key_angles = {
        'Temporal': 0,
        'Inferior': np.pi / 2,
        'Superior': 3 * np.pi / 2
    }
    
    colors = {
        'Temporal': (255, 0, 255),    # Magenta
        'Superior': (0, 255, 255),    # Cyan
        'Inferior': (255, 165, 0)     # Orange
    }
    
    # Draw rim width lines at key positions
    for label, theta in key_angles.items():
        # Disk point
        disk_pt = get_ellipse_point(disk_x, disk_y, disk_width/2, disk_height/2, 
                                     disk_angle_rad, theta)
        # Cup point (approximate)
        cup_pt = get_ellipse_point(cup_x, cup_y, cup_width/2, cup_height/2,
                                    cup_angle_rad, theta)
        
        color = colors[label]
        
        # Draw line from cup to disk
        cv2.line(original_rgb_cp, cup_pt, disk_pt, color, 2)
        
        # Draw arrows
        cv2.arrowedLine(original_rgb_cp, cup_pt, disk_pt, color, 2, tipLength=0.1)
        
        # Add text label
        text_pos = (disk_pt[0] -50, disk_pt[1] - 10)
        rim_value = ist_report[f'{label.lower()}_rim']
        cv2.putText(original_rgb_cp, f'{label}: {rim_value:.1f}px', 
                   text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 
                   (255, 255, 255), 1)
    

    axes[2, 0].imshow(original_rgb_cp)
    axes[2, 0].set_title(f"IST rule: I: {ist_report['inferior_rim']:.1f} || S: {ist_report['superior_rim']:.1f} || T: {ist_report['temporal_rim']:.1f}", fontsize = 14, fontweight="bold")
    axes[2, 0].axis("off")


    # 4. Final Result with Optical Cup boundary and Optical Disk circle
    result_with_boundaries = create_enhanced_visualization(
        original_image, binary_mask, ellipse_params
    )
    
    # Draw optical disk circle if detected
    if disk_ellipse is not None:
        cv2.ellipse(result_with_boundaries, disk_ellipse, (255, 255, 0), 2)  # Cyan circle
        center = tuple(map(int, disk_ellipse[0]))
        cv2.circle(result_with_boundaries, center, 2, (255, 255, 0), -1)  # Center point
    
    if ellipse_params is not None and disk_ellipse is not None and cdr_metrics is not None:

        # Calculate vertical line endpoints (perpendicular to horizontal)
        v_angle = np.pi / 2
        disk_vertical_half = disk_height / 2
        disk_v_top = (
            int(disk_x + disk_vertical_half * np.cos(v_angle + disk_angle_rad)),
            int(disk_y + disk_vertical_half * np.sin(v_angle + disk_angle_rad))
        )
        disk_v_bottom = (
            int(disk_x - disk_vertical_half * np.cos(v_angle + disk_angle_rad)),
            int(disk_y - disk_vertical_half * np.sin(v_angle + disk_angle_rad))
        )

        cup_vertical_half = cup_height / 2
        cup_v_top = (
            int(cup_x + cup_vertical_half * np.cos(v_angle + cup_angle_rad)),
            int(cup_y + cup_vertical_half * np.sin(v_angle + cup_angle_rad))
        )
        cup_v_bottom = (
            int(cup_x - cup_vertical_half * np.cos(v_angle + cup_angle_rad)),
            int(cup_y - cup_vertical_half * np.sin(v_angle + cup_angle_rad))
        )

        # Calculate horizontal line endpoints
        h_angle = 0
        
        # Disk horizontal endpoints
        disk_h_half = disk_width / 2
        disk_h_right = (
            int(disk_x + disk_h_half * np.cos(h_angle + disk_angle_rad)),
            int(disk_y + disk_h_half * np.sin(h_angle + disk_angle_rad))
        )
        disk_h_left = (
            int(disk_x - disk_h_half * np.cos(h_angle + disk_angle_rad)),
            int(disk_y - disk_h_half * np.sin(h_angle + disk_angle_rad))
        )
        
        # Cup horizontal endpoints
        cup_h_half = cup_width / 2
        cup_h_right = (
            int(cup_x + cup_h_half * np.cos(h_angle + cup_angle_rad)),
            int(cup_y + cup_h_half * np.sin(h_angle + cup_angle_rad))
        )
        cup_h_left = (
            int(cup_x - cup_h_half * np.cos(h_angle + cup_angle_rad)),
            int(cup_y - cup_h_half * np.sin(h_angle + cup_angle_rad))
        )

        # Draw lines with arrows
        # Vertical CDR (Magenta for disk, Red for cup)
        cv2.arrowedLine(result_with_boundaries, disk_v_top, disk_v_bottom, 
                       (255, 0, 255), 2, tipLength=0.03)  # Magenta
        cv2.arrowedLine(result_with_boundaries, disk_v_bottom, disk_v_top, 
                       (255, 0, 255), 2, tipLength=0.03)
        
        cv2.arrowedLine(result_with_boundaries, cup_v_top, cup_v_bottom, 
                       (0, 0, 255), 2, tipLength=0.05)  # Red
        cv2.arrowedLine(result_with_boundaries, cup_v_bottom, cup_v_top, 
                       (0, 0, 255), 2, tipLength=0.05)
        
        # Horizontal CDR (Cyan for disk, Blue for cup)
        cv2.arrowedLine(result_with_boundaries, disk_h_left, disk_h_right, 
                       (255, 255, 0), 2, tipLength=0.03)  # Cyan
        cv2.arrowedLine(result_with_boundaries, disk_h_right, disk_h_left, 
                       (255, 255, 0), 2, tipLength=0.03)
        
        cv2.arrowedLine(result_with_boundaries, cup_h_left, cup_h_right, 
                       (255, 0, 0), 2, tipLength=0.05)  # Blue
        cv2.arrowedLine(result_with_boundaries, cup_h_right, cup_h_left, 
                       (255, 0, 0), 2, tipLength=0.05)
        # Add text annotations
        vertical_cdr = cdr_metrics.get('vertical_cdr', 0)
        horizontal_cdr = cdr_metrics.get('horizontal_cdr', 0)
        
        # Position text near the lines
        text_offset = 5
        cv2.putText(result_with_boundaries, f'V-CDR: {vertical_cdr:.1f}', 
                   (disk_v_top[0] + text_offset, disk_v_top[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(result_with_boundaries, f'H-CDR: {horizontal_cdr:.1f}', 
                   (disk_h_right[0] + text_offset, disk_h_right[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    result_rgb = cv2.cvtColor(result_with_boundaries, cv2.COLOR_BGR2RGB)
    axes[2, 1].imshow(result_rgb)
    axes[2, 1].set_title('Final Result (Cup: Blue, Disk: Cyan). Vertical CDR: Disk_height (Magneta) / Cup_height (Red)', fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved comprehensive visualization to: {save_path}")
    
    return fig