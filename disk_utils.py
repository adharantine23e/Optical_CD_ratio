import cv2
import numpy as np


def isbright(image, dim=10, thresh=0.5):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L/np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh

def ref_based_color_normalization(input_image: np.ndarray, 
                                  reference_image: np.ndarray):
    
    # Based on the paper: https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/iet-ipr.2019.0969
    # In the paper they say: a healthy fundus image, in which vessels are in good contrast with the background, as the reference image
    
    # Convert image and reference ROI to L*a*b* color space
    lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    lab_reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)

    # Calculate mean value
    mean_l_ref, mean_a_ref, mean_b_ref = cv2.mean(lab_reference_image)[:3] 
    mean_l_input, mean_a_input, mean_b_input = cv2.mean(lab_image)[:3]

    # Calculate the difference in mean value
    diff_l = mean_l_ref - mean_l_input
    diff_a = mean_a_ref - mean_a_input
    diff_b = mean_b_ref - mean_b_input

    # Assert the diff to value
    normalized_lab_image = lab_image.astype(np.float32)
    normalized_lab_image[:, :, 0] += diff_l
    normalized_lab_image[:, :, 1] += diff_a
    normalized_lab_image[:, :, 2] += diff_b

    normalized_lab_image[:, :, 0] = np.clip(normalized_lab_image[:, :, 0], 0, 255)
    normalized_lab_image[:, :, 1] = np.clip(normalized_lab_image[:, :, 1], 0, 255) # a and b are typically -128 to 127, but cv2 stores them as 0-255
    normalized_lab_image[:, :, 2] = np.clip(normalized_lab_image[:, :, 2], 0, 255)

    # Convert back to bgr
    output_image = cv2.cvtColor(normalized_lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)    
    return output_image

def decrease_brightness(image: np.ndarray,
                        value: int = 50):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = cv2.subtract(v, value)  # Safe subtraction, won't go below 0
    # OR: v = np.clip(v.astype(int) - value, 0, 255).astype(np.uint8)
    
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def preprocessing_optical_disk(image: np.ndarray,
                            ball_shaped_radius: int):
    # Check the image brightness
    is_bright = isbright(image= image)
    if is_bright:
        print(f"The image is bright or not: {is_bright}")
        image = decrease_brightness(image= image)
    
    b, g, r = cv2.split(image)
    # Define the kernel size
    # Adaptive kernel sizing based on image dimensions
    img_size = min(image.shape[0], image.shape[1])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ball_shaped_radius, ball_shaped_radius))
    closing_size = max(int(img_size * 0.02), 9)  # 2% of image size, minimum 11
    kernel_closing = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (closing_size, closing_size)
        )
    # Apply top hat transform 
    tophat_image = cv2.morphologyEx(r, cv2.MORPH_TOPHAT, kernel)
    blackhat_image = cv2.morphologyEx(r, cv2.MORPH_BLACKHAT, kernel)
    # Image + tophat - blackhat
    enhanced_image = cv2.add(r, tophat_image)
    enhanced_image = cv2.subtract(enhanced_image, blackhat_image)
    # Remove small artifacts (vessels, holes) with opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    enhanced = cv2.morphologyEx(enhanced_image, cv2.MORPH_OPEN, kernel_open)
    
    # Apply closing to fill remaining gaps
    closing = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_closing)
    
    # # Gaussian blur to reduce noise before CLAHE
    # closing = cv2.GaussianBlur(closing, (3, 3), 1)
    
    # Median filter 
    closing = cv2.medianBlur(closing, 51)
    
    # Adaptive histogram equalization with adjusted parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    result = clahe.apply(closing)
    
    # Optional: Apply bilateral filter to preserve edges while smoothing
    result = cv2.bilateralFilter(result, 9, 75, 75)
    
    return result

def polar_transform_opencv(image, center, max_radius):
    """
    Convert image to polar coordinates using OpenCV's optimized function.
    
    Args:
        image: Input image (grayscale or color)
        center: Tuple (x, y) representing the center of OD
        max_radius: Maximum radius for polar transform
    
    Returns:
        Polar transformed image
    """
    # For newer OpenCV (4.x+), use warpPolar
    # For older OpenCV (3.x), use linearPolar
    
    try:
        # OpenCV 4.x method (recommended)
        polar_img = cv2.warpPolar(
            image, 
            dsize=(360, 360),  # (width=angles, height=radius)
            center=center,
            maxRadius=max_radius,
            flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR
        )
    except AttributeError:
        # Fallback for OpenCV 3.x
        polar_img = cv2.linearPolar(
            image,
            center=center,
            maxRadius=max_radius,
            flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR
        )
    
    return polar_img

def inverse_polar_transform_opencv(polar_image, center, max_radius, output_shape):
    """
    Convert polar image back to Cartesian using OpenCV.
    
    Args:
        polar_image: Polar image to convert
        center: Center point (x, y)
        max_radius: Maximum radius
        output_shape: Shape of output image (height, width)
    
    Returns:
        Cartesian image
    """
    try:
        # OpenCV 4.x
        cartesian_img = cv2.warpPolar(
            polar_image,
            dsize=(output_shape[1], output_shape[0]),  # (width, height)
            center=center,
            maxRadius=max_radius,
            flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    except AttributeError:
        # OpenCV 3.x
        cartesian_img = cv2.linearPolar(
            polar_image,
            center=center,
            maxRadius=max_radius,
            flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    
    return cartesian_img

def component_filtering(binary_polar: np.ndarray, 
                        left_ratio: float = 0.8,
                        min_area_ratio: float = 0.05):
    height, width = binary_polar.shape
    # Determine the left most position
    max_left_position = width * left_ratio

    # Find the largest connected part
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_polar, connectivity=8
    )
    
    if num_labels <= 1:
        return binary_polar
    
    # Filter the component
    result_mask = np.zeros_like(binary_polar)
    min_area = (height * width) * min_area_ratio

    valid_components = []
    for i in range(1, num_labels):
        x_left = stats[i, cv2.CC_STAT_LEFT]
        area = stats[1, cv2.CC_STAT_AREA]

        if x_left <= max_left_position and area >= min_area:
            valid_components.append((i, area))

    # Get the largest component
    best_idx = max(valid_components, key= lambda x: x[1])[0]
    result_mask = (labels == best_idx).astype(np.uint8) * 255
    return result_mask

def adaptive_strip_binarization(polar_image, num_strips=10, method='adaptive'):
    """
    Divide polar image into horizontal strips and apply binarization.
    
    Args:
        polar_image: Polar transformed image (grayscale)
        num_strips: Number of horizontal strips
        method: 'adaptive', 'otsu', or 'mean'
    
    Returns:
        Binarized polar image
    """
    height, width = polar_image.shape
    strip_height = height // num_strips
    binary_result = np.zeros_like(polar_image)
    
    for i in range(num_strips):
        start_row = i * strip_height
        end_row = start_row + strip_height if i < num_strips - 1 else height
        strip = polar_image[start_row:end_row, :]
        
        if method == 'adaptive':
            # Adaptive Gaussian thresholding
            binary_strip = cv2.adaptiveThreshold(
                strip, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11, C=2
            )
        elif method == 'otsu':
            # Otsu's method per strip
            _, binary_strip = cv2.threshold(
                strip, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:  # mean
            # Mean threshold per strip
            strip_mean = np.mean(strip)
            _, binary_strip = cv2.threshold(
                strip, strip_mean, 255,
                cv2.THRESH_BINARY
            )
        
        binary_result[start_row:end_row, :] = binary_strip
    filtered_result = component_filtering(binary_polar= binary_result,
                                          left_ratio= 0.8,
                                          min_area_ratio= 0.05)
    return filtered_result

def process_od_image(orig_img, enhanced, od_center, od_radius, num_strips=10, 
                     polar_size: tuple[int] = (100, 360), binarization_method='otsu'):
    """
    Complete pipeline for OD extraction.
    
    Args:
        image_path: Path to fundus image
        od_center: Tuple (x, y) of OD center
        od_radius: Estimated radius of OD
        num_strips: Number of strips for binarization
    """

    
    # Apply polar transform
    polar_img = polar_transform_opencv(enhanced, od_center, od_radius)
    
    # Resize polar image if needed
    if polar_img.shape != polar_size[::-1]:  # OpenCV uses (height, width)
        polar_img = cv2.resize(polar_img, polar_size)

    # Apply strip-wise binarization
    binary_polar = adaptive_strip_binarization(
        polar_img, num_strips, method=binarization_method
    )
    
    # Convert back to Cartesian (optional)
    binary_cartesian = inverse_polar_transform_opencv(
        binary_polar, od_center, od_radius, enhanced.shape
    )
    return enhanced, polar_img, binary_polar, binary_cartesian

def post_processing(binary_mask: np.ndarray, kernel_size: int = 1, iterations: int = 1):
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8  # Fixed: binary_mask not binary_map
    )
    
    # If no components found (empty mask), return empty
    if num_labels <= 1:  # Only background
        return np.zeros_like(binary_mask)
    
    # Find the largest component - SIMPLIFIED AND CORRECT
    largest_component_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_component_idx).astype(np.uint8) * 255
    
    print(f"Original mask non-zero pixels: {np.count_nonzero(binary_mask)}")
    print(f"Largest mask non-zero pixels: {np.count_nonzero(largest_mask)}")
    print(f"Number of components found: {num_labels - 1}")
    print(f"Largest component area: {stats[largest_component_idx, cv2.CC_STAT_AREA]}")
    
    # kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    # eroded_mask = cv2.erode(largest_mask, kernel, iterations=iterations)
    
    # # Get the edge
    # mark = cv2.subtract(eroded_mask, largest_mask)
    
    return largest_mask

def fit_ellipse_direct_least_squares(edge_points):
    """
    Fit ellipse using Direct Least Squares method.
    
    Args:
        edge_points: Nx2 array of edge pixel coordinates (x, y)
    
    Returns:
        Ellipse parameters: (center_x, center_y, major_axis, minor_axis, angle)
        or None if fitting fails
    """
    try:
        # OpenCV's fitEllipse uses least squares internally
        ellipse = cv2.fitEllipse(edge_points)
        # Returns: ((center_x, center_y), (width, height), angle)
        return ellipse
    except:
        return None

def fit_ellipse_on_od_edges(edges):
    """
    Extract edge pixel coordinates and fit ellipse.
    
    Args:
        edges: Binary edge image
    
    Returns:
        Ellipse parameters or None
    """
    # # Find non-zero pixels (edge points)
    # edge_points = np.column_stack(np.where(edges > 0))
    
    # # Convert from (row, col) to (x, y)
    # edge_points = np.flip(edge_points, axis=1).astype(np.float32)
    
    contours,hierarchy = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

    # select the first contour
    cnt = contours[0]
    if len(cnt) < 6:
        return None
    
    return fit_ellipse_direct_least_squares(cnt)

def extract_optical_disk_v1(image: np.ndarray):
    """
    Detect optical disk and calculate its radius using red channel analysis.
    
    Args:
        image: Image from cv2.imread
        show_steps: Whether to display intermediate processing steps
    
    Returns:
        center: (x, y) coordinates of optical disk center
        radius: Radius of the optical disk in pixels
        result_image: Annotated image with detected disk
    """

    height, width = image.shape[:2]
    # Extract the red channel
    red_channel = image[:, :, 2]
    # Apply CLAHE  
    clahe = cv2.createCLAHE(clipLimit= 2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(red_channel)
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)   

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        raise ValueError("No contours found in the image")
    
    # Filter contours that are reasonably centered and sized
    valid_contours = []
    for contour in contours:
        (x, y), r = cv2.minEnclosingCircle(contour)
        area = cv2.contourArea(contour)
        
        # Filter criteria:
        # 1. Center should be within image bounds with margin
        # 2. Reasonable size (e.g., 1-20% of image area)
        # 3. Circularity check
        margin = 50
        if (margin < x < width - margin and 
            margin < y < height - margin and
            0.01 * height * width < area < 0.2 * height * width):
            
            # Check circularity (area vs perimeter ratio)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.7:  # Circle should be ~1.0
                    valid_contours.append((contour, area))
    
    if len(valid_contours) == 0:
        # Fallback to largest contour with bounds checking
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        # Get largest valid contour
        largest_contour = max(valid_contours, key=lambda x: x[1])[0]
    
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    
    # Clamp to bounds
    center_x = int(np.clip(x, 0, width - 1))
    center_y = int(np.clip(y, 0, height - 1))
    max_allowed_radius = min(center_x, center_y, width - center_x, height - center_y)
    radius = int(min(radius, max_allowed_radius))
    
    return center_x, center_y, radius

def detect_optical_disk_ellipse_v2(ref_image: np.ndarray, 
                                   input_image: np.ndarray,
                                   ball_shaped_radius: int = 151,
                                   num_strips: int = 15,
                                   binarization_method: str = "otsu"
                                   ) -> tuple[any]:
    image = ref_based_color_normalization(input_image= input_image,
                                          reference_image= ref_image)
    
    # Preprocess the optical disk
    preprocessed_image = preprocessing_optical_disk(image= image,
                                                    ball_shaped_radius= ball_shaped_radius)
    
    # Binarization the preprocessed image
    enhanced, polar_img, binary_polar, binary_map = process_od_image(
        orig_img= image,
        enhanced= preprocessed_image,
        od_center= (int(preprocessed_image.shape[0]// 2), int(preprocessed_image.shape[1] // 2)),
        od_radius= int(preprocessed_image.shape[0] // 2),
        num_strips= num_strips,
        binarization_method= binarization_method)
    
    # Post-processing
    mark = post_processing(binary_mask= binary_map)
    # Transform it to ellipse
    ellipse = fit_ellipse_on_od_edges(edges= mark)
    return ellipse, mark

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ref_image = cv2.imread("/mnt/data/shared_trainee/dungplq/optical_cup_segmentation/cup_disk_ratio_api/reference_image/CTEH-003617.jpg.png")
    input_image = cv2.imread("/mnt/data/shared_trainee/dungplq/optical_cup_segmentation/cup_disk_ratio_api/20230419113331090.jpg.png")

    image = ref_based_color_normalization(input_image= input_image,
                                          reference_image= ref_image)
    
    # Preprocess the optical disk
    preprocessed_image = preprocessing_optical_disk(image= image,
                                                    ball_shaped_radius= 151)
    
    # Binarization the preprocessed image
    enhanced, polar_img, binary_polar, binary_map = process_od_image(
        orig_img= image,
        enhanced= preprocessed_image,
        od_center= (int(preprocessed_image.shape[0]// 2), int(preprocessed_image.shape[1] // 2)),
        od_radius= int(preprocessed_image.shape[0] // 2),
        num_strips= 15,
        binarization_method= "otsu")
    
    # Post-processing
    mark = post_processing(binary_mask= binary_map)
    # Transform it to ellipse
    ellipse = fit_ellipse_on_od_edges(edges= mark)

    fig, axes = plt.subplots(2, 3, figsize = (15, 15))

    axes[0, 0].imshow(image)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Color reference image")
    
    axes[0, 1].imshow(preprocessed_image)
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Preprocessed image")

    axes[0, 2].imshow(polar_img)
    axes[0, 2].axis("off")
    axes[0, 2].set_title("Polar image")

    axes[1, 0].imshow(binary_polar)
    axes[1, 0].axis("off")
    axes[1, 0].set_title("Binary Polar image")

    axes[1, 1].imshow(mark)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Final image")

    axes[1, 2].imshow(enhanced)
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Enhanced image")

    plt.tight_layout()
    plt.savefig("Disk_segmentation.png")