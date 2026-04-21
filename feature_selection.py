from typing import List, Tuple, Dict
import numpy as np
import cv2
from skimage.segmentation import slic, mark_boundaries
from skimage import io, img_as_float
import matplotlib.pyplot as plt

def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    # Ensure the image has only one channel
    if len(image.shape) > 2:
        print("Image has more than one channel. Only the first channel will be processed.")

    # Apply histogram equalization
    clahe_cup = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_image = clahe_cup.apply(image)
    return enhanced_image

def dyalic_gaussian_pyramid(image: np.ndarray, scale: int = 9) -> List[np.ndarray]:
    # Supposedly it would give the np.array of (num_scale, img_height, img_width)
    """
    Level 1: 1:1
    Level 2: 1:2
    ....
    Level 8: 1:256
    """
    paramids = []
    ratios = []

    current_img = image.copy()
    for level in range(scale):
        paramids.append(current_img)

        # Scale down the image
        ratio = 2 ** level
        if level < scale -1:
            current_img = cv2.pyrDown(current_img)
    
    return paramids
    
def segment_hist_calculation(slic_segments: np.ndarray, image_chosen: np.ndarray,
                             ) -> np.ndarray:
    # Supposedly it should give out a 2D array: (n_segments, 256)
    histogram_values = []
    # Access each segmnent from the slic
    for (segID, segVal) in enumerate(np.unique(slic_segments)):
        mask = np.zeros(image_chosen.shape[:2], dtype = "uint8")
        mask[slic_segments == segVal] == 255

        # Get the segment area based on the mask
        segemented_area = cv2.bitwise_and(image_chosen, image_chosen, mask = mask)
        # Apply histogram equalization to the segmented area
        equalized_segmented_area = apply_histogram_equalization(segemented_area)
        # Calculate the histogram 
        hist_segment = cv2.calcHist([equalized_segmented_area], [0], None, [256], [0, 256])
        # Change the shape from [256, 1] to [256]
        hist_segment = hist_segment.flatten()
        # Append the histogram to the list
        histogram_values.append(hist_segment)
    
    # Convert to numpy array
    histogram_values = np.array(histogram_values)
    return histogram_values

def feature_extraction(image: np.ndarray, n_segments: int = 100, compactness: int= 20) -> np.ndarray:
    # image = cv2.imread(image_path)
    orig_height = image.shape[0]
    orig_width = image.shape[1]

    #### CONTRAST ENHANCED HISTOGRAM ####
    # Extract the green, blue, hue, saturation channel from the image
    blue, green, _ = cv2.split(image)
    hue, saturation, _  = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    # SLIC segments with a total of 100 segments and compactness of 20
    slic_segments = slic(img_as_float(image), n_segments = n_segments, compactness= compactness)

    # Get the histogram values
    blue_hist = segment_hist_calculation(slic_segments, blue)
    green_hist = segment_hist_calculation(slic_segments, green)
    hue_hist = segment_hist_calculation(slic_segments, hue)
    saturation_hist = segment_hist_calculation(slic_segments, saturation)
    # Concatenate the histogram values to get the feature vector the shape should be (n_segments, 256 * 4)
    ceh_features = np.concatenate((blue_hist, green_hist, hue_hist, saturation_hist), axis = 1)
    
    #### CENTRE SURROUND STATISTICS ####
    # In the paper, the combinations are (2, 5), (2, 6), (3, 6), (3, 7), (4, 7) and (4, 8)
    combinations = [(2, 5), (2, 6), (3, 6), (3, 7), (4, 7), (4, 8)]
    # Extract 9 features map from dyalic gaussian pyramid wth each chosen map
    map_features_blue = extract_map_features(blue, combinations, orig_height, orig_width)
    map_features_green = extract_map_features(green, combinations, orig_height, orig_width)
    # Combine the 2 feature maps to get the css features (total: 12 features)
    map_features = np.concatenate((map_features_blue, map_features_green), axis = 0)
    # Compute the mean and standard deviation of the css features for each superpixel
    css_features = []
    for (segID, segVal) in enumerate(np.unique(slic_segments)):
        mask = (slic_segments == segVal)
        superpixel_features = []
        for diff_map in map_features:
            pixel_value = diff_map[mask]
            # Compute the first moment (mean) and second moment (variance)
            mean_value = np.mean(pixel_value)   
            var_value = np.var(pixel_value)
            superpixel_features.extend([mean_value, var_value])
        css_features.append(superpixel_features)
    # Convert to np array. The shape should be (n_segments, 24) As: 24 = 12 (number of map) * 2 (mean and variance)
    css_features = np.array(css_features)
    #Expand it with the 4 neighboring pixels the shape should be (n_segments, 120) 120 = 24 (map_features) * (4 (neighboring pixels) + 1 (current pixels))
    expanded_css_features = expand_css_neighbors(css_features, slic_segments)

    # Min and Max Nomalized the features
    ceh_features_normalized = ceh_features / (np.sum(np.abs(ceh_features), axis= 1, keepdims= True) + 1e-10)
    css_min = expanded_css_features.min(axis = 0, keepdims = True)
    css_max = expanded_css_features.max(axis = 0, keepdims = True)
    css_features_normalized = (expanded_css_features - css_min) / (css_max - css_min + 1e-10)
    #### DISTANCE BETWEEN SP(j) AND THE CENTER OF THE DISC ####
    total_dist = []
    for (segID, segVal) in enumerate(np.unique(slic_segments)):
        total_dist.append(calculate_distance(slic_segments, segVal, 
                                             image_center_y=orig_height // 2, 
                                             image_center_x=orig_width // 2, 
                                             orig_height=orig_height, 
                                             orig_width=orig_width))
    # Convert the list to np array the shape should be (n_segments, 1)
    total_dist = np.array(total_dist).reshape(-1, 1)

    # Combine the 2 features together
    normalized_features = np.concatenate((ceh_features_normalized, css_features_normalized), axis = 1)
    # Combine the 3 features together the shape should be (n_segments, 1145) 1145 = 1024 (ceh_features_normalized) + 120 (css_features_normalized) + 1 (total_dist)
    total_features = np.concatenate((normalized_features, total_dist), axis = 1)
    return total_features, slic_segments

def calculate_distance(segments: np.ndarray, seg_id: int, image_center_y: int, image_center_x: int,
                       orig_height: int, orig_width: int) -> float:
    # Based on the papers, get the distance D(j) between the center of superpixel and the center of the disc as location information
    mask = (segments == seg_id)
    # Find the center of the current superpixel
    coord_id = np.argwhere(mask)
    center_y, center_x = coord_id.mean(axis=0).astype(int) 
    
    # Normalized distance x
    x_dist_normalized = float(((image_center_x - center_x) / orig_height) ** 2)
    # Normalized distance y
    y_dist_normalized = float(((image_center_y - center_y) / orig_width) ** 2)

    return np.sqrt(x_dist_normalized + y_dist_normalized)
    

def find_neighboring_pixels(segments: np.ndarray, seg_id: int) -> Dict[str, int]:
    # Based on the paper, find the 4 neighboring pixels for each superpixel (left, right, up, down)
    neighboring_pixels = {}
    mask = (segments == seg_id)

    # Find the center of the current superpixel
    coord_id = np.argwhere(mask)
    center_y, center_x = coord_id.mean(axis=0).astype(int)
    height, width = segments.shape

    # Left neightbor pixel
    left_pixel = seg_id
    for x in range(center_x - 1, -1, -1):
        if segments[center_y, x] != seg_id:
            left_pixel = segments[center_y, x]
            break
    neighboring_pixels['left'] = left_pixel

    # Right neightbor pixel
    right_pixel = seg_id
    for x in range(center_x + 1, width):
        if segments[center_y, x] != seg_id:
            right_pixel = segments[center_y, x]
            break
    neighboring_pixels['right'] = right_pixel

    # Up neightbor pixel
    up_pixel = seg_id
    for y in range(center_y - 1, -1, -1):
        if segments[y, center_x] != seg_id:
            up_pixel = segments[y, center_x]
            break
    neighboring_pixels['up'] = up_pixel

    # Down neightbor pixel
    down_pixel = seg_id
    for y in range(center_y + 1, height):
        if segments[y, center_x] != seg_id:
            down_pixel = segments[y, center_x]
            break
    neighboring_pixels['down'] = down_pixel
    return neighboring_pixels

def expand_css_neighbors(css_features: np.ndarray, slic_segments: np.ndarray) -> np.ndarray:
    n_segments = css_features.shape[0]
    expanded_css_features = []
    unique_segments = np.unique(slic_segments)
    seg_id_to_idx = {seg_id: idx for idx, seg_id in enumerate(unique_segments)}

    for segID, segVal in enumerate(unique_segments):
        neighbors = find_neighboring_pixels(slic_segments, segVal)
        # Get the current css feature
        curr_css_feature = css_features[segID]
        # Get the css features for each neighbor (left, right, up, down)
        neighbor_orders = ["left", "right", "up", "down"]
        neighbor_features = []
        for direction in neighbor_orders:
            neighbor_dir = neighbors[direction]
            neighbor_idx = seg_id_to_idx[neighbor_dir]
            neighbor_features.append(css_features[neighbor_idx])
        expanded_css = np.concatenate([curr_css_feature] + neighbor_features)
        expanded_css_features.append(expanded_css)
    # Convert to np array
    expanded_css_features = np.array(expanded_css_features)
    return expanded_css_features    

def extract_map_features(image: np.ndarray, combinations: List[Tuple], org_h: int, org_w: int) -> np.ndarray:
    # In the paper will only use the green and blue channel of the image 
    # Output will be a list of np.array with shape [6, img_height, img_width]
    final_map_features = []
    dgp_features = dyalic_gaussian_pyramid(image)
    for idx, combination in enumerate(combinations):
        # Extract the 2 features map from the dyalic gaussian pyramid
        finer_map = dgp_features[combination[0]]
        coarser_map = dgp_features[combination[1]]
        # Get the centre surround difference
        centre_surr_diff = conbine_dgp_map([finer_map, coarser_map], orig_h = org_h, orig_w= org_w)
        final_map_features.append(centre_surr_diff)
    # Export this to a numpy array
    final_map_features = np.array(final_map_features)
    return final_map_features

def conbine_dgp_map(feature_maps: List[np.ndarray], orig_h: int, orig_w: int):
    # Features map should only contain the 2 samples [finer samples, coarser samples]
    finer_sample = feature_maps[0]
    coarser_sample = feature_maps[1]
    # Get the centre (finer) height and width 
    finer_height = finer_sample.shape[0]
    finer_width = finer_sample.shape[1]

    # Interpolate I(s)to be the same size as I(c)
    coarser_sample_resized = cv2.resize(coarser_sample.astype(float), (finer_width, finer_height),
                                        interpolation= cv2.INTER_CUBIC,
                                        )
    # Compute center-surround difference: finer - interpolated coarser
    centre_surr_diff = finer_sample.astype(float) - coarser_sample_resized
    # Resize the center surround difference to the original image size
    centre_surr_diff = cv2.resize(centre_surr_diff, (orig_w, orig_h), interpolation= cv2.INTER_CUBIC)
    return centre_surr_diff


if __name__ == "__main__":
    # # image_path = "cropped_img/od_cropped/Glaucoma/CVO_Glaucoma_076.jpg.png"
    # # asfas =  feature_extraction(image_path)
    # # print(asfas.shape)
    # image_path = "cropped_img/od_cropped_for_oc/image_55.jpg"
    # mask_path = "cropped_img/mask_od_cropped_for_oc/image_55.png"
    # image = cv2.imread(image_path)
    # result = image.copy()
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # colors = {
    # 0: (0, 0, 0),       # Black for class 0
    # 1: (0, 255, 0),     # Green for class 1
    # 2: (255, 0, 0),     # Blue for class 2
    # }

    # # Draw contours for each class
    # for class_id in [1, 2]:  # Skip background (0)
    #     # Create binary mask for this class
    #     class_mask = (mask == class_id).astype(np.uint8)
        
    #     # Find contours
    #     contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     # Draw contours
    #     cv2.drawContours(result, contours, -1, colors[class_id], 2)

    # plt.imshow(result)
    # plt.tight_layout()

    # plt.savefig("test.png")
    print("You dummy")