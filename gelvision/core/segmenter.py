import cv2
import numpy as np
from scipy import ndimage

def segment_fragments(image_rgb, center, radius, min_area_px=20):
    """
    Segments hydrogel fragments from a single RGB frame.
    Uses Blue-Red chrominance difference since gel is tan/yellow (absorbs blue)
    while background glass/water is blue-gray.
    """
    h, w = image_rgb.shape[:2]
    
    # 1. Create spatial masks
    # Central basin: 85% of dish radius
    # Annular edge: 85% to 96% of dish radius
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # We ignore everything outside 96% of radius (glass rim, meniscus, etc.)
    valid_mask = dist_from_center < (radius * 0.96)
    
    # 2. Chrominance calculation (Blue - Red)
    image_float = image_rgb.astype(np.float32)
    b_minus_r = image_float[:, :, 2] - image_float[:, :, 0]
    
    # Normal background B-R is ~29. Gel B-R is ~18.
    # So gel is where B-R is significantly lower.
    # Let's compute local mean of B-R in the valid dish area to dynamically threshold.
    bg_median = np.median(b_minus_r[valid_mask])
    
    # Fragments are areas where B-R is lower than background by at least some threshold
    # E.g., threshold = bg_median - 8
    # If bg is 29, thresh is 21. Gel is 18, so it will be flagged.
    thresh = bg_median - 6.0
    
    # Initial binary mask
    fg_mask = (b_minus_r < thresh) & valid_mask
    
    # Convert to uint8 for morphological operations
    fg_uint8 = fg_mask.astype(np.uint8) * 255
    
    # 3. Morphological cleaning
    # Close small holes (like internal cracks or noise)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(fg_uint8, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove single-pixel noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # 4. Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    
    fragments = []
    # stats format: [x, y, w, h, area]
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter by minimum area (e.g. 20 px)
        if area >= min_area_px:
            # Check circularity to reject obvious bubbles
            # Extract single component mask to get perimeter
            comp_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                cnt = contours[0]
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    # Bubbles are perfectly circular (>0.9). Splinters are not.
                    if circularity > 0.92 and area < 200:
                        continue # reject micro-bubbles
                
                fragments.append({
                    'id': len(fragments) + 1,
                    'area_px': area,
                    'centroid': centroids[i],
                    'bbox': stats[i, :4],
                    'contour': cnt,
                    'circularity': circularity if perimeter > 0 else 0
                })
                
    return fragments, opened
