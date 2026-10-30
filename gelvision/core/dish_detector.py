import cv2
import numpy as np

def detect_dish(image_rgb):
    """
    Detects the 90mm Pyrex dish in the image.
    Returns (center_x, center_y), radius_px, and px_per_mm scale.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Hough Circles is very robust for petri dishes
    # We expect a large circle. For a 4K image, dish is ~2000-3000px.
    # Height is 2160, so max radius is ~1080.
    min_dist = gray.shape[0] // 2
    
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=min_dist,
        param1=50, 
        param2=30, 
        minRadius=gray.shape[0] // 4, 
        maxRadius=int(gray.shape[0] * 0.9)
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Assuming the most prominent circle is our dish
        best_circle = circles[0]
        cx, cy, r = best_circle
        
        # 90 mm diameter -> 45 mm radius
        px_per_mm = (r * 2) / 90.0
        
        return (cx, cy), r, px_per_mm
    
    # Fallback if detection fails (assume centered and ~80% of height)
    h, w = gray.shape
    r = int(h * 0.45)
    cx, cy = w // 2, h // 2
    px_per_mm = (r * 2) / 90.0
    return (cx, cy), r, px_per_mm
