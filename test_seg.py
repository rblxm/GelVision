import cv2
import numpy as np
from gelvision.core.dish_detector import detect_dish
from gelvision.core.segmenter import segment_fragments

img = cv2.imread("../AutoAnalysis/51580_365_24_S1.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

center, radius, scale = detect_dish(img_rgb)
fragments, mask = segment_fragments(img_rgb, center, radius, min_area_px=20)

# Create an overlay
overlay = img.copy()
# draw mask in green
overlay[mask > 0] = [0, 255, 0]

cv2.imwrite("overlay.png", cv2.addWeighted(img, 0.7, overlay, 0.3, 0))
