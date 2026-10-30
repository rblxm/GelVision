import cv2
import numpy as np
from gelvision.core.video_streamer import VideoStreamer

def detect_rupture_events(video_path, center, radius, sample_rate=1.0, threshold_sigma=4.0):
    """
    Scans the video and returns timestamp of first rupture and a list of all rupture events.
    Returns: t_first (float), event_timestamps (list), time_array, diff_array
    """
    time_array = []
    diff_array = []
    
    # Mask to isolate dish
    valid_mask = None
    
    prev_frame_gray = None
    
    with VideoStreamer(video_path) as streamer:
        for t, frame_rgb in streamer.coarse_scan(sample_rate=sample_rate):
            if valid_mask is None:
                h, w = frame_rgb.shape[:2]
                Y, X = np.ogrid[:h, :w]
                dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
                valid_mask = (dist < (radius * 0.90)).astype(np.uint8)
                
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            # Mask out background
            gray_masked = gray * valid_mask
            
            if prev_frame_gray is not None:
                # Compute absolute difference
                diff = cv2.absdiff(gray_masked, prev_frame_gray)
                # Sum of absolute differences in the dish
                delta = np.sum(diff) / 1000.0  # scale down
                time_array.append(t)
                diff_array.append(delta)
                
            prev_frame_gray = gray_masked

    if not diff_array:
        return None, [], [], []

    # Detect peaks (events)
    # Background drift
    diff_np = np.array(diff_array)
    bg_mean = np.median(diff_np)
    bg_std = np.std(diff_np)
    
    threshold = bg_mean + (threshold_sigma * bg_std)
    # Ensure a minimum absolute threshold to ignore pure noise
    threshold = max(threshold, 50.0) 
    
    event_timestamps = []
    in_event = False
    
    for i, delta in enumerate(diff_array):
        if delta > threshold:
            if not in_event:
                event_timestamps.append(time_array[i])
                in_event = True
        else:
            in_event = False
            
    t_first = event_timestamps[0] if event_timestamps else None
    
    return t_first, event_timestamps, time_array, diff_array
