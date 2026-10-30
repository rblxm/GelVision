import cv2
import numpy as np

class VideoStreamer:
    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or np.isnan(self.fps):
            self.fps = 30.0  # Fallback
            
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps

    def get_frame_at_sec(self, seconds):
        """Seek to a specific time and return the RGB frame."""
        frame_idx = int(seconds * self.fps)
        if frame_idx >= self.frame_count:
            frame_idx = self.frame_count - 1
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def coarse_scan(self, sample_rate=1.0, callback=None):
        """
        Scans the video at a given sample rate (frames per second of video).
        Yields (timestamp_sec, frame_rgb).
        If callback is provided, it's called with progress (0.0 to 1.0).
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        step_frames = max(1, int(self.fps / sample_rate))
        current_frame = 0
        
        while current_frame < self.frame_count:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = self.cap.read()
            if not ret:
                break
                
            timestamp = current_frame / self.fps
            yield timestamp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if callback:
                callback(current_frame / self.frame_count)
                
            current_frame += step_frames

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
