import streamlit as st
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
import os
from gelvision.core.dish_detector import detect_dish
from gelvision.core.segmenter import segment_fragments
from gelvision.core.kinetic_detector import detect_rupture_events
from gelvision.core.video_streamer import VideoStreamer

st.set_page_config(page_title="GelVision Rupture Analysis", layout="wide")

st.title("🧪 GELVISION | Szczepanski Polymer Lab")
st.markdown("Automated hydrogel rupture and fragment analysis.")

# Sidebar controls
st.sidebar.header("⚙️ Controls & Parameters")
assay_type = st.sidebar.selectbox("Assay Module", ["Rupture (Image)", "Rupture (Video Kinetics)"])
min_area = st.sidebar.slider("Min Fragment Size (px)", 5, 200, 20)
chrominance_offset = st.sidebar.slider("Chrominance Sensitivity Offset", 0.0, 15.0, 6.0, 0.5)

if assay_type == "Rupture (Image)":
    st.header("📂 Step 1: Load Experiment Image")
    uploaded_file = st.file_uploader("Upload a final frame (.png, .tif, .jpg)", type=["png", "tif", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        st.header("🔬 Step 2: Segmentation & Verification")
        
        with st.spinner("Detecting dish and calibrating scale..."):
            center, radius, scale = detect_dish(image_rgb)
            
        st.info(f"Detected Pyrex Dish Center: {center}, Radius: {radius} px. Calibration: {scale:.2f} px/mm.")
        
        with st.spinner("Segmenting hydrogel fragments..."):
            def segment_fragments_adj(image_rgb, center, radius, min_area_px, offset):
                h, w = image_rgb.shape[:2]
                Y, X = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
                valid_mask = dist_from_center < (radius * 0.96)
                
                image_float = image_rgb.astype(np.float32)
                b_minus_r = image_float[:, :, 2] - image_float[:, :, 0]
                bg_median = np.median(b_minus_r[valid_mask])
                thresh = bg_median - offset
                
                fg_mask = (b_minus_r < thresh) & valid_mask
                fg_uint8 = fg_mask.astype(np.uint8) * 255
                
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                closed = cv2.morphologyEx(fg_uint8, cv2.MORPH_CLOSE, kernel_close)
                
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
                
                fragments = []
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= min_area_px:
                        comp_mask = (labels == i).astype(np.uint8)
                        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        if len(contours) > 0:
                            cnt = contours[0]
                            perimeter = cv2.arcLength(cnt, True)
                            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
                            if circularity > 0.92 and area < 200:
                                continue
                            fragments.append({
                                'id': len(fragments) + 1,
                                'area_px2': area,
                                'area_mm2': area / (scale**2),
                                'centroid': centroids[i],
                                'circularity': circularity,
                                'contour': cnt
                            })
                return fragments, opened

            fragments, mask = segment_fragments_adj(image_rgb, center, radius, min_area, chrominance_offset)
            
        # Draw overlay
        overlay = image_rgb.copy()
        cv2.circle(overlay, center, radius, (255, 0, 0), 5) # Draw dish
        for f in fragments:
            cv2.drawContours(overlay, [f['contour']], -1, (0, 255, 0), 2)
            cx, cy = int(f['centroid'][0]), int(f['centroid'][1])
            cv2.putText(overlay, f"#{f['id']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
            
        blended = cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_rgb, caption="Raw Frame", use_container_width=True)
        with col2:
            st.image(blended, caption="Detected Fragments", use_container_width=True)
            
        st.header("📊 Step 3: Results & Export")
        total_area_px = sum(f['area_px2'] for f in fragments)
        total_area_mm2 = total_area_px / (scale**2)
        st.metric("Total Fragments", len(fragments))
        st.metric("Total Area (mm²)", f"{total_area_mm2:.2f}")
        
        if fragments:
            df = pd.DataFrame(fragments)
            df.drop(columns=['centroid', 'contour'], inplace=True)
            st.dataframe(df)
            
            # Excel export
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Fragment Data')
            
            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
                file_name="rupture_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

elif assay_type == "Rupture (Video Kinetics)":
    st.header("📂 Step 1: Load Video")
    video_path = st.text_input("Enter absolute path to .mp4/.mov video file (e.g. C:/data/exp1.mp4):")
    
    if video_path and os.path.exists(video_path):
        st.success(f"Loaded: {os.path.basename(video_path)}")
        
        if st.button("Run Kinetic Analysis"):
            with st.spinner("Extracting first frame for calibration..."):
                with VideoStreamer(video_path) as streamer:
                    first_frame = streamer.get_frame_at_sec(0)
                center, radius, scale = detect_dish(first_frame)
                st.info(f"Detected Pyrex Dish Center: {center}, Radius: {radius} px. Calibration: {scale:.2f} px/mm.")
            
            with st.spinner("Scanning video for rupture events (this may take a minute)..."):
                t_first, event_times, t_arr, diff_arr = detect_rupture_events(video_path, center, radius, sample_rate=1.0)
            
            if t_first is not None:
                st.success(f"Detection complete. Found {len(event_times)} events.")
                col1, col2 = st.columns(2)
                col1.metric("Time to First Rupture (s)", f"{t_first:.1f}")
                col2.metric("Total Events", len(event_times))
                
                # Plot
                import plotly.express as px
                import pandas as pd
                df_plot = pd.DataFrame({"Time (s)": t_arr, "Delta": diff_arr})
                fig = px.line(df_plot, x="Time (s)", y="Delta", title="Frame Difference (Rupture Kinetics)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show event frames
                st.header("Event Highlights")
                with VideoStreamer(video_path) as streamer:
                    for i, et in enumerate(event_times):
                        # Show frame 1 sec after event
                        frame = streamer.get_frame_at_sec(et + 1.0)
                        if frame is not None:
                            # segment it
                            fragments, _ = segment_fragments(frame, center, radius, min_area_px=min_area)
                            overlay = frame.copy()
                            for f in fragments:
                                cv2.drawContours(overlay, [f['contour']], -1, (0, 255, 0), 2)
                                cx, cy = int(f['centroid'][0]), int(f['centroid'][1])
                                cv2.putText(overlay, f"#{f['id']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                            
                            st.image(overlay, caption=f"Event {i+1} at {et:.1f} seconds ({len(fragments)} fragments detected)", use_container_width=True)
            else:
                st.warning("No rupture events detected above the noise threshold.")
    elif video_path:
        st.error("File not found. Please check the path.")
