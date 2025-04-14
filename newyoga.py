import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import time
from PIL import Image, ImageDraw, ImageFont
import math
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading
import queue

st.set_page_config(
    page_title="Asana AI",
    page_icon="🧘",
    layout="wide"
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_holistic = mp.solutions.holistic

YOGA_POSES = {
    "Warrior II": {
        "description": "Arms extended parallel to the ground, front knee bent at 90°, back leg straight",
        "key_points": ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist", 
                      "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"],
        "angles": [
            {"name": "Front knee bend", "points": ["left_hip", "left_knee", "left_ankle"], "target": 90, "tolerance": 15},
            {"name": "Arms alignment", "points": ["left_wrist", "left_shoulder", "right_wrist"], "target": 180, "tolerance": 20},
            {"name": "Back leg straightness", "points": ["right_hip", "right_knee", "right_ankle"], "target": 170, "tolerance": 10}
        ],
        "difficulty": "Intermediate",
        "benefits": ["Strengthens legs and core", "Opens hips and chest", "Improves balance and focus"],
        "common_mistakes": ["Front knee extending past ankle", "Shoulders hunched", "Back heel lifting"]
    },
    "Tree Pose": {
        "description": "Stand on one leg with the other foot placed on inner thigh, hands in prayer position",
        "key_points": ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", 
                      "left_shoulder", "right_shoulder"],
        "angles": [
            {"name": "Standing leg alignment", "points": ["left_hip", "left_knee", "left_ankle"], "target": 170, "tolerance": 10},
            {"name": "Raised knee position", "points": ["right_hip", "right_knee", "right_ankle"], "target": 45, "tolerance": 15},
            {"name": "Torso alignment", "points": ["left_shoulder", "left_hip", "left_ankle"], "target": 180, "tolerance": 10}
        ],
        "difficulty": "Beginner",
        "benefits": ["Improves balance and focus", "Strengthens legs and core", "Opens hips"],
        "common_mistakes": ["Leaning too far to one side", "Placing foot on knee joint", "Looking down"]
    },
    "Downward Dog": {
        "description": "Hands and feet on the ground, forming an inverted V shape with the body",
        "key_points": ["left_shoulder", "left_wrist", "left_hip", "left_knee", "left_ankle", 
                      "right_shoulder", "right_wrist", "right_hip", "right_knee", "right_ankle"],
        "angles": [
            {"name": "Hip angle", "points": ["left_shoulder", "left_hip", "left_ankle"], "target": 60, "tolerance": 15},
            {"name": "Arm-shoulder alignment", "points": ["left_wrist", "left_elbow", "left_shoulder"], "target": 180, "tolerance": 15},
            {"name": "Leg straightness", "points": ["left_hip", "left_knee", "left_ankle"], "target": 160, "tolerance": 20}
        ],
        "difficulty": "Beginner",
        "benefits": ["Stretches hamstrings and calves", "Strengthens arms and shoulders", "Energizes the body"],
        "common_mistakes": ["Rounding the back", "Heels lifted too high", "Hands too close together"]
    },
    "Chair Pose": {
        "description": "Knees bent as if sitting in an invisible chair, arms raised overhead",
        "key_points": ["left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle"],
        "angles": [
            {"name": "Knee bend", "points": ["left_hip", "left_knee", "left_ankle"], "target": 120, "tolerance": 15},
            {"name": "Arm raise", "points": ["left_elbow", "left_shoulder", "left_hip"], "target": 170, "tolerance": 15},
            {"name": "Torso angle", "points": ["left_shoulder", "left_hip", "left_knee"], "target": 145, "tolerance": 15}
        ],
        "difficulty": "Beginner",
        "benefits": ["Strengthens thighs and ankles", "Stretches shoulders and chest", "Stimulates heart and abdominal organs"],
        "common_mistakes": ["Knees extending past toes", "Shoulders hunched", "Leaning forward too much"]
    },
    "Triangle Pose": {
        "description": "Legs wide apart, torso bent to the side with one arm extended upward",
        "key_points": ["left_shoulder", "left_hip", "left_knee", "left_ankle", "right_shoulder", "right_hip", "right_knee", "right_ankle"],
        "angles": [
            {"name": "Front leg straightness", "points": ["left_hip", "left_knee", "left_ankle"], "target": 170, "tolerance": 10},
            {"name": "Torso bend", "points": ["left_shoulder", "left_hip", "left_ankle"], "target": 60, "tolerance": 15},
            {"name": "Top arm alignment", "points": ["right_shoulder", "right_elbow", "right_wrist"], "target": 180, "tolerance": 10}
        ],
        "difficulty": "Intermediate",
        "benefits": ["Stretches legs, hips and spine", "Strengthens thighs and ankles", "Improves digestion"],
        "common_mistakes": ["Leaning forward or backward", "Collapsing the chest", "Bending the front knee"]
    },
    "Cobra Pose": {
        "description": "Lying face down, upper body lifted with arms straight, lower body on the ground",
        "key_points": ["left_shoulder", "left_elbow", "left_wrist", "left_hip", "right_shoulder", "right_elbow", "right_wrist"],
        "angles": [
            {"name": "Arm extension", "points": ["left_wrist", "left_elbow", "left_shoulder"], "target": 160, "tolerance": 20},
            {"name": "Upper body lift", "points": ["left_wrist", "left_shoulder", "left_hip"], "target": 45, "tolerance": 15},
            {"name": "Shoulder alignment", "points": ["left_wrist", "left_shoulder", "right_shoulder"], "target": 180, "tolerance": 15}
        ],
        "difficulty": "Beginner",
        "benefits": ["Strengthens spine", "Opens chest and lungs", "Improves posture"],
        "common_mistakes": ["Shoulders hunched", "Pressing into hands too much", "Straining neck"]
    },
    "Warrior I": {
        "description": "Lunge position with back foot at 45°, arms extended overhead",
        "key_points": ["left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle", 
                      "right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle"],
        "angles": [
            {"name": "Front knee bend", "points": ["left_hip", "left_knee", "left_ankle"], "target": 90, "tolerance": 15},
            {"name": "Arms extension", "points": ["left_elbow", "left_shoulder", "left_hip"], "target": 180, "tolerance": 15},
            {"name": "Back leg angle", "points": ["right_hip", "right_knee", "right_ankle"], "target": 150, "tolerance": 15}
        ],
        "difficulty": "Intermediate",
        "benefits": ["Strengthens legs, arms and shoulders", "Opens hips, chest and lungs", "Improves focus and balance"],
        "common_mistakes": ["Front knee extending past ankle", "Arching back", "Shoulders tensed"]
    }
}

POSE_IMAGES = {
    "Warrior II": "ypic\w.png",
    "Tree Pose": "ypic\k.png",
    "Downward Dog": "ypic\d.png",
    "Chair Pose": "ypic\c.png",
    "Triangle Pose": "ypic\m.png",
    "Cobra Pose": "ypic\cb.png",
    "Warrior I": "ypic\w1.png"
}

YOGA_SEQUENCES = {
    "Morning Energizer": ["Downward Dog", "Cobra Pose", "Warrior I", "Warrior II"],
    "Balance Focus": ["Tree Pose", "Warrior II", "Triangle Pose"],
    "Core Strength": ["Chair Pose", "Downward Dog", "Cobra Pose"],
    "Beginner Flow": ["Tree Pose", "Chair Pose", "Downward Dog"],
    "Full Body Workout": ["Warrior I", "Warrior II", "Triangle Pose", "Chair Pose", "Cobra Pose"]
}

LANDMARK_MAPPING = {
    "nose": 0,
    "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8,
    "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32
}


SAVE_DIR = "yoga_analyses"
os.makedirs(SAVE_DIR, exist_ok=True)

def calculate_angle(a, b, c):
    """Calculate the angle between three points (in degrees)"""
    if any(point is None for point in [a, b, c]):
        return None
    
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

# Function to extract landmark coordinates
def get_landmark_coords(landmarks, landmark_name):
    """Get x, y coordinates for a specific landmark"""
    if landmarks is None:
        return None
    
    idx = LANDMARK_MAPPING.get(landmark_name)
    if idx is None or not landmarks.landmark[idx].visibility > 0.5:
        return None
    
    return (landmarks.landmark[idx].x, landmarks.landmark[idx].y)

# Function to analyze pose and provide feedback
def analyze_pose(landmarks, selected_pose):
    """Analyze detected pose against the selected yoga pose"""
    if landmarks is None or selected_pose not in YOGA_POSES:
        return [], 0, {}
    
    pose_info = YOGA_POSES[selected_pose]
    feedback = []
    score = 0
    total_checks = 0
    detailed_angles = {}
    
    # Check angles
    for angle_info in pose_info["angles"]:
        points = angle_info["points"]
        coords = [get_landmark_coords(landmarks, point) for point in points]
        
        if None in coords:
            feedback.append(f"Cannot detect {' and '.join(points)} for {angle_info['name']} measurement")
            continue
            
        angle = calculate_angle(*coords)
        target = angle_info["target"]
        tolerance = angle_info["tolerance"]
        
        # Store the measured angle for detailed analytics
        detailed_angles[angle_info["name"]] = {
            "measured": angle,
            "target": target,
            "tolerance": tolerance,
            "is_correct": abs(angle - target) <= tolerance
        }
        
        total_checks += 1
        if abs(angle - target) <= tolerance:
            feedback.append(f"✅ {angle_info['name']}: Good alignment ({angle:.1f}°)")
            score += 1
        else:
            direction = "increase" if angle < target else "decrease"
            feedback.append(f"❌ {angle_info['name']}: Adjust angle from {angle:.1f}° to {target}±{tolerance}° ({direction})")
    
    # Add common mistakes feedback based on angles
    for mistake in pose_info.get("common_mistakes", []):
        if any("❌" in fb for fb in feedback):
            feedback.append(f"⚠️ Watch out for: {mistake}")
            break
    
    # Calculate final score as percentage
    percentage_score = (score / total_checks * 100) if total_checks > 0 else 0
    
    return feedback, percentage_score, detailed_angles

# Function to draw angle measurements on image
def draw_angle_visualization(image, landmarks, selected_pose):
    """Draw angle measurements for key points on the image"""
    if landmarks is None or selected_pose not in YOGA_POSES:
        return image
    
    pose_info = YOGA_POSES[selected_pose]
    img_h, img_w, _ = image.shape
    
    # Draw angles on image
    for angle_info in pose_info["angles"]:
        points = angle_info["points"]
        coords = []
        
        # Get pixel coordinates for the three points
        for point in points:
            idx = LANDMARK_MAPPING.get(point)
            if idx is None or landmarks.landmark[idx].visibility < 0.5:
                coords = None
                break
            
            x = int(landmarks.landmark[idx].x * img_w)
            y = int(landmarks.landmark[idx].y * img_h)
            coords.append((x, y))
        
        if coords is None:
            continue
        
        
        angle = calculate_angle(
            (coords[0][0]/img_w, coords[0][1]/img_h),
            (coords[1][0]/img_w, coords[1][1]/img_h),
            (coords[2][0]/img_w, coords[2][1]/img_h)
        )
        
       
        cv2.line(image, coords[0], coords[1], (0, 255, 0), 2)
        cv2.line(image, coords[1], coords[2], (0, 255, 0), 2)
        
        cv2.ellipse(image, coords[1], (20, 20), 
                    0, 
                    math.degrees(math.atan2(coords[0][1]-coords[1][1], coords[0][0]-coords[1][0])),
                    math.degrees(math.atan2(coords[2][1]-coords[1][1], coords[2][0]-coords[1][0])),
                    (0, 255, 0), 2)
        
       
        text_x = coords[1][0] + 20
        text_y = coords[1][1] - 20
        cv2.putText(image, f"{angle:.1f}°", (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

# Function to draw pose feedback on image
def draw_feedback_on_image(image, landmarks, selected_pose, feedback_text, score, show_angles=True):
    """Draw skeleton, angles, and feedback on the image"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    draw.rectangle([(10, 10), (300, 50)], fill=(0, 0, 0, 180))
    draw.text((20, 15), f"{selected_pose}: {score:.1f}%", font=font, fill=(255, 255, 255))
    
    y_offset = 60
    for text in feedback_text:
        if "✅" in text:
            text_color = (50, 205, 50)  
        elif "❌" in text:
            text_color = (255, 99, 71)  
        elif "⚠️" in text:
            text_color = (255, 191, 0) 
        else:
            text_color = (255, 255, 255)  
            
        draw.rectangle([(10, y_offset), (400, y_offset + 20)], fill=(0, 0, 0, 150))
        draw.text((20, y_offset), text, font=small_font, fill=text_color)
        y_offset += 25
    
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Draw skeleton
    if landmarks is not None:
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Draw angle visualizations if requested
        if show_angles:
            image = draw_angle_visualization(image, landmarks, selected_pose)
    
    return image

# Function to process video frames (for both webcam and uploaded videos)
def process_frame(frame, pose, segmentation, selected_pose, 
                 detection_confidence, tracking_confidence, visualization_mode,
                 show_angles=True, bg_color=None, breathing_guide=False):
    """Process a single video frame for pose detection and visualization"""
    height, width, _ = frame.shape
    max_dim = 640
    if height > max_dim or width > max_dim:
        scale = max_dim / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    pose_results = pose.process(frame_rgb)
    segmentation_results = segmentation.process(frame_rgb)
    frame_rgb.flags.writeable = True
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
    output_frame = frame.copy()
    pose_detected = pose_results.pose_landmarks is not None
    if segmentation_results.segmentation_mask is not None:
        segmentation_mask = segmentation_results.segmentation_mask > 0.1
        kernel = np.ones((5, 5), np.uint8)
        segmentation_mask = cv2.morphologyEx(
            segmentation_mask.astype(np.uint8) * 255, 
            cv2.MORPH_CLOSE, 
            kernel
        )
        segmentation_mask = cv2.morphologyEx(
            segmentation_mask,
            cv2.MORPH_OPEN,
            kernel
        )
        segmentation_mask_3channel = cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2BGR)
        segmentation_mask_blurred = cv2.GaussianBlur(segmentation_mask_3channel, (15, 15), 0)
        person_mask = segmentation_mask_blurred / 255.0
        background_mask = 1.0 - person_mask

        if visualization_mode == "enhanced":
            blurred_bg = cv2.GaussianBlur(frame, (25, 25), 0)
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (center_x, center_y), min(h, w) // 2, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (h//4*2+1, w//4*2+1), 0)
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
            blurred_bg = blurred_bg * 0.7  
            blurred_bg[:,:,0] = np.clip(blurred_bg[:,:,0] * 1.2, 0, 255)
            blurred_bg = blurred_bg * mask_3channel
            output_frame = frame * person_mask + blurred_bg * background_mask
            
        elif visualization_mode == "blur":
            blurred_bg = cv2.GaussianBlur(frame, (55, 55), 0)
            output_frame = frame * person_mask + blurred_bg * background_mask
            
        elif visualization_mode == "color":
            if bg_color:
                bg_color = bg_color.lstrip('#')
                r, g, b = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
                bg_color_bgr = (b, g, r)  
                color_bg = np.ones_like(frame) * np.array(bg_color_bgr)
            else:
                color_bg = np.ones_like(frame) * np.array([120, 60, 120])
           
            h, w = frame.shape[:2]
            gradient = np.linspace(0.8, 1.2, h).reshape(-1, 1)
            color_bg = color_bg * gradient.reshape(h, 1, 1)
            color_bg = np.clip(color_bg, 0, 255)
            output_frame = frame * person_mask + color_bg * background_mask
            
        elif visualization_mode == "remove":
            black_bg = np.zeros_like(frame)
            output_frame = frame * person_mask + black_bg * background_mask
            
        elif visualization_mode == "skeleton":
            canvas = np.zeros_like(frame)
            if pose_detected:
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    if (pose_results.pose_landmarks.landmark[start_idx].visibility > 0.5 and
                        pose_results.pose_landmarks.landmark[end_idx].visibility > 0.5):
                        
                        start_point = (
                            int(pose_results.pose_landmarks.landmark[start_idx].x * width),
                            int(pose_results.pose_landmarks.landmark[start_idx].y * height)
                        )
                        
                        end_point = (
                            int(pose_results.pose_landmarks.landmark[end_idx].x * width),
                            int(pose_results.pose_landmarks.landmark[end_idx].y * height)
                        )
                        
                        if start_idx in range(11, 17) or end_idx in range(11, 17):  # Arms
                            color = (245, 117, 66)  # Orange
                        elif start_idx in range(23, 31) or end_idx in range(23, 31):  # Legs
                            color = (66, 245, 200)  # Teal
                        else:  # Torso
                            color = (245, 66, 230)  # Pink
                        
                        cv2.line(canvas, start_point, end_point, color, 4)
                for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    if landmark.visibility > 0.5:
                        cx = int(landmark.x * width)
                        cy = int(landmark.y * height)
                        
                        cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), -1)
                        
                        if idx in range(11, 17):  
                            cv2.circle(canvas, (cx, cy), 6, (245, 117, 66), -1)
                        elif idx in range(23, 31):  
                            cv2.circle(canvas, (cx, cy), 6, (66, 245, 200), -1)
                        else:  
                            cv2.circle(canvas, (cx, cy), 6, (245, 66, 230), -1)
                
               
                if show_angles:
                    draw_angle_visualization_on_canvas(canvas, pose_results.pose_landmarks, selected_pose, frame.shape)
            
            output_frame = canvas
        else: 
            output_frame = frame.copy()
    else:
        output_frame = frame.copy()

    output_frame = output_frame.astype(np.uint8)
    
    
    feedback_text, score, detailed_angles = [], 0, {}
    if pose_detected:
        feedback_text, score, detailed_angles = analyze_pose(pose_results.pose_landmarks, selected_pose)
        if visualization_mode != "skeleton":
            output_frame = draw_feedback_on_image(
                output_frame, 
                pose_results.pose_landmarks, 
                selected_pose, 
                feedback_text, 
                score, 
                show_angles
            )
        else:
            # For skeleton mode, draw minimal feedback
            y_offset = 30
            cv2.putText(
                output_frame, 
                f"{selected_pose}: {score:.1f}%", 
                (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Add minimal feedback text
            for i, fb in enumerate(feedback_text[:3]):  
                if "✅" in fb:
                    color = (0, 255, 0) 
                elif "❌" in fb:
                    color = (0, 0, 255)  
                else:
                    color = (255, 255, 255)  
                
                y_offset += 25
                cv2.putText(
                    output_frame, 
                    fb[:50] + "..." if len(fb) > 50 else fb, 
                    (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    1
                )
        
        # Add breathing guide if enabled
        if breathing_guide:
            t = time.time()
            phase = math.sin(t * 0.5)  
            
           
            h, w, _ = output_frame.shape
            guide_x = int(w * 0.85)
            guide_y = int(h * 0.9)
            guide_radius = int(min(h, w) * 0.05)
            
          
            breathing_radius = int(guide_radius * (1 + 0.3 * phase))
            if phase > 0:
                cv2.circle(output_frame, (guide_x, guide_y), breathing_radius, (0, 255, 0), 2)
                cv2.putText(output_frame, "Inhale", (guide_x - 25, guide_y - breathing_radius - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.circle(output_frame, (guide_x, guide_y), breathing_radius, (255, 0, 0), 2)
                cv2.putText(output_frame, "Exhale", (guide_x - 25, guide_y - breathing_radius - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return output_frame, feedback_text, score, detailed_angles, pose_detected


def draw_angle_visualization_on_canvas(canvas, landmarks, selected_pose, shape):
    """Draw angle measurements on skeleton canvas"""
    if landmarks is None or selected_pose not in YOGA_POSES:
        return canvas
    
    img_h, img_w = shape[:2]
    pose_info = YOGA_POSES[selected_pose]
    
   
    for angle_info in pose_info["angles"]:
        points = angle_info["points"]
        coords = []
        
      
        for point in points:
            idx = LANDMARK_MAPPING.get(point)
            if idx is None or landmarks.landmark[idx].visibility < 0.5:
                coords = None
                break
            
            x = int(landmarks.landmark[idx].x * img_w)
            y = int(landmarks.landmark[idx].y * img_h)
            coords.append((x, y))
        
        if coords is None:
            continue
        
      
        angle = calculate_angle(
            (coords[0][0]/img_w, coords[0][1]/img_h),
            (coords[1][0]/img_w, coords[1][1]/img_h),
            (coords[2][0]/img_w, coords[2][1]/img_h)
        )
        
        
        target = angle_info["target"]
        tolerance = angle_info["tolerance"]
        
       
        if abs(angle - target) <= tolerance:
            color = (0, 255, 0)  
        else:
            color = (0, 0, 255) 
        
       
        cv2.line(canvas, coords[0], coords[1], color, 2)
        cv2.line(canvas, coords[1], coords[2], color, 2)
        
        cv2.ellipse(canvas, coords[1], (20, 20), 
                    0, 
                    math.degrees(math.atan2(coords[0][1]-coords[1][1], coords[0][0]-coords[1][0])),
                    math.degrees(math.atan2(coords[2][1]-coords[1][1], coords[2][0]-coords[1][0])),
                    color, 2)
        
        #
        text_x = coords[1][0] + 20
        text_y = coords[1][1] - 20
        cv2.putText(canvas, f"{angle:.1f}° ({target}°)", (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return canvas
# Function to generate a downloadable report
def generate_report(user_name, pose_data, detailed_angles, screenshot=None):
    """Generate a downloadable HTML report with pose analysis details"""
   
    buffer = BytesIO()
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Yoga Pose Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #9370DB; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .header {{ background-color: #f0e6ff; padding: 20px; border-radius: 10px; }}
            .pose-info {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 10px; }}
            .feedback {{ margin: 10px 0; }}
            .good {{ color: green; }}
            .improve {{ color: red; }}
            .warning {{ color: orange; }}
            .score {{ font-size: 24px; font-weight: bold; text-align: center; margin: 20px 0; }}
            .angles-chart {{ width: 100%; height: 300px; margin: 20px 0; }}
            .screenshot {{ max-width: 100%; margin: 20px 0; border-radius: 10px; }}
            .footer {{ margin-top: 30px; text-align: center; font-size: 12px; color: #888; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧘 Asana AI - Pose Analysis Report</h1>
                <p>User: {user_name}</p>
                <p>Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="pose-info">
                <h2>Pose: {pose_data['pose_name']}</h2>
                <p>{YOGA_POSES[pose_data['pose_name']]['description']}</p>
                <p><strong>Difficulty:</strong> {YOGA_POSES[pose_data['pose_name']]['difficulty']}</p>
                
                <h3>Benefits:</h3>
                <ul>
    """
    
    # Add benefits
    for benefit in YOGA_POSES[pose_data['pose_name']]['benefits']:
        html_content += f"<li>{benefit}</li>"
    
    html_content += """
                </ul>
                
                <div class="score">
                    Overall Score: <span style="color: #9370DB;">{:.1f}%</span>
                </div>
            </div>
            
            <h3>Feedback:</h3>
            <div class="feedback">
    """.format(pose_data['score'])
    
    # Add feedback
    for feedback in pose_data['feedback']:
        if "✅" in feedback:
            html_content += f'<p class="good">{feedback}</p>'
        elif "❌" in feedback:
            html_content += f'<p class="improve">{feedback}</p>'
        elif "⚠️" in feedback:
            html_content += f'<p class="warning">{feedback}</p>'
        else:
            html_content += f'<p>{feedback}</p>'
    
    html_content += """
            </div>
            
            <h3>Detailed Angle Analysis:</h3>
    """
    
    # Add angle measurements table
    html_content += """
            <table border="1" style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <tr style="background-color: #9370DB; color: white;">
                    <th>Measurement</th>
                    <th>Your Angle</th>
                    <th>Target Angle</th>
                    <th>Status</th>
                </tr>
    """
    
    # Add each angle measurement
    for name, data in detailed_angles.items():
        status = "✓ Correct" if data['is_correct'] else "✗ Needs Adjustment"
        status_color = "green" if data['is_correct'] else "red"
        html_content += f"""
                <tr>
                    <td>{name}</td>
                    <td>{data['measured']:.1f}°</td>
                    <td>{data['target']}° (±{data['tolerance']}°)</td>
                    <td style="color: {status_color};">{status}</td>
                </tr>
        """
    
    html_content += """
            </table>
    """
    
    # Add screenshot if available
    if screenshot is not None:
        # Convert the screenshot to base64 for embedding in HTML
        _, buffer_img = cv2.imencode('.png', screenshot)
        img_str = base64.b64encode(buffer_img).decode('utf-8')
        html_content += f"""
            <h3>Pose Screenshot:</h3>
            <img class="screenshot" src="data:image/png;base64,{img_str}" alt="Yoga Pose Screenshot">
        """
    
    # Add footer
    html_content += """
            <div class="footer">
                <p>Generated by Yoga Assistant | © 2025</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML content to the buffer
    buffer.write(html_content.encode())
    buffer.seek(0)
    
    return buffer
def main():
    """Main function to run the Streamlit app"""
    st.title("🧘Asana AI")
    
    # Add sidebar for options
    st.sidebar.header("Settings")
    
    # User information
    with st.sidebar.expander("User Profile", expanded=False):
        user_name = st.text_input("Your Name", "Name")
        user_experience = st.select_slider(
            "Yoga Experience Level",
            options=["Beginner", "Intermediate", "Advanced"],
            value="Intermediate"
        )
    
    
    session_mode = st.sidebar.radio(
        "Choose Mode",
        ["Live Practice", "Video Analysis", "Learning Hub", "Progress Tracker"]
    )
    
   
    if 'practice_history' not in st.session_state:
        st.session_state.practice_history = []
    
    if 'sequence_history' not in st.session_state:
        st.session_state.sequence_history = {}
    
    
    if session_mode == "Live Practice":
        live_practice_mode()
    elif session_mode == "Video Analysis":
        video_analysis_mode()
    elif session_mode == "Learning Hub":
        learning_hub_mode()
    else:  # Progress Tracker
        progress_tracker_mode()

def live_practice_mode():
    """Simple webcam implementation with continuous pose analysis"""
    st.header("Live Yoga Practice")
    

    col1, col2 = st.columns([3, 1])
    
    with col2:
      
        selected_pose = st.selectbox("Select Pose to Practice", list(YOGA_POSES.keys()))
        
        
        with st.expander("Pose Information", expanded=True):
            st.write(f"**Description**: {YOGA_POSES[selected_pose]['description']}")
            st.write(f"**Difficulty**: {YOGA_POSES[selected_pose]['difficulty']}")
            st.write("**Key Points:**")
            for angle in YOGA_POSES[selected_pose]['angles']:
                st.write(f"- {angle['name']}: {angle['target']}° (±{angle['tolerance']}°)")
        
        st.write("### Visualization Settings")
        visualization_mode = st.selectbox(
            "Visualization Mode",
            ["original", "enhanced","blur", "color", "remove", "skeleton",],
            index=0,
            help="Choose how to visualize your pose. 'Skeleton' shows only your body structure."
        )
        
        if visualization_mode == "color":
            bg_color = st.color_picker("Background Color", "#9370DB")
        else:
            bg_color = None
        
        show_angles = st.checkbox("Show Angle Measurements", True)
        show_breathing = st.checkbox("Show Breathing Guide", False)
        
        # Advanced settings
        with st.expander("Detection Settings", expanded=False):
            detection_confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.3, 0.05)
            
            if visualization_mode not in ["original", "skeleton"]:
                st.subheader("Segmentation Quality")
                segmentation_quality = st.select_slider(
                    "Segmentation Quality",
                    options=["Fast", "Balanced", "High Quality"],
                    value="Balanced",
                    help="Higher quality may be slower but provides better segmentation"
                )
                
                # Map quality settings to model selection
                if segmentation_quality == "Fast":
                    segmentation_model_idx = 0
                elif segmentation_quality == "Balanced":
                    segmentation_model_idx = 0
                else:  # High Quality
                    segmentation_model_idx = 1
            else:
                segmentation_model_idx = 0
    
    with col1:
        col_start, col_stop = st.columns(2)
        with col_start:
            start = st.button("Start Webcam", use_container_width=True)
        with col_stop:
            stop = st.button("Stop Webcam", use_container_width=True)
       
        frame_placeholder = st.empty()
        feedback_container = st.container()
        score_placeholder = st.empty()
        capture = st.button("Capture and Analyze Pose", use_container_width=True)
    
    if start:
  
        cap = cv2.VideoCapture(0)
        st.session_state.webcam_running = True
        pose_model = mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
            model_complexity=1  
        )
    
        segmentation_model = mp_selfie_segmentation.SelfieSegmentation(model_selection=segmentation_model_idx)
        
        # For storing latest results
        latest_feedback = []
        latest_score = 0
        latest_detailed_angles = {}
        
        while st.session_state.get('webcam_running', False):
           
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
          
            frame = cv2.flip(frame, 1)
       
            processed_frame, feedback, score, detailed_angles, pose_detected = process_frame(
                frame,
                pose_model,
                segmentation_model,
                selected_pose,
                detection_confidence,
                detection_confidence,
                visualization_mode,
                show_angles,
                bg_color,
                show_breathing
            )
            
          
            if pose_detected:
                latest_feedback = feedback
                latest_score = score
                latest_detailed_angles = detailed_angles
                
               
                with score_placeholder:
                    st.metric("Pose Score", f"{score:.1f}%")
          
            frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            
          
            if capture and pose_detected:
                with feedback_container:
                    st.success("Pose captured and analyzed!")
                   
                    correct_points = [fb for fb in feedback if "✅" in fb]
                    improvement_points = [fb for fb in feedback if "❌" in fb]
                    warnings = [fb for fb in feedback if "⚠️" in fb]
                    
                    if correct_points:
                        st.success("**Good alignment:**")
                        for point in correct_points:
                            st.write(point)
                    
                    if improvement_points:
                        st.error("**Needs improvement:**")
                        for point in improvement_points:
                            st.write(point)
                    
                    if warnings:
                        st.warning("**Watch out for:**")
                        for warning in warnings:
                            st.write(warning)
                    
                    
                    with st.expander("Detailed Angle Analysis", expanded=False):
                        angle_data = []
                        for name, details in detailed_angles.items():
                            angle_data.append({
                                'Angle': name,
                                'Your Measurement': f"{details['measured']:.1f}°",
                                'Target': f"{details['target']}° (±{details['tolerance']}°)",
                                'Status': '✅ Good' if details['is_correct'] else '❌ Adjust'
                            })
                        
                        if angle_data:
                            st.table(pd.DataFrame(angle_data))
                    
                    
                    st.session_state.last_frame = processed_frame
                    st.session_state.last_analysis = {
                        'pose_name': selected_pose,
                        'score': score,
                        'feedback': feedback,
                        'detailed_angles': detailed_angles,
                        'timestamp': datetime.datetime.now(),
                        'pose_detected': True
                    }
                    
                   
                    if 'practice_history' not in st.session_state:
                        st.session_state.practice_history = []
                    
                    st.session_state.practice_history.append(st.session_state.last_analysis)
                    
                  
                    report_buffer = generate_report(
                        user_name=st.session_state.get('user_name', 'User'),
                        pose_data=st.session_state.last_analysis,
                        detailed_angles=detailed_angles,
                        screenshot=processed_frame
                    )
                    
                    st.download_button(
                        label="Download Pose Analysis Report",
                        data=report_buffer,
                        file_name=f"yoga_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
                        mime="text/html"
                    )
            
            if stop:
                break
                
            
            time.sleep(0.05)
        
       
        cap.release()
        pose_model.close()
        segmentation_model.close()
        st.session_state.webcam_running = False
        
       
        frame_placeholder.empty()
        score_placeholder.empty()

def video_analysis_mode():
    """Video analysis mode for uploaded videos"""
    st.header("Video Analysis")
    
  
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
       
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
        temp_file.write(uploaded_file.read())
      
        col1, col2 = st.columns([2, 1])
        
        with col2:
           
            selected_pose = st.selectbox(
                "Select Pose to Analyze",
                list(YOGA_POSES.keys()),
                key="video_pose_select"
            )
            
            
            with st.expander("Pose Information", expanded=False):
                st.write(f"**Description**: {YOGA_POSES[selected_pose]['description']}")
                st.write(f"**Difficulty**: {YOGA_POSES[selected_pose]['difficulty']}")
                st.write("**Key Points to Check**:")
                for angle in YOGA_POSES[selected_pose]['angles']:
                    st.write(f"- {angle['name']}: {angle['target']}° (±{angle['tolerance']}°)")
            
          
            visualization_mode = st.selectbox(
                "Visualization Mode",
                ["original", "enhanced", "skeleton","blur", "color", "remove"],
                index=0,
                key="video_viz_mode",
                help="Choose how to visualize the pose analysis"
            )
            
            if visualization_mode == "color":
                bg_color = st.color_picker("Background Color", "#9370DB", key="video_bg_color")
            else:
                bg_color = None
            
            show_angles = st.checkbox("Show Angle Measurements", True, key="video_show_angles")
            
          
            with st.expander("Processing Parameters", expanded=False):
                detection_confidence = st.slider(
                    "Detection Confidence", 
                    min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                    key="video_detection_conf"
                )
                
                tracking_confidence = st.slider(
                    "Tracking Confidence", 
                    min_value=0.1, max_value=1.0, value=0.5, step=0.05,
                    key="video_tracking_conf"
                )
                
                sampling_rate = st.slider(
                    "Sampling Rate (frames)", 
                    min_value=1, max_value=30, value=5, step=1
                )
                
                # Add segmentation quality for enhanced modes
                if visualization_mode in ["enhanced","blur", "color", "remove"]:
                    st.subheader("Segmentation Quality")
                    segmentation_quality = st.select_slider(
                        "Segmentation Quality",
                        options=["Fast", "Balanced", "High Quality"],
                        value="Balanced",
                        key="video_seg_quality",
                        help="Higher quality may be slower but provides better segmentation"
                    )
                    
                  
                    if segmentation_quality == "Fast":
                        segmentation_model_idx = 0
                    elif segmentation_quality == "Balanced":
                        segmentation_model_idx = 0
                    else:  # High Quality
                        segmentation_model_idx = 1
                else:
                    segmentation_model_idx = 0
            
          
            if st.button("Start Analysis"):
                st.session_state.video_analysis_requested = True
                st.session_state.video_path = temp_file.name
                st.session_state.video_analysis_results = []
        
        with col1:
      
            if hasattr(st.session_state, 'video_analysis_requested') and st.session_state.video_analysis_requested:
               
                with mp_pose.Pose(
                    min_detection_confidence=detection_confidence,
                    min_tracking_confidence=tracking_confidence,
                    model_complexity=1 
                ) as pose, mp_selfie_segmentation.SelfieSegmentation(
                    model_selection=segmentation_model_idx
                ) as segmentation:
              
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    video = cv2.VideoCapture(st.session_state.video_path)
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = video.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps
                   
                    video_placeholder = st.empty()
                   
                    analysis_results = []
                    best_frame = None
                    best_score = 0
                    frame_scores = []
                    frame_times = []
                    
                    frame_count = 0
                    analyzed_count = 0
                    
                    while True:
                        ret, frame = video.read()
                        if not ret:
                            break
                        
                      
                        if frame_count % sampling_rate == 0:
                         
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"Analyzing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
                            
                            processed_frame, feedback, score, detailed_angles, pose_detected = process_frame(
                                frame,
                                pose,
                                segmentation,
                                selected_pose,
                                detection_confidence,
                                tracking_confidence,
                                visualization_mode,  
                                show_angles,
                                bg_color,
                                False 
                            )
                            
                      
                            video_placeholder.image(
                                cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                                caption=f"Frame {frame_count} - Score: {score:.1f}%",
                                use_column_width=True
                            )
                            
                            # Store analysis results
                            if pose_detected:
                                timestamp = frame_count / fps
                                frame_scores.append(score)
                                frame_times.append(timestamp)
                                
                             
                                result = {
                                    'frame': frame_count,
                                    'timestamp': timestamp,
                                    'score': score,
                                    'feedback': feedback,
                                    'detailed_angles': detailed_angles,
                                    'visualization_mode': visualization_mode  
                                }
                                analysis_results.append(result)
                                
                                # Track best frame
                                if score > best_score:
                                    best_score = score
                                    best_frame = processed_frame.copy()
                                
                                analyzed_count += 1
                        
                        frame_count += 1
                    video.release()
                    
                    # Update progress to completion
                    progress_bar.progress(1.0)
                    status_text.text(f"Analysis complete! Analyzed {analyzed_count} frames.")
                    
                    # Store results in session state
                    st.session_state.video_analysis_results = analysis_results
                    st.session_state.video_best_frame = best_frame
                    st.session_state.video_best_score = best_score
                    st.session_state.video_frame_scores = frame_scores
                    st.session_state.video_frame_times = frame_times
                    st.session_state.video_visualization_mode = visualization_mode
                    
                    # Reset analysis request flag
                    st.session_state.video_analysis_requested = False
            
            # Display analysis results if available
            if hasattr(st.session_state, 'video_analysis_results') and st.session_state.video_analysis_results:
                st.subheader("Analysis Results")
               
                if hasattr(st.session_state, 'video_best_frame'):
                    st.image(
                        cv2.cvtColor(st.session_state.video_best_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Best Frame - Score: {st.session_state.video_best_score:.1f}%",
                        use_column_width=True
                    )
               
                if hasattr(st.session_state, 'video_visualization_mode'):
                    viz_mode = st.session_state.video_visualization_mode
                    st.info(f"Visualization used: {viz_mode.title()} mode")
               
                if hasattr(st.session_state, 'video_frame_scores') and len(st.session_state.video_frame_scores) > 0:
                    fig = px.line(
                        x=st.session_state.video_frame_times,
                        y=st.session_state.video_frame_scores,
                        labels={'x': 'Time (seconds)', 'y': 'Pose Score (%)'},
                        title='Pose Score Over Time'
                    )
                    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Excellent")
                    fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Good")
                    st.plotly_chart(fig, use_container_width=True)
                
               
                if st.session_state.video_analysis_results and len(st.session_state.video_analysis_results) > 1:
                    angle_data = {}
                    
                    for result in st.session_state.video_analysis_results:
                        if 'detailed_angles' in result:
                            for angle_name, details in result['detailed_angles'].items():
                                if angle_name not in angle_data:
                                    angle_data[angle_name] = {
                                        'times': [],
                                        'angles': [],
                                        'target': details['target']
                                    }
                                
                                angle_data[angle_name]['times'].append(result['timestamp'])
                                angle_data[angle_name]['angles'].append(details['measured'])
                    
                   
                    st.subheader("Angle Measurements Over Time")
                    
                    for angle_name, data in angle_data.items():
                        if len(data['times']) > 1:
                            fig = px.line(
                                x=data['times'], 
                                y=data['angles'],
                                labels={'x': 'Time (seconds)', 'y': 'Angle (degrees)'},
                                title=f'{angle_name} Over Time'
                            )
                         
                            fig.add_hline(
                                y=data['target'], 
                                line_dash="dash", 
                                line_color="green", 
                                annotation_text="Target"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Generate report
                st.subheader("Generate Report")
                best_result = max(st.session_state.video_analysis_results, key=lambda x: x['score'])
             
                if st.button("Generate Video Analysis Report"):
                    report_buffer = generate_report(
                        user_name=st.session_state.get('user_name', 'User'),
                        pose_data={
                            'pose_name': selected_pose,
                            'score': best_result['score'],
                            'feedback': best_result['feedback']
                        },
                        detailed_angles=best_result['detailed_angles'],
                        screenshot=st.session_state.video_best_frame
                    )
                    
                    st.download_button(
                        label="Download Analysis Report (HTML)",
                        data=report_buffer,
                        file_name=f"yoga_video_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
                        mime="text/html"
                    )
                
               
                st.subheader("Export Analysis Video")
                if st.button("Coming Soon: Export Analysis Video Clip"):
                    st.info("This feature is coming in the next update! It will allow you to export a video clip of your best pose segment with overlaid analysis.")
def learning_hub_mode():
    """Learning hub with tutorials and pose guides"""
    st.header("Yoga Learning Hub")
    tab1, tab2, tab3 = st.tabs(["Pose Library", "Sequences", "Tutorials"])
    
    with tab1:
        st.subheader("Yoga Pose Library")
        col1, col2 = st.columns(2)
        with col1:
            difficulty_filter = st.multiselect(
                "Filter by Difficulty",
                ["Beginner", "Intermediate", "Advanced"],
                default=["Beginner", "Intermediate"]
            )
        
        with col2:
          
            search_term = st.text_input("Search Poses", "")
        
        filtered_poses = {
            name: details for name, details in YOGA_POSES.items()
            if details["difficulty"] in difficulty_filter and 
               (search_term.lower() in name.lower() or 
                search_term.lower() in details["description"].lower())
        }
        
        if not filtered_poses:
            st.info("No poses match your filters. Try adjusting your search criteria.")
        else:
           
            cols = st.columns(3)
            
            for i, (pose_name, pose_details) in enumerate(filtered_poses.items()):
                with cols[i % 3]:
                    with st.expander(f"{pose_name} ({pose_details['difficulty']})", expanded=False):
                        st.image(POSE_IMAGES.get(pose_name, "https://cdn-icons-png.flaticon.com/512/2043/2043831.png"), 
                                width=150)
                        
                        st.write(pose_details["description"])
                        st.write("**Benefits:**")
                        for benefit in pose_details["benefits"]:
                            st.write(f"- {benefit}")
                        
                        st.write("**Key Alignments:**")
                        for angle in pose_details["angles"]:
                            st.write(f"- {angle['name']}: {angle['target']}° (±{angle['tolerance']}°)")
                        
                        if st.button("Practice This Pose", key=f"practice_{pose_name}"):
                            st.session_state.selected_practice_pose = pose_name
                            st.rerun()
    
    with tab2:
        st.subheader("Yoga Sequences")
        
        for seq_name, poses in YOGA_SEQUENCES.items():
            with st.expander(seq_name, expanded=False):
              
                if seq_name == "Morning Energizer":
                    st.write("A revitalizing sequence to wake up your body and mind.")
                elif seq_name == "Balance Focus":
                    st.write("Improve stability and focus with these balancing poses.")
                elif seq_name == "Core Strength":
                    st.write("Build abdominal strength and stability with this sequence.")
                elif seq_name == "Beginner Flow":
                    st.write("An accessible sequence perfect for yoga beginners.")
                elif seq_name == "Full Body Workout":
                    st.write("A comprehensive sequence targeting all major muscle groups.")
                
                st.write("**Poses in this sequence:**")
                for i, pose in enumerate(poses, 1):
                    st.write(f"{i}. {pose}")
   
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    if st.button("Start Practice", key=f"start_{seq_name}"):
                        st.session_state.selected_sequence = seq_name
                        st.session_state.sequence_poses = poses
                        st.session_state.current_pose_index = 0
                        st.rerun()
                
                with btn_col2:
                    if st.button("Learn More", key=f"learn_{seq_name}"):
                        st.session_state.show_sequence_details = seq_name
                        st.rerun()
    
    with tab3:
        st.subheader("Yoga Tutorials")
        tutorials = [
    {
        "title": "Yoga Breathing Techniques",
        "description": "Learn fundamental pranayama techniques to enhance your practice.",
        "image": "ypic\\eath.png",  
        "link": "https://www.youtube.com/watch?v=tbMK48EoaBA&ab_channel=YogaWithAdriene"            
    },
    {
        "title": "Proper Alignment Basics",
        "description": "Understand the fundamentals of safe alignment for any pose.",
        "image": "ypic\\pos.png",  
        "link": "https://www.youtube.com/watch?v=HTuSi6TZxRs&ab_channel=Yoga%26You"
    },
    {
        "title": "Meditation for Beginners",
        "description": "Simple meditation techniques to integrate into your yoga practice.",
        "image": "ypic\\ex.png", 
        "link": "https://www.youtube.com/watch?v=VpHz8Mb13_Y&ab_channel=Lavendaire"
    }
]
        
        tutorial_cols = st.columns(3)
    for i, tutorial in enumerate(tutorials):
        with tutorial_cols[i]:
            st.image(tutorial["image"], width=100)
            st.write(f"**{tutorial['title']}**")
            st.write(tutorial["description"])
           
            if st.button("Watch Tutorial", key=f"tutorial_{i}"):
                st.markdown(f"<a href='{tutorial['link']}' target='_blank'>Click to open tutorial</a>", 
                        unsafe_allow_html=True)
                st.markdown(f"""
                    <script>window.open('{tutorial["link"]}', '_blank').focus();</script>
                    """, unsafe_allow_html=True)
            
def progress_tracker_mode():
    """Progress tracking and statistics"""
    st.header("Your Yoga Progress")
    
    if 'practice_history' not in st.session_state:
        st.session_state.practice_history = []
    
    if not st.session_state.practice_history:
        st.info("No practice history available yet. Complete some pose analyses in Live Practice mode to track your progress.")
        st.write("Here's how your progress tracker will look after you practice:")
        import random
        dates = pd.date_range(end=datetime.datetime.now(), periods=10).tolist()
        example_poses = ["Warrior II", "Tree Pose", "Downward Dog", "Cobra Pose", "Chair Pose"]
        example_scores = [random.uniform(60, 95) for _ in range(10)]
        
        fig = px.line(
            x=dates, 
            y=example_scores,
            labels={'x': 'Date', 'y': 'Pose Score (%)'},
            title='Example Progress Chart'
        )
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Excellent")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Good")
        st.plotly_chart(fig, use_container_width=True)
        
        # Example pose breakdown
        st.subheader("Example Pose Mastery")
        pose_data = {
            'Pose': example_poses,
            'Average Score': [85, 75, 90, 65, 70],
            'Practice Count': [7, 5, 10, 3, 4]
        }
        
        df = pd.DataFrame(pose_data)
        fig = px.bar(
            df,
            x='Pose',
            y='Average Score',
            color='Average Score',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[50, 100],
            title='Example Pose Mastery'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return

    tab1, tab2, tab3 = st.tabs(["Overview", "Pose Details", "Practice History"])
    
  
    try:
        
        history_data = []
        for entry in st.session_state.practice_history:
            if not isinstance(entry, dict):
                continue
            entry_data = {
                'date': entry.get('timestamp', datetime.datetime.now()).date() if hasattr(entry.get('timestamp', datetime.datetime.now()), 'date') else datetime.datetime.now().date(),
                'timestamp': entry.get('timestamp', datetime.datetime.now()),
                'pose': entry.get('pose_name', 'Unknown'),
                'score': float(entry.get('score', 0)),
                'detected': entry.get('pose_detected', False)
            }
            
            history_data.append(entry_data)
        if history_data:
            df = pd.DataFrame(history_data)
        else:
            df = pd.DataFrame(columns=['date', 'timestamp', 'pose', 'score', 'detected'])
    except Exception as e:
        st.error(f"Error processing history data: {e}")
        df = pd.DataFrame(columns=['date', 'timestamp', 'pose', 'score', 'detected'])
    
    with tab1:
        st.subheader("Practice Overview")
        
        if df.empty:
            st.warning("No valid practice data found. Try completing some poses in Live Practice mode.")
        else:
            total_sessions = len(df)
            total_time = total_sessions * 2  
            avg_score = df['score'].mean() if not df.empty else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Practice Sessions", total_sessions)
            col2.metric("Total Practice Time", f"{total_time} minutes")
            col3.metric("Average Pose Score", f"{avg_score:.1f}%")
            st.subheader("Progress Over Time")
            
            try:
               
                df = df.sort_values('timestamp')
                daily_progress = df.groupby('date')['score'].mean().reset_index()
                if not daily_progress.empty:
                   
                    fig = px.line(
                        daily_progress, 
                        x='date', 
                        y='score',
                        labels={'date': 'Date', 'score': 'Average Score (%)'},
                        title='Your Progress Over Time'
                    )
                    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Excellent")
                    fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Good")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for daily progress chart.")
            except Exception as e:
                st.error(f"Error creating progress chart: {e}")
        
            try:
                if 'pose' in df.columns and not df.empty:
                    st.subheader("Most Practiced Poses")
                    pose_counts = df['pose'].value_counts().reset_index()
                    pose_counts.columns = ['Pose', 'Count']
                    
                    fig = px.bar(
                        pose_counts, 
                        x='Pose', 
                        y='Count',
                        title='Poses by Practice Frequency'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No pose data available for visualization.")
            except Exception as e:
                st.error(f"Error creating pose frequency chart: {e}")
    
    with tab2:
        st.subheader("Pose-Specific Progress")
        
        if df.empty:
            st.warning("No practice data available yet. Try some poses in Live Practice mode.")
        else:
            try:
                practiced_poses = df['pose'].unique().tolist() if 'pose' in df.columns else []
                if practiced_poses:
                    selected_pose = st.selectbox("Select Pose", practiced_poses)
                    pose_df = df[df['pose'] == selected_pose].copy()
                    if not pose_df.empty:
                        if selected_pose in YOGA_POSES:
                            st.write(f"**Description**: {YOGA_POSES[selected_pose]['description']}")
                            st.write(f"**Difficulty**: {YOGA_POSES[selected_pose]['difficulty']}")
                        st.subheader(f"{selected_pose} Progress")
                        if len(pose_df) >= 2:
                            pose_df = pose_df.sort_values('timestamp')
                            fig = px.line(
                                pose_df, 
                                x='timestamp', 
                                y='score',
                                labels={'timestamp': 'Date/Time', 'score': 'Score (%)'},
                                title=f'{selected_pose} Score Over Time'
                            )
                            fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Excellent")
                            fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Good")
                            st.plotly_chart(fig, use_container_width=True)
                            
                           
                            first_score = pose_df['score'].iloc[0]
                            last_score = pose_df['score'].iloc[-1]
                            improvement = last_score - first_score
                            
                            delta_color = "normal" if improvement >= 0 else "inverse"
                            st.metric(
                                "Improvement Since First Practice", 
                                f"{last_score:.1f}%",
                                delta=f"{improvement:.1f}%",
                                delta_color=delta_color
                            )
                        else:
                            st.info("Need at least two practice sessions to show progress.")
                        
                    
                        recent_entry = None
                        for entry in reversed(st.session_state.practice_history):
                            if entry.get('pose_name') == selected_pose and 'detailed_angles' in entry:
                                recent_entry = entry
                                break
                        
                        if recent_entry and 'detailed_angles' in recent_entry:
                            st.subheader("Latest Angle Analysis")
                          
                            angle_data = []
                            for name, data in recent_entry['detailed_angles'].items():
                                angle_data.append({
                                    'Angle': name,
                                    'Your Measurement': data['measured'],
                                    'Target': data['target'],
                                    'Difference': abs(data['measured'] - data['target']),
                                    'Status': 'Correct' if data['is_correct'] else 'Needs Improvement'
                                })
                            
                            if angle_data:
                                angle_df = pd.DataFrame(angle_data)
                                st.table(angle_df[['Angle', 'Your Measurement', 'Target', 'Status']])
                                fig = px.bar(
                                    angle_df,
                                    x='Angle',
                                    y=['Your Measurement', 'Target'],
                                    barmode='group',
                                    title='Angle Measurements vs. Targets'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No practice data available for {selected_pose}")
                else:
                    st.info("No poses found in your practice history.")
            except Exception as e:
                st.error(f"Error analyzing pose progress: {e}")
    
    with tab3:
        st.subheader("Practice History")
        
        try:
            if not df.empty:
           
                history_display = []
                for i, (_, row) in enumerate(df.sort_values('timestamp', ascending=False).iterrows(), 1):
                    history_display.append({
                        'Session': i,
                        'Date': row['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(row['timestamp'], 'strftime') else 'Unknown',
                        'Pose': row['pose'],
                        'Score': f"{row['score']:.1f}%",
                        'Status': 'Good' if row['score'] >= 70 else 'Needs Practice'
                    })
                
                if history_display:
                    history_df = pd.DataFrame(history_display)
                    st.dataframe(history_df, use_container_width=True)
           
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Clear Practice History"):
                            st.session_state.practice_history = []
                            st.success("Practice history cleared successfully!")
                            st.experimental_rerun()
                    
                    with col2:
                        csv = history_df.to_csv(index=False)
                        st.download_button(
                            "Download Practice History (CSV)",
                            csv,
                            "yoga_practice_history.csv",
                            "text/csv",
                            key='download-csv'
                        )
                else:
                    st.info("No practice history available yet.")
            else:
                st.info("No practice history available yet.")
        except Exception as e:
            st.error(f"Error displaying practice history: {e}")

        with st.expander("Debug: View Raw Practice Data", expanded=False):
            st.json(st.session_state.practice_history)
def guided_sequence_mode(sequence_name, poses):
    """Guided practice for a yoga sequence"""
    st.header(f"Guided Practice: {sequence_name}")
    
    if 'current_pose_index' not in st.session_state:
        st.session_state.current_pose_index = 0
  
    current_index = st.session_state.current_pose_index
    if current_index >= len(poses):
        current_index = len(poses) - 1
        st.session_state.current_pose_index = current_index
    
    current_pose = poses[current_index]
    
    progress_text = f"Pose {current_index + 1} of {len(poses)}"
    st.progress(float(current_index + 1) / len(poses))
    st.subheader(f"Current Pose: {current_pose} ({progress_text})")
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.write(f"**Description**: {YOGA_POSES[current_pose]['description']}")
        st.write(f"**Difficulty**: {YOGA_POSES[current_pose]['difficulty']}")
        
        st.write("**Benefits:**")
        for benefit in YOGA_POSES[current_pose]['benefits']:
            st.write(f"- {benefit}")
        
        st.write("**Key Alignments:**")
        for angle in YOGA_POSES[current_pose]['angles']:
            st.write(f"- {angle['name']}: {angle['target']}° (±{angle['tolerance']}°)")
        
   
        st.write("### Visualization Settings")
        visualization_mode = st.selectbox(
            "Visualization Mode",
            ["original", "enhanced", "skeleton","blur", "color", "remove"],
            index=0,
            key="seq_viz_mode",
            help="Choose how to visualize your pose."
        )
        
        if visualization_mode == "color":
            bg_color = st.color_picker("Background Color", "#9370DB", key="seq_bg_color")
        else:
            bg_color = None
        
        show_angles = st.checkbox("Show Angle Measurements", True, key="seq_show_angles")
        show_breathing = st.checkbox("Show Breathing Guide", True, key="seq_breathing")
    
        hold_time = st.slider("Hold Time (seconds)", 30, 120, 60, step=10)
        
      
        with st.expander("Detection Settings", expanded=False):
            detection_confidence = st.slider(
                "Detection Confidence", 
                min_value=0.1, max_value=1.0, value=0.3, step=0.05,
                key="seq_detection_conf"
            )
            
            if visualization_mode not in ["original", "skeleton"]:
                st.subheader("Segmentation Quality")
                segmentation_quality = st.select_slider(
                    "Segmentation Quality",
                    options=["Fast", "Balanced", "High Quality"],
                    value="Balanced",
                    key="seq_seg_quality",
                    help="Higher quality may be slower but provides better segmentation"
                )
                if segmentation_quality == "Fast":
                    segmentation_model_idx = 0
                elif segmentation_quality == "Balanced":
                    segmentation_model_idx = 0
                else: 
                    segmentation_model_idx = 1
            else:
                segmentation_model_idx = 0
        
        with st.form(key="sequence_navigation"):
            button_cols = st.columns(3)
            
            with button_cols[0]:
                prev_button = st.form_submit_button("Previous Pose", use_container_width=True)
                    
            with button_cols[1]:
                timer_button = st.form_submit_button("Start Timer", use_container_width=True)
            
            with button_cols[2]:
                next_button = st.form_submit_button("Next Pose", use_container_width=True)
        
      
        if prev_button and st.session_state.current_pose_index > 0:
            st.session_state.current_pose_index -= 1
            if 'timer_running' in st.session_state:
                del st.session_state.timer_running
            st.rerun()
                
        if timer_button:
            st.session_state.timer_running = True
            st.session_state.timer_start = time.time()
            st.session_state.timer_duration = hold_time
        
        if next_button and st.session_state.current_pose_index < len(poses) - 1:
            st.session_state.current_pose_index += 1
            if 'timer_running' in st.session_state:
                del st.session_state.timer_running
            st.rerun()
        
        if st.button("Exit Sequence", use_container_width=True):
            if 'selected_sequence' in st.session_state:
                del st.session_state.selected_sequence
            if 'sequence_poses' in st.session_state:
                del st.session_state.sequence_poses
            if 'current_pose_index' in st.session_state:
                del st.session_state.current_pose_index
            if 'timer_running' in st.session_state:
                del st.session_state.timer_running
            st.rerun()
            
    with col1:
        col_start, col_stop = st.columns(2)
        with col_start:
            start = st.button("Start Webcam", key="seq_start", use_container_width=True)
        with col_stop:
            stop = st.button("Stop Webcam", key="seq_stop", use_container_width=True)
    
        frame_placeholder = st.empty()
        feedback_container = st.container()
        score_placeholder = st.empty()
        timer_placeholder = st.empty()
    
        # Use the same webcam approach as in live_practice_mode
        if start:
            cap = cv2.VideoCapture(0)
            st.session_state.seq_webcam_running = True
            pose_model = mp_pose.Pose(
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=detection_confidence,
                model_complexity=1
            )
            
            segmentation_model = mp_selfie_segmentation.SelfieSegmentation(model_selection=segmentation_model_idx)
            
            # For storing latest results
            latest_feedback = []
            latest_score = 0
            latest_detailed_angles = {}
            
            while st.session_state.get('seq_webcam_running', False):
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    break
                frame = cv2.flip(frame, 1)
                
                # Process frame with continuous pose analysis and segmentation
                processed_frame, feedback, score, detailed_angles, pose_detected = process_frame(
                    frame,
                    pose_model,
                    segmentation_model,
                    current_pose,
                    detection_confidence,
                    detection_confidence,
                    visualization_mode,
                    show_angles,
                    bg_color,
                    show_breathing
                )
                
                # Store latest results
                if pose_detected:
                    latest_feedback = feedback
                    latest_score = score
                    latest_detailed_angles = detailed_angles
                    
                    # Update feedback in real-time
                    with score_placeholder:
                        st.metric("Pose Score", f"{score:.1f}%")
                
                # Add sequence progress text to frame
                cv2.putText(
                    processed_frame, 
                    f"Sequence: {sequence_name} - {progress_text}",
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                
                # Add timer if running
                if 'timer_running' in st.session_state and st.session_state.timer_running:
                    elapsed = time.time() - st.session_state.timer_start
                    remaining = max(0, st.session_state.timer_duration - elapsed)
                    
                    # Draw timer on frame
                    cv2.putText(
                        processed_frame,
                        f"Hold: {int(remaining)}s",
                        (processed_frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2
                    )
                    
                    # Draw progress circle for timer
                    center = (processed_frame.shape[1] - 50, 60)
                    radius = 30
                    
                    # Background circle
                    cv2.circle(processed_frame, center, radius, (50, 50, 50), -1)
                    
                    # Progress arc (ensure angles are integers)
                    progress_angle = int(360 * (1 - remaining / st.session_state.timer_duration))
                    cv2.ellipse(
                        processed_frame, 
                        center, 
                        (radius, radius), 
                        -90, 
                        0, 
                        progress_angle, 
                        (0, 255, 255), 
                        5
                    )
                    
                    # Update timer display in UI
                    with timer_placeholder:
                        timer_cols = st.columns([1, 2])
                        timer_cols[0].metric("Time Remaining", f"{int(remaining)}s")
                        timer_cols[1].progress(1 - (remaining / st.session_state.timer_duration))
                    
                    # Check if timer is complete
                    if remaining <= 0:
                        st.session_state.timer_running = False
                        
                        # Add to sequence history with latest score and feedback
                        if sequence_name not in st.session_state.sequence_history:
                            st.session_state.sequence_history[sequence_name] = []
                        
                        st.session_state.sequence_history[sequence_name].append({
                            'pose': current_pose,
                            'score': latest_score,
                            'feedback': latest_feedback,
                            'detailed_angles': latest_detailed_angles,
                            'timestamp': datetime.datetime.now()
                        })
                        
                        # Show completion message
                        with feedback_container:
                            st.success(f"✅ Completed {current_pose}! Score: {latest_score:.1f}%")
                            
                            if latest_score >= 80:
                                st.balloons()
                            
                            # Offer to proceed to next pose if available
                            if st.session_state.current_pose_index < len(poses) - 1:
                                # Create a callback for the continue button
                                def go_to_next_pose():
                                    st.session_state.current_pose_index += 1
                                    st.experimental_rerun()
                                
                                st.button("Continue to next pose", key="auto_next", on_click=go_to_next_pose)
                
                # Display the processed frame
                frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                                       channels="RGB", use_column_width=True)
                
                # Display feedback when pose is detected
                if pose_detected and timer_placeholder.empty():  # Don't update if timer is showing
                    with feedback_container:
                        # Display feedback by category
                        correct_points = [fb for fb in feedback if "✅" in fb]
                        improvement_points = [fb for fb in feedback if "❌" in fb]
                        warnings = [fb for fb in feedback if "⚠️" in fb]
                        
                        if correct_points:
                            st.success("**Good alignment:**")
                            for point in correct_points:
                                st.write(point)
                        
                        if improvement_points:
                            st.error("**Needs improvement:**")
                            for point in improvement_points:
                                st.write(point)
                        
                        if warnings:
                            st.warning("**Watch out for:**")
                            for warning in warnings:
                                st.write(warning)
                
              
                if stop:
                    break
                  
                time.sleep(0.05)
            
            cap.release()
            pose_model.close()
            segmentation_model.close()
            st.session_state.seq_webcam_running = False
            
            frame_placeholder.empty()
            score_placeholder.empty()
            timer_placeholder.empty()
        
        if st.session_state.current_pose_index == len(poses) - 1 and 'timer_running' in st.session_state and not st.session_state.timer_running:
            with feedback_container:
                st.success("🎉 Congratulations! You've completed the entire sequence!")
                st.balloons()
            
                if sequence_name in st.session_state.sequence_history and st.session_state.sequence_history[sequence_name]:
                    scores = [entry['score'] for entry in st.session_state.sequence_history[sequence_name]]
                    avg_score = sum(scores) / len(scores)
                    st.metric("Overall Sequence Score", f"{avg_score:.1f}%")
                    
                    if st.button("Generate Sequence Report", key="seq_report"):
                        st.info("Sequence report functionality coming soon!")

if __name__ == "__main__":
    if hasattr(st.session_state, 'selected_practice_pose'):
        st.session_state.session_mode = "Live Practice"
        selected_pose = st.session_state.selected_practice_pose
        del st.session_state.selected_practice_pose
        main()
    
    elif hasattr(st.session_state, 'selected_sequence') and hasattr(st.session_state, 'sequence_poses'):
        guided_sequence_mode(st.session_state.selected_sequence, st.session_state.sequence_poses)
    else:
        main()
