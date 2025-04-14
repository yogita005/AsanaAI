# Asana AI: AI-Powered Yoga Assistant

## Overview

Asana AI is a comprehensive yoga assistant application that uses computer vision and pose detection to help users improve their yoga practice. The application analyzes yoga poses in real-time, provides personalized feedback on alignment, and tracks progress over time.

https://github.com/user-attachments/assets/0afdfddd-51a5-4694-8a26-69130971d99d

## Features

### Live Practice Mode
- Real-time pose detection and analysis
- Multiple visualization options (original, enhanced, blur, color, remove background, skeleton)
- Detailed angle measurements and alignment feedback
- Breathing guide for better practice
- Capture and analyze specific frames
- Generate downloadable practice reports

### Video Analysis
- Upload and analyze pre-recorded yoga videos
- Frame-by-frame pose detection
- Score tracking throughout the video
- Angle analysis over time
- Identification of best frames and poses
- Generate comprehensive analysis reports

### Guided Sequences
- Follow predefined yoga sequences
- Timed pose holds with on-screen countdown
- Progress tracking through sequences
- Real-time feedback on each pose
- Sequence completion metrics

### Progress Tracking
- Long-term pose improvement tracking
- Visual charts of progress over time
- Pose-specific history and analytics
- Detailed angle comparisons between sessions
- Exportable practice history

### Learning Hub
- Pose library with detailed descriptions and benefits
- Yoga sequence guides for different goals
- Tutorial integration with helpful resources
- Searchable pose database with difficulty filters

## Technical Details

### Technology Stack
- **Python**: Core programming language
- **Streamlit**: Web application framework and UI
- **OpenCV**: Image and video processing
- **MediaPipe**: Pose detection and landmark tracking
- **NumPy**: Numerical operations
- **Pandas**: Data analysis and manipulation
- **Plotly**: Interactive data visualization
- **PIL**: Image manipulation

### Key Components
- **Pose Detection**: Uses MediaPipe's pose detection model to identify key body landmarks
- **Angle Calculation**: Measures key angles between body parts for pose analysis
- **Segmentation**: Background removal and effects using MediaPipe's Selfie Segmentation
- **Report Generation**: Custom HTML report generation with detailed metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/asana-ai.git
cd asana-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run newyoga.py
```

## Requirements

```
streamlit==1.28.0
opencv-python==4.8.1
mediapipe==0.10.7
numpy==1.24.4
pandas==2.1.1
plotly==5.17.0
pillow==10.0.1
matplotlib==3.8.0
```

## Usage

1. **Select a Mode**: Choose between Live Practice, Video Analysis, Learning Hub, or Progress Tracker
2. **Live Practice**: 
   - Select a pose to practice
   - Configure visualization settings
   - Start webcam and follow alignment guidance
   - Capture frames for detailed analysis

3. **Video Analysis**:
   - Upload a yoga video
   - Select the pose to analyze
   - Adjust detection parameters
   - View the analysis results

4. **Guided Sequences**:
   - Select a predefined sequence
   - Follow the timed instructions for each pose
   - Track your score through the sequence

5. **Progress Tracker**:
   - Monitor your improvement over time
   - View pose-specific analytics
   - Export your practice history

