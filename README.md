# Driver-Drowsiness-Alarm-System

## Overview
This system is designed to monitor drowsiness by analyzing eye movements, blinking patterns, and yawning in real time using computer vision and machine learning techniques. It provides alerts when signs of fatigue are detected, helping to prevent accidents and improve safety.

## Features
- **Real-time Eye Blink Detection**: Uses facial landmark detection to monitor blinking patterns.
- **Yawning Detection**: Detects yawning by calculating the Mouth Aspect Ratio (MAR).
- **Adaptive Thresholds**: Adjustable sensitivity to improve accuracy.
- **Low-Light Enhancement**: Implements histogram equalization for better face detection in poor lighting.
- **Alarm System**: Triggers an alert sound when drowsiness or excessive yawning is detected.
- **Efficient Processing**: Optimized with NumPy and multithreading for real-time performance.

## Technologies Used
- Python
- OpenCV
- Dlib (Facial Landmark Detection)
- NumPy
- Multithreading

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/paridhi-singhal/Driver-Drowsiness-Alarm-System.git
   cd drowsiness-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the program:
   ```bash
   python app.py
   ```

## Usage
- The system will continuously analyze video input from the webcam.
- If prolonged eye closure or yawning is detected, an alert message will be displayed, and an alarm will sound.
- Press `r` to reset the alarm manually.
- Press `Esc` to exit the program.


