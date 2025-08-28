Automatic Number Plate Detection & Replacement with Company Logo

This project demonstrates the integration of an Automatic Number Plate Detection (ANPD) pipeline using YOLOv8 and OpenCV, along with optional EasyOCR for plate text extraction.The system detects license plates in vehicle images and replaces them with a given company logo.

Features:
1) YOLOv8 for robust license plate detection
2) Perspective-aware logo placement 
3) EasyOCR integration for plate text extraction
4) Flask app for multiple file uploads and batch processing

Tech Stack:

1) Computer Vision & Machine Learning

a) Object Detection: YOLO 
b) OCR: EasyOCR
c) Image Processing: OpenCV
d) Numerical Operations: NumPy

2) Backend

a) Python API: Flask 
b) File Handling: Werkzeug 


Project Structure:
.
├── ai/                     # Virtual environment 
├── outputs/                # Stores processed result ZIP files
│   └── processed_images.zip
├── templates/              # HTML templates for Flask
│   └── index.html          # Upload UI
├── app.py                  # Flask web app 
├── main.py                 # Standalone script for single-image testing
├── README.md               # Documentation
└── requirements.txt        # Python dependencies

A standalone script (main.py) is for single image number platedection using YOLO and company logo replacement logic using OpenCV.

A Flask web application (app.py) for uploading multiple images, a YOLO model, and a logo, and then downloading the processed results as a ZIP.

Running the Application:

1) Create env for project:
        python -m env ai
2) Activate env:
        ai\Scripts\activate
3) Install Python Dependencies:
        python install -r requirements.txt
4) Single Image Script (main.py):
        python main.py

    The script will:
        Detect license plate
        Run OCR (optional) to print detected plate text
        Replace the plate region with the logo
        Save & show the final result

5) Flask Web Application (app.py):
        python app.py

    The app runs at localhost:5000
    Upload:
        YOLO model file (.pt)
        Company logo 
        One or more vehicle images
    
    The app will:
        Detect and replace plates
        Zip processed images