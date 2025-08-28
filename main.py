#Single Image Script 

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO  # import all the dependencies

# STEP 1: Load YOLOv8 License Plate Model 

model_path = r"C:\Users\Admin\Desktop\SportzInteractive2\best.pt"
#model_path = r"C:\Users\Admin\Desktop\SportzInteractive2\best2.pt"
model = YOLO(model_path)
reader = easyocr.Reader(['en'], gpu=False)  # EasyOCR for plate text recognition (optional)

# STEP 2: Setting Scaling logic for logo fitting

# Aspect-ratio preserving logo placement 
def place_logo_in_box(base_image, logo_rgba, x1, y1, x2, y2, margin_ratio=0.9):
    x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(base_image.shape[1], int(x2)); y2 = min(base_image.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return

    target_w = x2 - x1
    target_h = y2 - y1
    # Apply margin inside the box
    inner_w = max(1, int(target_w * margin_ratio))
    inner_h = max(1, int(target_h * margin_ratio))

    # Original logo size and channels
    logo_h, logo_w = logo_rgba.shape[:2]
    has_alpha = logo_rgba.shape[2] == 4 if len(logo_rgba.shape) == 3 else False

    # Compute scale to fit inside inner box while preserving aspect
    scale = min(inner_w / float(logo_w), inner_h / float(logo_h))
    new_w = max(1, int(round(logo_w * scale)))
    new_h = max(1, int(round(logo_h * scale)))

    # Choose interpolation based on scaling direction
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized_logo = cv2.resize(logo_rgba, (new_w, new_h), interpolation=interp)

    # Center within target box
    offset_x = x1 + (target_w - new_w) // 2
    offset_y = y1 + (target_h - new_h) // 2 #Places logo in center of bounding box
    # Clip to image bounds
    xA = max(0, offset_x)
    yA = max(0, offset_y)
    xB = min(base_image.shape[1], offset_x + new_w)
    yB = min(base_image.shape[0], offset_y + new_h)
    if xB <= xA or yB <= yA:
        return

    logo_xA = xA - offset_x
    logo_yA = yA - offset_y
    logo_xB = logo_xA + (xB - xA)
    logo_yB = logo_yA + (yB - yA)

    roi = base_image[yA:yB, xA:xB]
    logo_cropped = resized_logo[logo_yA:logo_yB, logo_xA:logo_xB]

    if has_alpha and logo_cropped.shape[2] == 4:
        alpha = (logo_cropped[:, :, 3].astype(np.float32) / 255.0)[..., None]
        rgb_logo = logo_cropped[:, :, :3].astype(np.float32)
        roi_float = roi.astype(np.float32)
        blended = alpha * rgb_logo + (1.0 - alpha) * roi_float
        base_image[yA:yB, xA:xB] = blended.astype(np.uint8)
    else:
        base_image[yA:yB, xA:xB] = logo_cropped[:, :, :3]

# Perspective-aware placement 
def _order_points_clockwise(points):
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def place_logo_in_quad(base_image, logo_rgba, quad_points):
    quad = _order_points_clockwise(quad_points)
    h_logo, w_logo = logo_rgba.shape[:2]
    src = np.array([[0, 0], [w_logo - 1, 0], [w_logo - 1, h_logo - 1], [0, h_logo - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, quad)

    has_alpha = logo_rgba.shape[2] == 4 if len(logo_rgba.shape) == 3 else False
    if has_alpha:
        logo_rgb = logo_rgba[:, :, :3]
        logo_a = logo_rgba[:, :, 3]
    else:
        logo_rgb = logo_rgba[:, :, :3]
        logo_a = np.full((h_logo, w_logo), 255, dtype=np.uint8)

    warped_rgb = cv2.warpPerspective(logo_rgb, M, (base_image.shape[1], base_image.shape[0]))
    warped_a = cv2.warpPerspective(logo_a, M, (base_image.shape[1], base_image.shape[0]))

    mask = (warped_a.astype(np.float32) / 255.0)[..., None]
    inv_mask = 1.0 - mask
    base_region = base_image.astype(np.float32)
    comp = mask * warped_rgb.astype(np.float32) + inv_mask * base_region
    base_image[:] = comp.astype(np.uint8)

def _shrink_quad(quad, shrink_ratio=0.9):
    # Inset the quad towards its centroid to avoid bleeding beyond borders
    quad = quad.astype(np.float32)
    cx = np.mean(quad[:, 0])
    cy = np.mean(quad[:, 1])
    center = np.array([cx, cy], dtype=np.float32)
    shrunk = center + (quad - center) * shrink_ratio
    return shrunk.astype(np.float32)

def _is_rectangular(pts):
    # Check right angles via cosine between edges
    def angle_cos(p0, p1, p2):
        d1 = p0 - p1
        d2 = p2 - p1
        denom = (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-6)
        cosv = float(np.dot(d1, d2) / denom)
        return abs(cosv)
    pts = pts.reshape(4, 2)
    csum = 0.0
    for i in range(4):
        csum += angle_cos(pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4])
    return csum / 4.0 < 0.25  # average cosine close to 0 -> near right angles

def detect_plate_quad(base_image, x1, y1, x2, y2):
    x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(base_image.shape[1], int(x2)); y2 = min(base_image.shape[0], int(y2))
    roi = base_image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 160], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)

    edges = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(edges, thr)
    combined = cv2.bitwise_and(combined, mask_white)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    roi_area = roi.shape[0] * roi.shape[1]
    best_rect = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.01 * roi_area or area > 0.9 * roi_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        rect = _order_points_clockwise(approx.reshape(-1, 2).astype(np.float32))
        w = np.linalg.norm(rect[1] - rect[0])
        w2 = np.linalg.norm(rect[2] - rect[3])
        h = np.linalg.norm(rect[3] - rect[0])
        h2 = np.linalg.norm(rect[2] - rect[1])
        mean_w = (w + w2) / 2.0
        mean_h = (h + h2) / 2.0
        if mean_h == 0 or mean_w == 0:
            continue
        ratio = mean_w / mean_h
        if ratio < 1.0 or ratio > 8.0:
            continue
        rect_score = (area / roi_area) * (1.0 - abs(ratio - 3.0) / 7.0)
        if _is_rectangular(rect):
            rect_score += 0.3
        if rect_score > best_score:
            best_score = rect_score
            best_rect = rect

    if best_rect is not None:
        best_rect[:, 0] += x1
        best_rect[:, 1] += y1
        best_rect = _shrink_quad(best_rect, shrink_ratio=1.1) #here 0.98
        return best_rect
    return None

# STEP 3: Load Images 
input_image_path = r"C:\Users\Admin\Downloads\Dataset\New folder\Data\data18.png"
logo_image_path = r"C:\Users\Admin\Downloads\Dataset\New folder\logo.png"
output_image_path = r"C:\Users\Admin\Desktop\SportzInteractive2\output_car_18_updated.jpg"

image = cv2.imread(input_image_path) # Read input image
if image is None:
    raise FileNotFoundError(f"Input image not found at {input_image_path}")

logo = cv2.imread(logo_image_path, cv2.IMREAD_UNCHANGED) # Read logo image (with alpha if available)
if logo is None:
    raise FileNotFoundError(f"Logo image not found at {logo_image_path}")

#STEP 4: Detect Number Plate
results = model.predict(source=image, conf=0.25, verbose=False)  # was 0.5

if not results or results[0].boxes is None or len(results[0].boxes) == 0:
    print("No number plate detected in the image.")
else:
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Safety check to avoid invalid coordinates
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop detected plate region
            plate_region = image[y1:y2, x1:x2]
            cv2.imshow("Extracted Number Plate", plate_region) # to see th extracted number plate

            # (Optional) OCR on plate
            text_results = reader.readtext(plate_region)
            if text_results:
                print("Detected Plate Text:", text_results[0][1])

            # STEP 5: Replace with Logo 
            quad = detect_plate_quad(image, x1, y1, x2, y2)
            if quad is not None:
                place_logo_in_quad(image, logo, quad)
            else:
                place_logo_in_box(image, logo, x1, y1, x2, y2, margin_ratio=0.98) # here  



    #STEP 6: Save Results

    cv2.imshow("Final Output", image)

    cv2.imwrite(output_image_path, image)
    print("Processed image saved as", output_image_path)

    cv2.waitKey(0)  
    cv2.destroyAllWindows()
