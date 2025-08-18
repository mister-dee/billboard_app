import traceback
from PIL import Image
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import requests
from io import BytesIO
import cv2  # For angle detection

# ---------------------------
# Legal Billboard Size Rules (meters)
# ---------------------------
BILLBOARD_RULES = {
    "Tamil Nadu": {
        "Urban": {"max_width_m": 6, "max_height_m": 3},
        "Rural": {"max_width_m": 8, "max_height_m": 4},
        "Highway": {"max_width_m": 10, "max_height_m": 5}
    },
    "Delhi": {
        "Urban": {"max_width_m": 4, "max_height_m": 2},
        "Rural": {"max_width_m": 6, "max_height_m": 3},
        "Highway": {"max_width_m": 8, "max_height_m": 4}
    }
}

# ---------------------------
# Camera Reference Width (meters)
# ---------------------------
CAMERA_REAL_WORLD_WIDTH_M = 10  # adjust based on known FOV & distance

# ---------------------------
# Load YOLOv8 Model
# ---------------------------
local_path = hf_hub_download(
    repo_id="maco018/billboard-detection-Yolo12",
    filename="yolo12m.pt"
)
yolo_model = YOLO(local_path)
print(f"[INFO] Custom YOLOv8 model loaded from: {local_path}")

# ---------------------------
# Helper: Get Billboard Angle
# ---------------------------
def get_billboard_angle(full_image, xmin, ymin, xmax, ymax):
    try:
        # Crop detected region
        crop = np.array(full_image.crop((xmin, ymin, xmax, ymax)))
        if crop.size == 0:
            return None

        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Get largest contour (likely billboard border)
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)  # ((cx, cy), (w, h), angle)

        angle = rect[2]
        if angle < -45:
            angle = 90 + angle

        return round(angle, 2)

    except Exception as e:
        print(f"[WARN] Angle detection failed: {e}")
        return None

# ---------------------------
# Analyze Billboard Function
# ---------------------------
def analyze_billboard(image_input, state, area_type, visualize=False):
    try:
        # Handle URL or PIL Image
        if isinstance(image_input, str):
            original_image = Image.open(BytesIO(requests.get(image_input).content)).convert("RGB")
        else:
            original_image = image_input

        img_width, img_height = original_image.size
        total_image_area = img_width * img_height

        # Resize image for YOLO
        resized_image = np.array(original_image.resize((640, int(640 * img_height / img_width))))

        # YOLO inference
        results = yolo_model(resized_image, conf=0.25)
        detections = results[0].boxes.xyxy.cpu().numpy()

        if detections.shape[0] == 0:
            raise ValueError("No billboards detected by YOLOv8 custom model.")

        # Pick largest detection
        areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
        largest_idx = areas.argmax()
        xmin, ymin, xmax, ymax = detections[largest_idx]

        # Scale back to original image size
        scale_x = img_width / 640
        scale_y = img_height / resized_image.shape[0]
        xmin *= scale_x
        xmax *= scale_x
        ymin *= scale_y
        ymax *= scale_y

        # Calculate billboard pixel dimensions
        billboard_width_px = xmax - xmin
        billboard_height_px = ymax - ymin

        # Convert pixels → meters (approx)
        billboard_width_m = (billboard_width_px / img_width) * CAMERA_REAL_WORLD_WIDTH_M
        billboard_height_m = (billboard_height_px / img_width) * CAMERA_REAL_WORLD_WIDTH_M

        # Calculate area percentage
        largest_area = billboard_width_px * billboard_height_px
        area_percentage = float((largest_area / total_image_area) * 100)
        oversized = bool(area_percentage >= 50.0)

        # Check legal compliance
        rules = BILLBOARD_RULES.get(state, {}).get(area_type, None)
        if rules:
            if billboard_width_m > rules["max_width_m"] or billboard_height_m > rules["max_height_m"]:
                legal_status = "illegal"
                reason = f"Billboard exceeds legal size limit for {state} ({area_type})"
            else:
                legal_status = "legal"
                reason = "Billboard within legal size limits"
        else:
            legal_status = "unknown"
            reason = f"No rules found for {state} - {area_type}"

        # Billboard angle detection
        angle_deg = get_billboard_angle(original_image, xmin, ymin, xmax, ymax)

        # Crop billboard
        cropped_image = original_image.crop((xmin, ymin, xmax, ymax))
        cropped_image.save("cropped_local.jpg")
        cropped_image_url_placeholder = "cropped_local.jpg"

        # Visualization
        if visualize:
            debug_img = results[0].plot()
            debug_img_pil = Image.fromarray(debug_img)
            visualized_path = "visualized_temp.jpg"
            debug_img_pil.save(visualized_path)
            debug_img_pil.show()
        else:
            visualized_path = None

        return {
            "analysis": {
                "is_compliant": legal_status == "legal",
                "legal_status": legal_status,
                "reason": reason,
                "billboard_width_m": billboard_width_m,
                "billboard_height_m": billboard_height_m,
                "billboard_angle_deg": angle_deg
            },
            "croppedImageUrl": cropped_image_url_placeholder,
            "oversized": oversized,
            "billboard_area_percentage": area_percentage,
            "visualized_image": visualized_path
        }

    except Exception as e:
        print(f"[ERROR] AI Service exception: {e}")
        traceback.print_exc()
        return {
            "analysis": {"error": str(e)},
            "croppedImageUrl": None,
            "oversized": False,
            "billboard_area_percentage": 0.0,
            "visualized_image": None
        }