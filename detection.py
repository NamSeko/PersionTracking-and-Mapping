from ultralytics import YOLO #type: ignore
model = YOLO("detect_weight/best.pt")

def detect_objects(img, camera_id, color_point=(255, 255, 255), flip=False):
    """
    Thực hiện phát hiện đối tượng trên ảnh.
    
    Args:
        img: Ảnh đầu vào
        model: Mô hình YOLO đã tải

    Returns:
        Kết quả phát hiện
    """
    conf_threshold = 0.7  # Ngưỡng confidence
    result = model.predict(img, agnostic_nms=True, iou=0.4, conf=conf_threshold)[0]
    # Lấy các điểm góc của bounding box
    detection = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:  # Chỉ lấy bounding box của người (class 0)
            continue
        
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        middle_down_x = (x1 + x2) // 2
        middle_down_y = y2
                
        if flip:
            middle_down_x = img.shape[1] - middle_down_x
            middle_down_y = img.shape[0] - middle_down_y
            
        detection.append({
            "bbox": [x1, y1, x2, y2],
            "center": [middle_down_x, middle_down_y],
            "color": color_point,
            "confidence": conf,
            "class_id": 'person',
            "camera_id": camera_id ,
            "obj_id": None
        })
    return detection