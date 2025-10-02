import cv2
import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from PIL import Image

from ultralytics import YOLO #type: ignore

transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PersonReIDModel(nn.Module):
    def __init__(self, img_feature_dim=128, his_feature_dim=128, n_split=4, dropout=0.2):
        super(PersonReIDModel, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(backbone.fc.in_features, img_feature_dim)
        self.cnn_encoder = backbone
        self.n_split = n_split
        self.dropout = dropout

        self.histogram_encoder = nn.Sequential(
            nn.Linear(self.n_split*self.n_split*64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, his_feature_dim)
        )
        
        feature_dim = img_feature_dim + his_feature_dim
        self.fc = nn.Sequential(
            nn.Linear(feature_dim*4, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        
    def forward_once(self, img):
        img_feat = self.cnn_encoder(img)
        hist = self.compute_histogram(img) # (B, n_split*n_split, 256)
        hist = hist.view(hist.size(0), -1)
        hist_feat = self.histogram_encoder(hist)
        out = torch.cat((img_feat, hist_feat), dim=1) # (B, feature_dim)
        return out
        
    def forward(self, img1, img2, return_features=False):
        feat1 = self.forward_once(img1) # (B, feature_dim)
        feat2 = self.forward_once(img2) # (B, feature_dim)
        
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        diff = torch.abs(feat1 - feat2) # (B, feature_dim)
        prod = feat1 * feat2 # (B, feature_dim)
        combined = torch.cat((feat1, feat2, diff, prod), dim=1) # (B, feature_dim*4)
        out = self.fc(combined) # (B, 1)
        if return_features:
            return out, feat1, feat2
        return out        

    def compute_histogram(self, img_tensor, bins=64):
        gray_img = img_tensor.mean(dim=1)
        
        B, H, W = gray_img.shape
        step_h, step_w = H // self.n_split, W // self.n_split

        # List cho batch
        batch_hist = []

        for b in range(B):
            img = gray_img[b]
            hist_blocks = []
            for i in range(self.n_split):
                for j in range(self.n_split):
                    patch = img[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w]
                    patch = (patch * 255).clamp(0, 255)
                    h_patch = torch.histc(patch, bins=bins, min=0, max=255)
                    h_patch = h_patch / (h_patch.sum() + 1e-6)
                    hist_blocks.append(h_patch)
            hist_blocks = torch.stack(hist_blocks, dim=0) # (n_split*n_split, bins)
            batch_hist.append(hist_blocks)
        batch_hist = torch.stack(batch_hist, dim=0) # (B, n_split*n_split, bins)
        return batch_hist

def Track_Cam(frame, model, model_reid, person_data, cam_id=None):
    detection = []
    results = model.track(frame, persist=True, conf=0.8, iou=0.5, classes=[0], agnostic_nms=True, tracker='./reid_module/tracker/botsort.yaml', verbose=False)
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        _id = int(result.id[0]) if result.id is not None else -1
        person_img = frame[y1:y2, x1:x2]
        person_img_png = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        person_img_tensor = transform(person_img_png).unsqueeze(0).to(device)
        if _id != -1:
            best_id = None
            is_now_id = False
            for exitting_id, img in enumerate(person_data):
                if img is None:
                    continue
                img = img.to(device)
                model_reid.to(device)
                model_reid.eval()
                with torch.no_grad():
                    sim = model_reid(person_img_tensor, img)
                    is_now_id = True if sim.item() >= 0.09 else False
                if is_now_id:
                    best_id = exitting_id
                    break
            if best_id is not None:
                _id = best_id
            detection.append({
                'id': _id,
                'center': ((x1 + x2) / 2, y2),  # Điểm dưới cùng ở giữa bbox
                'cam_id': cam_id
            })
    return detection

def create_data_id_person(path):
    img_person = os.listdir(path)
    persons_id = []
    for img in img_person:
        person_img = Image.open(os.path.join(path, img)).convert('RGB')
        person_img = transform(person_img).unsqueeze(0).to(device)
        persons_id.append(person_img)
    return persons_id

# Vẽ điểm lên ảnh
def plot_point(color_list, detection, result):
    for point in detection:
        points = np.array(point['center'], dtype=np.float32)
        color = color_list[int(point['id'])]
        cv2.circle(result, (int(points[0]), int(points[1])), 5, color, -1)
    return result

# Chuyển đổi tọa độ điểm
def transform_points(detection, T, H):
    for point in detection:
        points = np.array(point['center'], dtype=np.float32)
        points = points.reshape(-1, 1, 2)
        points = cv2.perspectiveTransform(points, T @ H)
        points = points.reshape(-1, 2)[0]
        point['center'] = points
    return detection

# Hàm ánh xạ tọa độ
def transform_points_mapping(detection, min_x_crop, min_y_crop):
    for point in detection:
        point['center'] -= np.array([min_x_crop, min_y_crop], dtype=np.float32)
    return detection

def filter_point(detections, threshold):
    objects = []
    for detection in detections:
        for obj in detection:
            objects.append({
                'id': obj.get('id'),
                "center": obj.get('center'),
                "cam_id": obj.get('cam_id')
            })

    filtered_objects = []
    removed = set()  # lưu index các điểm bị loại

    for i in range(len(objects)):
        if i in removed:
            continue
        keep = True
        for j in range(i + 1, len(objects)):
            if j in removed:
                continue
            if objects[i]["cam_id"] != objects[j]["cam_id"]:
                # Tính khoảng cách Euclidean
                x1, y1 = objects[i]["center"]
                x2, y2 = objects[j]["center"]
                dist = math.dist((x1, y1), (x2, y2))
                if dist < threshold:
                    # Nếu 2 điểm quá gần -> chỉ giữ i, loại j
                    removed.add(j)
        filtered_objects.append(objects[i])
        
    return filtered_objects

# Caculate min_max crop
def caculate_crop(width, height):
    list_H_path = [
        'homography/HC0.npy',
        'homography/HC1.npy',
        'homography/HC2.npy',
        'homography/HC3.npy'
    ]
    list_cam_path = [
        'samples/images/cam0.jpg',
        'samples/images/cam1.jpg',
        'samples/images/cam2.jpg',
        'samples/images/cam3.jpg'
    ]
    corners_map = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    list_H = [np.load(path) for path in list_H_path]
    
    wrapeds = []
    for cam_path, H in zip(list_cam_path, list_H):
        img = cv2.imread(cam_path)
        corners = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        wrapeds.append(warped_corners)
        
    all_corners = np.concatenate([corners_map] + wrapeds, axis=0)
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())

    T = np.array([
        [1, 0, -min(0, x_min)],
        [0, 1, -min(0, y_min)],
        [0, 0, 1]
    ], dtype=np.float32)
    
    min_x_crop, min_y_crop = 0, 0
    min_cropped = cv2.perspectiveTransform(np.array([[[x_min, y_min]]], dtype=np.float32), T)
    min_x_crop, min_y_crop = min_cropped[0][0]
    
    return min_x_crop, min_y_crop, T, list_H

# Hàm ánh xạ
def mapping(list_cam_path):
    # Tạo màu ngẫu nhiên cho các ID
    color_list = random.choices(range(256), k=3*1000)
    color_list = np.array(color_list).reshape(-1, 3).tolist()
    
    map_img = cv2.imread('samples/ref/map.jpg')
    height, width = map_img.shape[:2]    
    print("Calculate crop...")
    min_x_crop, min_y_crop, T, list_H = caculate_crop(width, height)
    print("Done calculate crop")
    cap0 = cv2.VideoCapture(list_cam_path[0])
    cap1 = cv2.VideoCapture(list_cam_path[1])
    cap2 = cv2.VideoCapture(list_cam_path[2])
    cap3 = cv2.VideoCapture(list_cam_path[3])
    
    fps = int(cap0.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./mapping_output.avi', fourcc, fps, (int(width), int(height)))
    
    model_reid = PersonReIDModel(img_feature_dim=256, his_feature_dim=128, n_split=8, dropout=0.33)
    model_reid.load_state_dict(torch.load('./reid_module/6p_4c/models/best.pth'))
    
    model_track_c0 = YOLO('./detection_module/detection_weight/best.pt')
    model_track_c1 = YOLO('./detection_module/detection_weight/best.pt')
    model_track_c2 = YOLO('./detection_module/detection_weight/best.pt')
    model_track_c3 = YOLO('./detection_module/detection_weight/best.pt')
    
    person_data = create_data_id_person('./reid_module/id_person/6p_4c')
    
    print("Start mapping...")
    frame_idx = 0
    while cap0.isOpened() and cap1.isOpened() and cap2.isOpened() and cap3.isOpened():
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        
        if not (ret0 and ret1 and ret2 and ret3):
            break
        if frame_idx % (fps*0.2) == 0:
            detections = []
            detection1 = Track_Cam(frame0, model_track_c0, model_reid, person_data, cam_id=0)
            detection1 = transform_points(detection1, T, list_H[0])
            detection1 = transform_points_mapping(detection1, min_x_crop, min_y_crop)
            detections.append(detection1)
            
            detection2 = Track_Cam(frame1, model_track_c1, model_reid, person_data, cam_id=1)
            detection2 = transform_points(detection2, T, list_H[1])
            detection2 = transform_points_mapping(detection2, min_x_crop, min_y_crop)
            detections.append(detection2)
            
            detection3 = Track_Cam(frame2, model_track_c2, model_reid, person_data, cam_id=2)
            detection3 = transform_points(detection3, T, list_H[2])
            detection3 = transform_points_mapping(detection3, min_x_crop, min_y_crop)
            detections.append(detection3)
            
            detection4 = Track_Cam(frame3, model_track_c3, model_reid, person_data, cam_id=3)
            detection4 = transform_points(detection4, T, list_H[3])
            detection4 = transform_points_mapping(detection4, min_x_crop, min_y_crop)
            detections.append(detection4)
            
            detection = filter_point(detections, threshold=90)
            crop = plot_point(color_list, detection, map_img.copy())
            out.write(crop)
        frame_idx += 1
    cap0.release()
    cap1.release()
    cap2.release()
    cap3.release()
    out.release()
    print("Mapping done. Output saved to mapping_output.avi")