from functions import *

list_H_path = [
    './homography/HC0.npy',
    './homography/HC1.npy',
    './homography/HC2.npy',
    './homography/HC3.npy'
]

list_cam_path = [
    './homo'
]

result, detections = mapping(list_cam_path, list_H_path)
cv2.imshow("Panorama", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("result.jpg", result)

for i, detection in enumerate(detections):
    print(f"Camera ID: {i}")
    for obj in detection:
        print(f"  Object: {obj['class_id']}, Center: {obj['center']}, Color: {obj['color']}, Camera ID: {obj['camera_id']}")
        