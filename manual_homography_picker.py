import cv2
import numpy as np
import os
import glob
import argparse
from typing import List, Tuple

"""
Interactive manual homography picker
- Choose or auto-pick a reference image
- For each target image, click corresponding points on REF and TARGET
- Compute H (target -> ref), save to results, and save a warped preview

Controls:
- Left click: add a point
- Right click: remove last point
- r: reset points for current pair
- ENTER/SPACE: compute homography (need >= 4 pairs)
- n: skip this target and go next
- q or ESC: quit

Notes:
- Images are displayed scaled for convenience; coordinates are mapped back to original size when computing H.
"""


def list_images(patterns: List[str]) -> List[str]:
    paths: List[str] = []
    for p in patterns:
        paths.extend(glob.glob(p))
    paths = [p for p in paths if os.path.isfile(p) and p.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return sorted(paths)


class ClickCollector:
    def __init__(self, win_name: str, scale: float):
        self.win = win_name
        self.scale = scale
        self.points: List[Tuple[int, int]] = []
        self._img_disp = None
        self._img_base = None

    def set_image(self, img):
        self._img_base = img
        disp = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self._img_disp = disp.copy()
        cv2.imshow(self.win, self._img_disp)

    def draw_points(self):
        if self._img_disp is None:
            return
        disp = self._img_disp.copy()
        for i, (x, y) in enumerate(self.points):
            cv2.circle(disp, (int(x*self.scale), int(y*self.scale)), 4, (0, 255, 0), -1)
            cv2.putText(disp, str(i+1), (int(x*self.scale)+6, int(y*self.scale)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow(self.win, disp)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = int(round(x / self.scale)), int(round(y / self.scale))
            self.points.append((ox, oy))
            self.draw_points()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.draw_points()

    def reset(self):
        self.points.clear()
        self.draw_points()


def compute_and_save_h(ref_img, tgt_img, ref_pts, tgt_pts, out_dir, ref_name, tgt_name, suffix: str):
    ref_pts_np = np.float32(ref_pts).reshape(-1, 1, 2)
    tgt_pts_np = np.float32(tgt_pts).reshape(-1, 1, 2)

    method = 0 if len(ref_pts) <= 4 else cv2.RANSAC
    H, mask = cv2.findHomography(tgt_pts_np, ref_pts_np, method, ransacReprojThreshold=5.0)  # target -> ref
    if H is None:
        return None

    h, w = ref_img.shape[:2]
    warped = cv2.warpPerspective(tgt_img, H, (w, h))

    os.makedirs(out_dir, exist_ok=True)
    tgt_base = os.path.splitext(tgt_name)[0]
    H_path = os.path.join(out_dir, f"H_{tgt_base}_{suffix}.npy")
    np.save(H_path, H)

    warped_path = os.path.join(out_dir, f"warped_{tgt_base}_{suffix}.jpg")
    cv2.imwrite(warped_path, warped)

    return {"H_path": H_path, "warped_path": warped_path}


def main():
    parser = argparse.ArgumentParser(description="Các lựa chọn cho việc chỉnh sửa ảnh")
    parser.add_argument('--ref', type=str, default=None, default=['samples/ref/map.jpg'], help='Đường dẫn tới reference image')
    parser.add_argument('--images', type=str, nargs='*', default=['samples/images/*.jpeg'], help='Thư mục hoặc mẫu ảnh để xử lý')
    parser.add_argument('--out', type=str, default='homography', help='Thư mục lưu kết quả H và ảnh đã biến đổi')
    parser.add_argument('--suffix', type=str, default='to_ref', help="Hậu tố cho tên tệp đầu ra, e.g., 'to_ref' -> H_<img>_to_ref.npy")
    parser.add_argument('--use-ref-name', action='store_true', help="Sử dụng tên tham chiếu cho hậu tố, i.e., 'to_<refbasename>'")
    parser.add_argument('--scale', type=float, default=1.2, help='Tỷ lệ màn hình hiển thị (0.2~1.0)')
    args = parser.parse_args()

    all_images = list_images(args.images)
    if not all_images:
        print('Không tìm thấy tệp images. Vui lòng kiểm tra --images patterns.')
        return

    if args.ref is not None:
        ref_path = args.ref
        if not os.path.isfile(ref_path):
            print(f'Ảnh Reference không tồn tại: {ref_path}')
            return
        targets = [p for p in all_images if os.path.abspath(p) != os.path.abspath(ref_path)]
    else:
        ref_path = all_images[0]
        targets = all_images[1:]

    print(f'Reference: {ref_path}')
    if not targets:
        print('Không có ảnh target để xử lý.')
        return

    ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    if ref_img is None:
        print(f'Không thể load reference image: {ref_path}')
        return

    ref_name = os.path.basename(ref_path)
    ref_base = os.path.splitext(ref_name)[0]
    suffix = f"to_{ref_base}" if args.use_ref_name else args.suffix

    ref_win = f"REF - {ref_name}"
    tgt_win = f"TARGET"
    cv2.namedWindow(ref_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(tgt_win, cv2.WINDOW_NORMAL)
    ref_collector = ClickCollector(ref_win, args.scale)
    tgt_collector = ClickCollector(tgt_win, args.scale)
    cv2.setMouseCallback(ref_win, ref_collector.on_mouse)
    cv2.setMouseCallback(tgt_win, tgt_collector.on_mouse)

    ref_collector.set_image(ref_img)

    for tgt_path in targets:
        tgt_img = cv2.imread(tgt_path, cv2.IMREAD_COLOR)
        if tgt_img is None:
            print(f'Ảnh không tồn tại: {tgt_path}')
            continue
        tgt_name = os.path.basename(tgt_path)
        print(f"\nTarget: {tgt_name}")

        # Reset points for new pair
        ref_collector.reset()
        tgt_collector.reset()

        tgt_collector.set_image(tgt_img)

        # Instruction overlay
        def draw_instructions():
            overlay_ref = ref_collector._img_disp.copy()
            overlay_tgt = tgt_collector._img_disp.copy()
            cv2.imshow(ref_win, overlay_ref)
            cv2.imshow(tgt_win, overlay_tgt)

        draw_instructions()

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 32):  # ENTER or SPACE
                if len(ref_collector.points) >= 4 and len(tgt_collector.points) == len(ref_collector.points):
                    res = compute_and_save_h(ref_img, tgt_img, ref_collector.points, tgt_collector.points, args.out, ref_name, tgt_name, suffix)
                    if res is None:
                        print('Không thể tính homography. Thử lại với điểm tốt hơn.')
                    else:
                        print(f"Đã lưu H -> {res['H_path']}")
                        print(f"Đã lưu wrapped_image -> {res['warped_path']}")
                    break
                else:
                    print('Cần >= 4 cặp điểm và số lượng điểm trên cả hai ảnh phải bằng nhau.')
            elif key in (ord('r'), ord('R')):
                ref_collector.reset()
                tgt_collector.reset()
                draw_instructions()
            elif key in (ord('n'), ord('N')):
                print('Bỏ qua ảnh này và chuyển sang ảnh tiếp theo.')
                break
            elif key in (ord('q'), ord('Q'), 27):
                print('Quit.')
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print('Xong!!!')


if __name__ == '__main__':
    main()
