# -*- coding: utf-8 -*-
# Python 3.8+
# pix2xy_map generator using chessboard + camera intrinsics (K, dist) from calibration npz
# Goal: make pix2xy behave SAME as calibration capture pipeline (backend/FourCC/FPS/warmup + undistort)
#
# (ADD)
#   - UNDISTORTED 화면에 "영상 중앙(center)" + "주점(principal point, cx,cy)"를 항상 표시
#   - UNDIST 좌표계 주점은 newK[0,2], newK[1,2] 기준 (중요!)

import os
import cv2
import argparse
import numpy as np
import platform
from pathlib import Path

IS_LINUX = (platform.system().lower() == "linux")

PROJECT_ROOT = Path(
    os.path.expandvars(os.path.expanduser(os.getenv("VISION_ROBOT_DIR", str(Path(__file__).resolve().parents[1]))))
).resolve()
DEFAULT_SAVE_DIR = str(PROJECT_ROOT / "data" / "calib")
DEFAULT_CALIB    = "camera_intrinsics_1280x720.npz"
DEFAULT_OUT_NPZ  = "pix2xy_map.npz"


# -------------------------
# Camera helpers (same style as calibration)
# -------------------------
def backend_flag(backend: str):
    b = (backend or "auto").lower()
    if b == "auto":
        return cv2.CAP_V4L2 if IS_LINUX else cv2.CAP_DSHOW
    if b == "v4l2":  return cv2.CAP_V4L2
    if b == "dshow": return cv2.CAP_DSHOW
    if b == "msmf":  return cv2.CAP_MSMF
    if b == "any":   return 0
    return 0

def parse_cam_arg(cam_arg: str):
    s = str(cam_arg).strip()
    if s.isdigit():
        return int(s)
    return s

def apply_v4l2_low_latency(cap, w, h, fps, fourcc="MJPG"):
    if w > 0 and h > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    if fourcc:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        except Exception:
            pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

def get_fourcc_str(cap):
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])


# -------------------------
# (ADD) Principal point overlay helpers
# -------------------------
def draw_cross(img, x, y, size=10, color=(255, 0, 255), thick=2):
    x, y = int(round(x)), int(round(y))
    cv2.line(img, (x - size, y), (x + size, y), color, thick, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thick, cv2.LINE_AA)

def draw_center_and_pp(img, K_or_none, label_prefix=""):
    """
    Draw image center and principal point on the SAME image space.
    - Center: yellow cross
    - Principal point: magenta cross (if K provided)
    - Also writes dx,dy (PP - center) in pixels.
    """
    H, W = img.shape[:2]
    cx0, cy0 = W * 0.5, H * 0.5

    # center
    draw_cross(img, cx0, cy0, size=12, color=(0, 255, 255), thick=2)
    cv2.putText(img, f"{label_prefix}CENTER", (int(cx0) + 8, int(cy0) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # principal point
    if K_or_none is not None:
        pp_x = float(K_or_none[0, 2])
        pp_y = float(K_or_none[1, 2])
        draw_cross(img, pp_x, pp_y, size=12, color=(255, 0, 255), thick=2)
        cv2.putText(img, f"{label_prefix}PP(cx,cy)", (int(pp_x) + 8, int(pp_y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

        dx = pp_x - cx0
        dy = pp_y - cy0
        cv2.putText(img, f"{label_prefix}PP-center dx={dx:+.1f}px dy={dy:+.1f}px",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, f"{label_prefix}PP: NA",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)


# -------------------------
# Chessboard helpers
# -------------------------
def detect_corners(gray, pattern_size):
    flags = (cv2.CALIB_CB_EXHAUSTIVE |
             cv2.CALIB_CB_ACCURACY |
             cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    if ret:
        return True, corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if ret:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 5e-4)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return ret, corners

def make_board_points_mm(cols, rows, square_mm):
    objp = np.zeros((rows * cols, 2), np.float32)
    for r in range(rows):
        for c in range(cols):
            x = c * square_mm
            y = (rows - 1 - r) * square_mm
            objp[r * cols + c] = (x, y)
    return objp.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    ap.add_argument("--calib", type=str, default=DEFAULT_CALIB)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT_NPZ)

    # camera (match calibration flags)
    ap.add_argument("--cam", type=str, default="2", help="0/1/2... 또는 /dev/video2")
    ap.add_argument("--backend", type=str, default=("v4l2" if IS_LINUX else "dshow"),
                    choices=["auto", "v4l2", "dshow", "msmf", "any"])
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", type=str, default="MJPG")
    ap.add_argument("--warmup", type=int, default=10)

    # undistort (match calibration alpha behavior)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="getOptimalNewCameraMatrix alpha (0=최소 검은영역, 1=FOV 최대)")

    # board
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--square_mm", type=float, default=20.0)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    calib_path = args.calib if os.path.isabs(args.calib) else os.path.join(args.save_dir, args.calib)
    out_path   = args.out   if os.path.isabs(args.out)   else os.path.join(args.save_dir, args.out)

    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"calib not found: {calib_path}")

    data = np.load(calib_path)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)
    calib_size = None
    if "image_size" in data:
        calib_size = tuple(map(int, data["image_size"].reshape(-1).tolist()))  # (W,H)

    # camera open (SAME as calibration)
    be = backend_flag(args.backend)
    cam_src = parse_cam_arg(args.cam)
    cap = cv2.VideoCapture(cam_src, be) if be != 0 else cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        print(f"[-] camera open fail: cam={args.cam}, backend={args.backend}")
        return

    if IS_LINUX:
        apply_v4l2_low_latency(cap, args.w, args.h, args.fps, fourcc=args.fourcc)
    else:
        if args.w > 0 and args.h > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.h))
        cap.set(cv2.CAP_PROP_FPS, int(args.fps))
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    for _ in range(max(0, int(args.warmup))):
        cap.grab()

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_str = get_fourcc_str(cap)
    fps_real = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    print(f"[INFO] CAP real: {W}x{H} fps={fps_real:.1f} fourcc={fourcc_str} backend={args.backend} cam={args.cam}")
    print(f"[INFO] calib file: {calib_path}")
    if calib_size is not None:
        print(f"[INFO] calib image_size: {calib_size[0]}x{calib_size[1]}")
        if (W, H) != calib_size:
            print("[WARN] 캘리브가 만들어진 해상도(image_size)와 현재 캡처 해상도가 다릅니다.")
            print("       가능한 한 calibration과 동일한 w/h/fourcc/backend로 맞추는 걸 권장합니다.")

    # undistort pipeline (SAME style)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), float(args.alpha), (W, H))

    # (ADD) print PP info
    cx0, cy0 = W * 0.5, H * 0.5
    print(f"[INFO] RAW K principal point:  cx={K[0,2]:.2f} cy={K[1,2]:.2f} | dx={K[0,2]-cx0:+.2f}px dy={K[1,2]-cy0:+.2f}px")
    print(f"[INFO] UND newK principal point: cx={newK[0,2]:.2f} cy={newK[1,2]:.2f} | dx={newK[0,2]-cx0:+.2f}px dy={newK[1,2]-cy0:+.2f}px")

    pattern_size = (int(args.cols), int(args.rows))
    board_xy = make_board_points_mm(args.cols, args.rows, args.square_mm)

    print("[INFO] 체커보드를 고정해두고 화면이 안정적이면 's'로 저장")
    print("[INFO] 'q' 종료")
    print(f"[INFO] board: cols={args.cols} rows={args.rows} square_mm={args.square_mm} | uv_space=undistorted | alpha={args.alpha}")

    H_inv_saved = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        und = cv2.undistort(frame, K, dist, None, newK)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        ok_cb, corners = detect_corners(gray, pattern_size)
        show = und.copy()

        # (ADD) draw center + principal point on UNDIST image space (newK)
        draw_center_and_pp(show, newK, label_prefix="UND ")

        if ok_cb:
            img_pts = corners.reshape(-1, 2).astype(np.float32)  # undistorted 좌표
            Hm, _ = cv2.findHomography(board_xy, img_pts)
            if Hm is not None:
                H_inv_saved = np.linalg.inv(Hm)
                cv2.drawChessboardCorners(show, pattern_size, corners, ok_cb)
                cv2.putText(show, "DETECTED (press 's' to save)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            else:
                cv2.putText(show, "Homography failed", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            cv2.putText(show, "No board", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("make_pix2xy_map (UNDISTORTED)", show)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            if H_inv_saved is None:
                print("[WARN] 아직 보드가 안정적으로 잡히지 않음")
                continue

            np.savez(
                out_path,
                H_inv=H_inv_saved.astype(np.float64),
                square_mm=float(args.square_mm),
                board_cols=int(args.cols),
                board_rows=int(args.rows),
                uv_space="undistorted",
                image_size=np.array([W, H], dtype=np.int32),
                alpha=float(args.alpha),
                fourcc=str(fourcc_str),
                backend=str(args.backend),
                cam=str(args.cam),
                # (ADD) save newK too for downstream consistency/debug
                newK=newK.astype(np.float64),
            )
            print(f"[OK] pix2xy_map saved -> {out_path}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
