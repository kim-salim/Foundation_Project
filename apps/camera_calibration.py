# -*- coding: utf-8 -*-
# Python 3.8+
# Camera intrinsics calibration (K, dist) using chessboard images from live camera
# Output: camera_intrinsics_1280x720.npz saved to SAVE_DIR
#
# Usage examples:
#   python3 calib_camera_intrinsics.py --cam 0
#   python3 calib_camera_intrinsics.py --cam /dev/video2 --backend v4l2
#   python3 calib_camera_intrinsics.py --cols 7 --rows 10 --square_mm 20 --w 1280 --h 720
#
# Controls:
#   c : capture current frame (if board detected)
#   u : toggle undistort preview after calibration (only after calib)
#   r : reset collected samples
#   q : quit
#
# (ADD)
#   - Always draw IMAGE CENTER and PRINCIPAL POINT (cx,cy)
#   - On combo (RAW|UNDIST), draw PP for RAW using K and for UNDIST using newK
#   - Also display dx,dy = (PP - center) in pixels

import os
import cv2
import time
import argparse
import numpy as np
import platform
from pathlib import Path

IS_LINUX = (platform.system().lower() == "linux")

PROJECT_ROOT = Path(
    os.path.expandvars(os.path.expanduser(os.getenv("VISION_ROBOT_DIR", str(Path(__file__).resolve().parents[1]))))
).resolve()
DEFAULT_SAVE_DIR = str(PROJECT_ROOT / "data" / "calib")


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


def detect_corners(gray, pattern_size):
    # SB가 가능하면 SB가 훨씬 잘 잡힘
    flags = (cv2.CALIB_CB_EXHAUSTIVE |
             cv2.CALIB_CB_ACCURACY |
             cv2.CALIB_CB_NORMALIZE_IMAGE)
    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
    if ret:
        return True, corners

    # fallback (일반)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)
    if ret:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return ret, corners


def make_obj_points(cols, rows, square_mm):
    # (0,0,0)부터 보드 평면상의 3D 점들 (단위 mm)
    objp = np.zeros((rows * cols, 3), np.float32)
    for r in range(rows):
        for c in range(cols):
            x = c * square_mm
            y = (rows - 1 - r) * square_mm
            objp[r * cols + c] = (x, y, 0.0)
    return objp


def reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    # 평균 재투영 오차 계산
    tot_err = 0.0
    tot_n = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        gt = imgpoints[i].reshape(-1, 2)
        err = np.linalg.norm(proj - gt, axis=1).mean()
        tot_err += err
        tot_n += 1
    return (tot_err / max(1, tot_n))


# ===== (ADD) principal point overlay helpers =====
def draw_cross(img, x, y, size=10, color=(255, 0, 255), thick=2):
    x, y = int(round(x)), int(round(y))
    cv2.line(img, (x - size, y), (x + size, y), color, thick, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thick, cv2.LINE_AA)


def draw_center_and_pp(img, K_or_none, label_prefix="", origin_offset=(0, 0)):
    """
    Draw image center and principal point on image.
    - Center: yellow cross
    - Principal point: magenta cross
    - Print dx,dy = PP - Center

    origin_offset: used when drawing on stitched canvas (e.g., right side of combo)
    """
    H, W = img.shape[:2]
    ox, oy = origin_offset

    # image center
    cx0, cy0 = W * 0.5, H * 0.5
    draw_cross(img, cx0 + ox, cy0 + oy, size=12, color=(0, 255, 255), thick=2)  # yellow
    cv2.putText(img, f"{label_prefix}CENTER", (int(cx0 + ox) + 8, int(cy0 + oy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # principal point
    if K_or_none is not None:
        pp_x = float(K_or_none[0, 2])
        pp_y = float(K_or_none[1, 2])
        draw_cross(img, pp_x + ox, pp_y + oy, size=12, color=(255, 0, 255), thick=2)  # magenta
        cv2.putText(img, f"{label_prefix}PP(cx,cy)", (int(pp_x + ox) + 8, int(pp_y + oy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

        dx = pp_x - cx0
        dy = pp_y - cy0
        cv2.putText(img, f"{label_prefix}PP-center dx={dx:+.1f}px dy={dy:+.1f}px",
                    (20 + ox, 110 + oy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, f"{label_prefix}PP: NA (not calibrated yet)",
                    (20 + ox, 110 + oy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    ap.add_argument("--out", type=str, default="camera_intrinsics_1280x720.npz")

    # camera
    ap.add_argument("--cam", type=str, default="2", help="0/1/2... 또는 /dev/video2")
    ap.add_argument("--backend", type=str, default=("v4l2" if IS_LINUX else "dshow"),
                    choices=["auto", "v4l2", "dshow", "msmf", "any"])
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", type=str, default="MJPG")
    ap.add_argument("--warmup", type=int, default=10)

    # board
    ap.add_argument("--cols", type=int, default=5, help="내부 코너 cols")
    ap.add_argument("--rows", type=int, default=8, help="내부 코너 rows")
    ap.add_argument("--square_mm", type=float, default=20.0)

    # capture
    ap.add_argument("--min_samples", type=int, default=15, help="이 이상 모이면 캘리브 권장")
    ap.add_argument("--max_samples", type=int, default=40)
    ap.add_argument("--cooldown", type=float, default=0.25, help="연속 캡처 방지(초)")

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.out)

    pattern_size = (int(args.cols), int(args.rows))
    objp = make_obj_points(args.cols, args.rows, args.square_mm)

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
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    for _ in range(max(0, int(args.warmup))):
        cap.grab()

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] capture size: {W}x{H}")

    objpoints = []
    imgpoints = []

    last_capture_t = 0.0
    calibrated = False
    K = None
    dist = None
    newK = None
    undist_on = False

    print("\n[INFO] Controls:")
    print("  c : capture (when board detected)")
    print("  r : reset samples")
    print("  u : toggle undistort preview (after calibration)")
    print("  q : quit\n")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = detect_corners(gray, pattern_size)

        show = frame.copy()

        if found:
            cv2.drawChessboardCorners(show, pattern_size, corners, found)

        # overlay info
        n = len(imgpoints)
        msg1 = f"samples: {n}/{args.min_samples} (max {args.max_samples})"
        msg2 = "DETECTED" if found else "NO BOARD"
        cv2.putText(show, msg1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2, cv2.LINE_AA)
        cv2.putText(show, msg2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if found else (0,0,255), 2, cv2.LINE_AA)

        # ===== (ADD) draw center + principal point on single preview =====
        # calibrated 전에는 K가 없으므로 CENTER만 확실히 찍고, PP는 NA 표시
        draw_center_and_pp(show, K if calibrated else None, label_prefix="RAW ")

        if calibrated and K is not None and dist is not None and undist_on:
            # 캘리브 된 이후 undistort 비교 보기
            if newK is None:
                newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0.0, (W, H))
            und = cv2.undistort(frame, K, dist, None, newK)
            combo = np.hstack([frame, und])

            cv2.putText(combo, "RAW", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(combo, "UNDIST", (W + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)

            # ===== (ADD) draw center + PP for each half =====
            # left half: RAW uses K
            draw_center_and_pp(combo, K, label_prefix="RAW ", origin_offset=(0, 0))
            # right half: UNDIST uses newK, so x offset +W
            draw_center_and_pp(combo, newK, label_prefix="UND ", origin_offset=(W, 0))

            cv2.imshow("camera_calibration (raw | undist)", combo)
        else:
            cv2.imshow("camera_calibration", show)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            objpoints.clear()
            imgpoints.clear()
            calibrated = False
            K = None
            dist = None
            newK = None
            undist_on = False
            print("[INFO] reset samples")

        elif key == ord("u"):
            if calibrated:
                undist_on = not undist_on
                print(f"[INFO] undist preview: {undist_on}")
            else:
                print("[WARN] not calibrated yet (collect samples then calibrate)")

        elif key == ord("c"):
            now = time.time()
            if (now - last_capture_t) < float(args.cooldown):
                continue
            last_capture_t = now

            if not found:
                print("[WARN] capture skipped: board not detected")
                continue

            if len(imgpoints) >= int(args.max_samples):
                print("[WARN] already reached max_samples")
                continue

            # 저장 (corners는 (N,1,2) 형태)
            objpoints.append(objp.copy())
            imgpoints.append(corners.copy())
            print(f"[OK] captured #{len(imgpoints)}")

            # 충분히 모이면 자동 캘리브
            if len(imgpoints) >= int(args.min_samples):
                print("[INFO] calibrating...")

                flags = 0
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

                ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, (W, H), None, None, flags=flags, criteria=criteria
                )

                if not ret:
                    print("[ERR] calibrateCamera failed")
                    calibrated = False
                    continue

                err = reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)
                print("[INFO] K:\n", K)
                print("[INFO] dist:", dist.ravel())
                print(f"[INFO] mean reprojection error: {err:.4f} px")

                # compute newK now (so we can store it too if you want)
                newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0.0, (W, H))

                np.savez(
                    out_path,
                    K=K.astype(np.float64),
                    dist=dist.astype(np.float64),
                    newK=newK.astype(np.float64),  # (ADD) store newK for consistency with other scripts
                    image_size=np.array([W, H], dtype=np.int32),
                    board_cols=int(args.cols),
                    board_rows=int(args.rows),
                    square_mm=float(args.square_mm),
                    reproj_error_px=float(err),
                )
                print(f"[OK] saved -> {out_path}")

                # print PP shift
                cx0, cy0 = W * 0.5, H * 0.5
                dx = float(K[0, 2]) - cx0
                dy = float(K[1, 2]) - cy0
                print(f"[INFO] principal point (K): cx={K[0,2]:.2f}, cy={K[1,2]:.2f} | dx={dx:+.2f}px dy={dy:+.2f}px")
                dx2 = float(newK[0, 2]) - cx0
                dy2 = float(newK[1, 2]) - cy0
                print(f"[INFO] principal point (newK): cx={newK[0,2]:.2f}, cy={newK[1,2]:.2f} | dx={dx2:+.2f}px dy={dy2:+.2f}px")

                calibrated = True
                undist_on = True  # 저장 후 바로 비교 보기 ON

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
