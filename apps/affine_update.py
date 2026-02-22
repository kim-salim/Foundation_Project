# -*- coding: utf-8 -*-
# Python 3.8+
# YOLOv5 + ray-plane(Z=plane_z) -> BASE(mm)
# + Affine FB (meas RAW -> true) 학습/적용 + save/load
#
# ✅ 기본 경로:
#   root_dir      : 프로젝트 루트(또는 VISION_ROBOT_DIR)
#   calib(default): {root_dir}/data/calib/camera_intrinsics_1280x720.npz
#   affine(default):{root_dir}/data/calib/affine_fb.npz
#   yolo_repo     : {root_dir}/yolov5
#   yolo_weight   : {root_dir}/weights/best.pt
#   cam           : 2 (V4L2)
#
# keys:
#   1~9 : RAW 캡쳐
#   g   : affine 학습(EMA update)
#   a   : APPLY ON + blend 증가
#   A   : APPLY OFF + blend=0
#   s   : affine 저장(.npz)
#   l   : affine 로드(.npz)
#   c   : 캡쳐 clear
#   r   : reset
#   space : pause YOLO
#   q   : quit

import os
import re
import time
import json
import shutil
import cv2
import threading
import argparse
import numpy as np
import serial
import platform
from glob import glob
from pathlib import Path
import pathlib
import warnings
from datetime import datetime
from serial.tools import list_ports

warnings.filterwarnings("ignore", category=FutureWarning)

# ✅ Windows에서 저장된 torch weights(best.pt) -> Linux 로드 패치
if platform.system().lower() != "windows":
    pathlib.WindowsPath = pathlib.PosixPath
    pathlib.PureWindowsPath = pathlib.PurePosixPath

import torch  # noqa: E402

IS_LINUX = (platform.system().lower() == "linux")
BAUDS_DEFAULT = [115200, 38400]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIB_REL = Path("data") / "calib" / "camera_intrinsics_1280x720.npz"
DEFAULT_CALIB_LEGACY_REL = Path("camera_intrinsics_1280x720.npz")
DEFAULT_AFFINE_REL = Path("data") / "calib" / "affine_fb.npz"
DEFAULT_AFFINE_HISTORY_REL = Path("data") / "calib" / "history" / "affine"


# =========================
# Serial helpers
# =========================
def guess_ports():
    ports = []
    by_id = sorted(glob("/dev/serial/by-id/*"))
    ports.extend(by_id)

    plist = list(list_ports.comports())
    pri = []
    for p in plist:
        dev = p.device
        desc = (p.description or "").upper()
        if ("CH340" in desc) or ("USB" in desc) or ("CP210" in desc) or ("FTDI" in desc):
            pri.append(dev)
    ports.extend(pri)

    for p in plist:
        dev = p.device
        if dev.startswith("/dev/ttyUSB") or dev.startswith("/dev/ttyACM"):
            ports.append(dev)

    seen = set()
    out = []
    for p in ports:
        if p not in seen:
            out.append(p)
            seen.add(p)
    if not out:
        out = [p.device for p in plist]
    return out


def open_serial(port: str, baud: int, timeout=0.15, toggle_dtr_rts=True):
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
        write_timeout=1.0,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )

    if toggle_dtr_rts:
        try:
            ser.setDTR(False)
            ser.setRTS(False)
            time.sleep(0.05)
            ser.setDTR(True)
            ser.setRTS(True)
        except Exception:
            pass

    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    return ser


def ser_write_line(ser: serial.Serial, cmd: str, eol: str):
    ser.write((cmd + eol).encode("utf-8"))
    ser.flush()


def ser_read_until(ser: serial.Serial, end_char=b">", max_wait=0.55):
    t0 = time.time()
    buf = b""
    while time.time() - t0 < max_wait:
        try:
            n = ser.in_waiting
        except Exception:
            n = 0
        if n:
            buf += ser.read(n)
            if end_char in buf:
                break
        else:
            time.sleep(0.01)
    return buf.decode(errors="ignore")


def ser_drain(ser: serial.Serial, max_wait=0.03):
    time.sleep(max_wait)
    try:
        n = ser.in_waiting
        if n:
            ser.read(n)
    except Exception:
        pass


RE_STATUS = re.compile(
    r"Cartesian coordinate\(XYZ RxRyRz\):"
    r"\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)",
    re.IGNORECASE,
)


def parse_state_token(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("<"):
        return s[1:].split(",", 1)[0].strip()
    return "UNKNOWN"


def query_status_pose(ser: serial.Serial, eol: str):
    try:
        ser_drain(ser, 0.01)
        ser_write_line(ser, "?", eol=eol)
        raw = ser_read_until(ser, end_char=b">", max_wait=0.55)

        if not raw.strip():
            alt = "\n" if eol == "\r\n" else "\r\n"
            ser_write_line(ser, "?", eol=alt)
            raw = ser_read_until(ser, end_char=b">", max_wait=0.55)

        state = parse_state_token(raw)
        m = RE_STATUS.search(raw)
        if not m:
            return False, state, None, None, raw

        x, y, z, rx, ry, rz = map(float, m.groups())
        return True, state, (x, y, z), (rx, ry, rz), raw
    except Exception as e:
        return False, "EXC", None, None, str(e)


class SerialWorker(threading.Thread):
    def __init__(self, ser: serial.Serial, eol: str, pose_hz: float):
        super().__init__(daemon=True)
        self.ser = ser
        self.eol = eol
        self.pose_dt = 1.0 / max(1e-6, pose_hz)
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()

        self.last_ok = False
        self.last_state = "UNKNOWN"
        self.last_pose = (0.0, 0.0, 0.0)
        self.last_rpy = (0.0, 0.0, 0.0)
        self.last_raw = ""

    def run(self):
        t_next_pose = 0.0
        while not self.stop_evt.is_set():
            now = time.time()
            if now >= t_next_pose:
                t_next_pose = now + self.pose_dt
                ok_pose, state, xyz, rpy, raw = query_status_pose(self.ser, eol=self.eol)
                with self.lock:
                    self.last_ok = ok_pose
                    self.last_state = state
                    self.last_raw = (raw or "").strip()
                    if ok_pose and xyz is not None:
                        self.last_pose = xyz
                        if rpy is not None:
                            self.last_rpy = rpy
            time.sleep(0.005)

    def snapshot(self):
        with self.lock:
            return {
                "ok": self.last_ok,
                "state": self.last_state,
                "pose": self.last_pose,
                "rpy": self.last_rpy,
                "raw": self.last_raw,
            }

    def stop(self):
        self.stop_evt.set()


# =========================
# Math
# =========================
def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def R_from_rpy_deg(rx, ry, rz, order="zyx"):
    ax = np.deg2rad(float(rx))
    ay = np.deg2rad(float(ry))
    az = np.deg2rad(float(rz))
    m = {"x": Rx(ax), "y": Ry(ay), "z": Rz(az)}
    R = np.eye(3, dtype=np.float64)
    for c in (order or "zyx").lower():
        R = R @ m[c]
    return R


def normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


# =========================
# Pixel -> ray -> plane intersect
# =========================
def pixel_to_ray_cam(u, v, K):
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (float(u) - cx) / fx
    y = (float(v) - cy) / fy
    d = np.array([x, y, 1.0], dtype=np.float64)
    return normalize(d)


def intersect_ray_plane_z(cam_pos_base, ray_dir_base, plane_z):
    dz = float(ray_dir_base[2])
    if abs(dz) < 1e-9:
        return None, dz, None
    s = (float(plane_z) - float(cam_pos_base[2])) / dz
    if s <= 0:
        return None, dz, s
    P = np.asarray(cam_pos_base, dtype=np.float64) + s * np.asarray(ray_dir_base, dtype=np.float64)
    return P, dz, s


# =========================
# Camera / calib
# =========================
def resolve_project_root() -> Path:
    env_root = os.getenv("VISION_ROBOT_DIR", "").strip()
    if env_root:
        return Path(os.path.expandvars(os.path.expanduser(env_root))).resolve()
    return PROJECT_ROOT


def resolve_calib_path(default_folder: str, user_path: str, default_name=None):
    """
    1) user_path(절대/상대)
    2) {root}/data/calib/default_name
    3) {root}/default_name (legacy)
    4) glob fallback
    """
    if user_path:
        p = os.path.expandvars(os.path.expanduser(user_path))
        if os.path.isfile(p):
            return p
        p2 = os.path.join(default_folder, user_path)
        if os.path.isfile(p2):
            return p2
        raise FileNotFoundError(f"--calib 파일을 찾을 수 없음: {user_path}")

    name = (default_name or DEFAULT_CALIB_REL.name)
    root = Path(default_folder)
    primary = root / DEFAULT_CALIB_REL.parent / name
    legacy = root / DEFAULT_CALIB_LEGACY_REL.parent / name
    for p in [primary, legacy]:
        if p.is_file():
            return str(p)

    patterns = [
        root / "data" / "calib" / "camera_intrinsics_*.npz",
        root / "camera_intrinsics_*.npz",
    ]
    for pat in patterns:
        cands = sorted(glob(str(pat)))
        if cands:
            return cands[-1]

    raise FileNotFoundError("캘리브 파일 없음")


def backend_flag(backend: str):
    b = (backend or "auto").lower()
    if b == "auto":
        return cv2.CAP_V4L2 if IS_LINUX else cv2.CAP_DSHOW
    if b == "v4l2":
        return cv2.CAP_V4L2
    if b == "dshow":
        return cv2.CAP_DSHOW
    if b == "msmf":
        return cv2.CAP_MSMF
    if b == "any":
        return 0
    return 0


def parse_cam_arg(cam_arg: str):
    s = str(cam_arg).strip()
    if s.isdigit():
        return int(s)
    return s


def apply_v4l2_low_latency(cap, w, h, fps, fourcc="MJPG"):
    if w > 0 and h > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
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


# =========================
# YOLO
# =========================
def resolve_path(p: str, root_dir: str):
    p = (p or "").strip()
    if not p:
        return ""
    q = Path(os.path.expandvars(os.path.expanduser(p)))
    if q.is_absolute():
        return str(q)
    return str((Path(root_dir) / q).resolve())


def load_yolo(repo_path: str, weight_path: str, conf: float, iou: float, root_dir: str):
    repo_path = resolve_path(repo_path, root_dir)
    weight_path = resolve_path(weight_path, root_dir)

    if not os.path.isdir(repo_path):
        raise FileNotFoundError(f"--yolo_repo 폴더가 유효하지 않음: {repo_path}")
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"--yolo_weight 파일이 유효하지 않음: {weight_path}")

    model = torch.hub.load(repo_path, "custom", path=weight_path, source="local")
    model.conf = float(conf)
    model.iou = float(iou)
    return model


def pick_best_detection(last_dets, target_name=""):
    if not last_dets:
        return None
    cand = last_dets
    if target_name:
        cand = [d for d in last_dets if str(d.get("name", "")) == str(target_name)]
        if not cand:
            return None
    cand.sort(key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    return cand[0]


# =========================
# Affine FB (EMA) + Save/Load
# =========================
class Affine2DCorrector:
    def __init__(self, gain=0.35, ransac_thres_mm=5.0):
        self.gain = float(gain)
        self.ransac_thres_mm = float(ransac_thres_mm)
        self.M = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]], dtype=np.float64)
        self.last_info = "init"
        self.last_ok = False

    def reset(self):
        self.M[:] = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], dtype=np.float64)
        self.last_info = "reset"
        self.last_ok = False

    @staticmethod
    def _collect(meas_pts, true_pts):
        src, dst = [], []
        for m, t in zip(meas_pts, true_pts):
            if m is None or t is None:
                continue
            src.append([float(m[0]), float(m[1])])
            dst.append([float(t[0]), float(t[1])])
        if len(src) < 3:
            return None, None
        return np.array(src, dtype=np.float64), np.array(dst, dtype=np.float64)

    @staticmethod
    def _noncollinear_ok(xy: np.ndarray) -> bool:
        if xy is None or len(xy) < 3:
            return False
        A = xy - xy.mean(axis=0, keepdims=True)
        return np.linalg.matrix_rank(A) >= 2

    def fit_ema(self, meas_pts, true_pts):
        src, dst = self._collect(meas_pts, true_pts)
        if src is None:
            self.last_ok = False
            self.last_info = "FAIL: need >=3 points"
            return False, self.last_info
        if not self._noncollinear_ok(src):
            self.last_ok = False
            self.last_info = "FAIL: degenerate(collinear). include Y-different points."
            return False, self.last_info

        M, inl = cv2.estimateAffine2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.ransac_thres_mm),
            maxIters=5000,
            confidence=0.99,
            refineIters=10
        )
        if M is None:
            self.last_ok = False
            self.last_info = "FAIL: estimateAffine2D failed"
            return False, self.last_info

        g = float(np.clip(self.gain, 0.0, 1.0))
        self.M = (1.0 - g) * self.M + g * M
        inliers = int(inl.sum()) if inl is not None else -1
        self.last_ok = True
        self.last_info = f"OK: n={len(src)}, inliers={inliers}, th={self.ransac_thres_mm:.1f}mm, gain={g:.2f}"
        return True, self.last_info

    def apply(self, x, y):
        x = float(x)
        y = float(y)
        xx = self.M[0, 0] * x + self.M[0, 1] * y + self.M[0, 2]
        yy = self.M[1, 0] * x + self.M[1, 1] * y + self.M[1, 2]
        return float(xx), float(yy)

    def save_npz(self, path: str, true_pts_list, meas_pts_list, meta: dict):
        """
        저장 포맷:
          - M: (2,3)
          - true_pts: (N,3)
          - meas_pts: (N,3)  (미캡쳐는 NaN)
          - meta: json string
        """
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        true_arr = np.array([np.asarray(p, dtype=np.float64) for p in true_pts_list], dtype=np.float64).reshape(-1, 3)
        meas_arr = np.full((len(true_pts_list), 3), np.nan, dtype=np.float64)
        for i, mp in enumerate(meas_pts_list):
            if mp is None:
                continue
            meas_arr[i, :] = np.asarray(mp, dtype=np.float64).reshape(3,)

        meta2 = dict(meta or {})
        meta2.update({
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "gain": float(self.gain),
            "ransac_thres_mm": float(self.ransac_thres_mm),
            "last_ok": bool(self.last_ok),
            "last_info": str(self.last_info),
        })

        np.savez(path, M=self.M.astype(np.float64), true_pts=true_arr, meas_pts=meas_arr,
                 meta=json.dumps(meta2, ensure_ascii=False))

    def load_npz(self, path: str):
        path = str(path)
        d = np.load(path, allow_pickle=False)
        M = d["M"]
        if M.shape != (2, 3):
            raise ValueError(f"Invalid affine M shape: {M.shape}, expected (2,3)")
        self.M = M.astype(np.float64)
        self.last_ok = True
        self.last_info = f"LOADED: {Path(path).name}"

        meta = {}
        if "meta" in d.files:
            try:
                meta = json.loads(d["meta"].item())
            except Exception:
                meta = {}
        return meta


def backup_affine_file(src_path: str, backup_dir: str):
    if not str(backup_dir).strip():
        return None
    src = Path(src_path)
    if not src.is_file():
        return None
    bdir = Path(backup_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = bdir / f"{src.stem}_{ts}{src.suffix}"
    n = 1
    while dst.exists():
        dst = bdir / f"{src.stem}_{ts}_{n:02d}{src.suffix}"
        n += 1
    shutil.copy2(src, dst)
    return str(dst)


# =========================
# UI helpers
# =========================
def draw_panel(img, x, y, w, h, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)


def put_text(img, s, x, y, scale=0.55, color=(0, 255, 255), thick=2):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_toggle_button(img, x, y, label, is_on, blend):
    w, h = 330, 40
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)

    pill_w = 92
    pill_x1 = x + w - pill_w - 8
    pill_y1 = y + 6
    pill_x2 = x + w - 8
    pill_y2 = y + h - 6
    cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2),
                  (0, 160, 0) if is_on else (0, 0, 160), -1)
    cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), (255, 255, 255), 1)

    put_text(img, label, x + 10, y + 26, scale=0.55, color=(255, 255, 255), thick=2)
    put_text(img, "ON" if is_on else "OFF", pill_x1 + 20, y + 26, scale=0.55, color=(255, 255, 255), thick=2)

    bar_x1 = x + 10
    bar_y1 = y + h + 8
    bar_w = w - 20
    bar_h = 10
    cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (80, 80, 80), -1)
    fill_w = int(bar_w * float(np.clip(blend, 0.0, 1.0)))
    cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), (0, 200, 200), -1)
    cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (255, 255, 255), 1)
    put_text(img, f"blend={blend:.2f}  (press 'a' to increase, 'A' to OFF)",
             bar_x1, bar_y1 + 24, scale=0.52, color=(220, 220, 220), thick=2)


# =========================
# Main
# =========================
def main():
    root_path = resolve_project_root()
    root_dir = str(root_path)
    default_affine_file = root_path / DEFAULT_AFFINE_REL
    default_affine_backup_dir = root_path / DEFAULT_AFFINE_HISTORY_REL

    ap = argparse.ArgumentParser()

    # camera / calib
    ap.add_argument("--calib", type=str, default="")  # default: root_dir/data/calib/camera_intrinsics_1280x720.npz
    ap.add_argument("--cam", type=str, default="2")
    ap.add_argument("--backend", type=str, default=("v4l2" if IS_LINUX else "dshow"),
                    choices=["auto", "v4l2", "dshow", "msmf", "any"])
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", type=str, default="MJPG")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=10)

    # yolo
    ap.add_argument("--yolo_repo", type=str, default=str(root_path / "yolov5"))
    ap.add_argument("--yolo_weight", type=str, default=str(root_path / "weights" / "best.pt"))
    ap.add_argument("--conf_thres", type=float, default=0.5)
    ap.add_argument("--iou_thres", type=float, default=0.45)
    ap.add_argument("--target_name", type=str, default="")

    # plane
    ap.add_argument("--plane_z", type=float, default=0.0)

    # serial
    ap.add_argument("--serial", type=str, default="auto")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--eol", type=str, default="\\r\\n")
    ap.add_argument("--pose_hz", type=float, default=10.0)
    ap.add_argument("--toggle_dtr_rts", action="store_true")

    # pose usage
    ap.add_argument("--use_serial_pose_as_camera", action="store_true")
    ap.add_argument("--use_serial_rpy_as_camera", action="store_true")
    ap.add_argument("--rpy_order", type=str, default="zyx")

    # camera position translation
    ap.add_argument("--cam_offset_xyz", type=float, nargs=3, default=[40.0, 0.0, 100.0])

    # fallback
    ap.add_argument("--cam_base_xyz", type=float, nargs=3, default=[0.0, 0.0, 500.0])
    ap.add_argument("--cam_base_rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0])

    # axis fix
    ap.add_argument("--cam_fix_rpy", type=float, nargs=3, default=None)

    # Affine FB
    ap.add_argument("--affine_gain", type=float, default=0.35)
    ap.add_argument("--affine_ransac_thres", type=float, default=5.0)
    ap.add_argument("--apply_step", type=float, default=0.10)
    ap.add_argument("--affine_blend_init", type=float, default=0.0)
    ap.add_argument("--affine_file", type=str, default=str(default_affine_file))
    ap.add_argument("--affine_backup_dir", type=str, default=str(default_affine_backup_dir))

    # True points
    ap.add_argument("--true_p1", type=float, nargs=3, default=[200.0, 50.0, 0.0])
    ap.add_argument("--true_p2", type=float, nargs=3, default=[200.0, 0.0, 0.0])
    ap.add_argument("--true_p3", type=float, nargs=3, default=[200.0, -50.0, 0.0])
    ap.add_argument("--true_p4", type=float, nargs=3, default=[240.0, 0.0, 0.0])
    ap.add_argument("--true_p5", type=float, nargs=3, default=[180.0, 0.0, 0.0])
    ap.add_argument("--true_p6", type=float, nargs=3, default=None)
    ap.add_argument("--true_p7", type=float, nargs=3, default=None)
    ap.add_argument("--true_p8", type=float, nargs=3, default=None)
    ap.add_argument("--true_p9", type=float, nargs=3, default=None)

    args = ap.parse_args()
    eol = "\r\n" if args.eol == "\\r\\n" else ("\n" if args.eol == "\\n" else args.eol)

    if not args.use_serial_pose_as_camera:
        args.use_serial_pose_as_camera = True

    cam_offset_xyz = np.array(args.cam_offset_xyz, dtype=np.float64)
    cam_fb_xyz = np.zeros(3, dtype=np.float64)

    # true pts
    true_pts = []
    for p in [args.true_p1, args.true_p2, args.true_p3, args.true_p4, args.true_p5,
              args.true_p6, args.true_p7, args.true_p8, args.true_p9]:
        if p is None:
            continue
        true_pts.append(np.array(p, dtype=np.float64))
    N_TRUE = len(true_pts)
    meas_pts = [None] * N_TRUE

    # affine
    aff = Affine2DCorrector(gain=args.affine_gain, ransac_thres_mm=args.affine_ransac_thres)

    affine_apply_runtime = False
    affine_blend_runtime = float(np.clip(args.affine_blend_init, 0.0, 1.0))
    apply_step = float(np.clip(args.apply_step, 0.001, 1.0))

    affine_file = resolve_path(args.affine_file, root_dir)
    affine_backup_dir = resolve_path(args.affine_backup_dir, root_dir)
    print("[INFO] root_dir:", root_dir)
    print("[INFO] affine_file:", affine_file)
    print("[INFO] affine_backup_dir:", affine_backup_dir)

    # calib
    calib_path = resolve_calib_path(root_dir, args.calib)
    data = np.load(calib_path)
    K, dist = data["K"], data["dist"]
    print("[INFO] calib:", calib_path)

    # camera open
    be = backend_flag(args.backend)
    cam_src = parse_cam_arg(args.cam)
    cap = cv2.VideoCapture(cam_src, be) if be != 0 else cv2.VideoCapture(cam_src)
    if not cap.isOpened():
        print(f"[-] camera open fail: cam={args.cam}, backend={args.backend}")
        return

    if IS_LINUX:
        apply_v4l2_low_latency(cap, args.w, args.h, args.fps, fourcc=args.fourcc)
    else:
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
    print(f"[INFO] CAP size: {W}x{H}")

    # undistort map
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), args.alpha, (W, H))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (W, H), cv2.CV_16SC2)

    # YOLO
    torch.set_grad_enabled(False)
    model = load_yolo(args.yolo_repo, args.yolo_weight, args.conf_thres, args.iou_thres, root_dir)
    print("[INFO] YOLO loaded")

    # serial worker
    ser = None
    worker = None
    if args.serial.lower() != "none":
        ports_ = guess_ports() if args.serial.lower() == "auto" else [args.serial]
        bauds = [args.baud] + [b for b in BAUDS_DEFAULT if b != args.baud]
        for p in ports_:
            for b in bauds:
                try:
                    ser = open_serial(p, b, timeout=0.15, toggle_dtr_rts=args.toggle_dtr_rts)
                    print(f"[INFO] serial CONNECTED {p}@{b}")
                    break
                except Exception as e:
                    ser = None
                    print(f"[WARN] serial connect fail {p}@{b}: {e}")
            if ser is not None:
                break
        if ser is None:
            print("[WARN] serial connect fail. fallback pose.")
    else:
        print("[INFO] serial disabled")

    if ser is not None:
        worker = SerialWorker(ser, eol=eol, pose_hz=args.pose_hz)
        worker.start()

    # axis fix
    if args.cam_fix_rpy is not None:
        R_cam_fix = R_from_rpy_deg(args.cam_fix_rpy[0], args.cam_fix_rpy[1], args.cam_fix_rpy[2], order="zyx")
        R_fix_src = f"cam_fix_rpy={args.cam_fix_rpy}"
    else:
        R_cam_fix = np.array([
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=np.float64)
        R_fix_src = "default_matrix"

    paused = False
    last_dets = []

    def count_captured():
        return sum(1 for m in meas_pts if m is not None)

    panel_w = 650
    panel_h = min(H - 20, 470)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            view = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            show = view.copy()

            # serial snapshot
            if worker is not None:
                snap = worker.snapshot()
            else:
                snap = {
                    "ok": False,
                    "state": "SERIAL_OFF",
                    "pose": tuple(args.cam_base_xyz),
                    "rpy": tuple(args.cam_base_rpy),
                    "raw": "",
                }

            pose_ok = bool(snap["ok"])
            state = str(snap.get("state", "UNKNOWN"))

            # camera position (translation only)
            if args.use_serial_pose_as_camera and pose_ok:
                ee_pos_base = np.array(snap["pose"], dtype=np.float64)
            else:
                ee_pos_base = np.array(args.cam_base_xyz, dtype=np.float64)
            cam_pos_base = ee_pos_base + cam_offset_xyz + cam_fb_xyz

            # camera rotation for ray only
            if args.use_serial_rpy_as_camera and pose_ok:
                rx, ry, rz = snap["rpy"]
                R_base_tool = R_from_rpy_deg(rx, ry, rz, order=args.rpy_order)
                R_base_cam = R_base_tool @ R_cam_fix
            else:
                R_base_cam = R_cam_fix

            # YOLO inference
            last_dets = []
            if not paused:
                frame_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                df = results.pandas().xyxy[0]

                for _, row in df.iterrows():
                    conf = float(row["confidence"])
                    if conf < float(args.conf_thres):
                        continue

                    cls_name = str(row["name"])
                    xmin, ymin = float(row["xmin"]), float(row["ymin"])
                    xmax, ymax = float(row["xmax"]), float(row["ymax"])
                    u = (xmin + xmax) * 0.5
                    v = (ymin + ymax) * 0.5

                    # draw bbox + center
                    cv2.rectangle(show, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), 2)
                    cv2.circle(show, (int(u), int(v)), 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(show, f"{cls_name} {conf:.2f}", (int(xmin), int(ymin) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    # RAW ray-plane
                    d_cam = pixel_to_ray_cam(u, v, newK)
                    d_base = R_base_cam @ d_cam
                    P_raw, _, _ = intersect_ray_plane_z(cam_pos_base, d_base, args.plane_z)
                    if P_raw is None:
                        continue

                    Xr, Yr, Zr = float(P_raw[0]), float(P_raw[1]), float(P_raw[2])

                    # OUT = blend(RAW, Affine(RAW))
                    Xo, Yo = Xr, Yr
                    if affine_apply_runtime and aff.last_ok:
                        Xa, Ya = aff.apply(Xr, Yr)
                        blend = float(np.clip(affine_blend_runtime, 0.0, 1.0))
                        Xo = (1.0 - blend) * Xr + blend * Xa
                        Yo = (1.0 - blend) * Yr + blend * Ya

                    cv2.putText(show, f"RAW X={Xr:.0f} Y={Yr:.0f} Z={Zr:.0f}",
                                (int(xmin), int(ymax) + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(show, f"OUT X={Xo:.0f} Y={Yo:.0f} Z={Zr:.0f}",
                                (int(xmin), int(ymax) + 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

                    last_dets.append({
                        "name": cls_name,
                        "conf": conf,
                        "P_raw": np.array([Xr, Yr, Zr], dtype=np.float64),
                    })

            # ===== UI PANEL =====
            px, py = 10, 10
            draw_panel(show, px, py, panel_w, panel_h, alpha=0.60)

            draw_toggle_button(
                show,
                px + 15, py + 14,
                "AFFINE APPLY (press 'a' repeatedly)",
                affine_apply_runtime and aff.last_ok,
                affine_blend_runtime if (aff.last_ok and affine_apply_runtime) else 0.0
            )

            put_text(show, "RUNNING" if not paused else "PAUSED",
                     px + 430, py + 38, scale=0.75,
                     color=(0, 255, 0) if not paused else (0, 0, 255), thick=2)

            put_text(show,
                     f"state:{state} | pose_ok:{pose_ok} | serial:{'ON' if worker else 'OFF'} | plane_z={args.plane_z:.1f}",
                     px + 15, py + 94, scale=0.55, color=(255, 255, 255), thick=2)
            put_text(show,
                     f"R_fix={R_fix_src} | use_rpy={bool(args.use_serial_rpy_as_camera)} | rpy_order={args.rpy_order}",
                     px + 15, py + 116, scale=0.55, color=(255, 255, 255), thick=2)
            put_text(show,
                     f"cam_pos = EE + cam_offset (translation only) | offset=({cam_offset_xyz[0]:.0f},{cam_offset_xyz[1]:.0f},{cam_offset_xyz[2]:.0f})",
                     px + 15, py + 138, scale=0.55, color=(255, 255, 255), thick=2)
            put_text(show, f"affine_file = {Path(affine_file).name}  (s=save, l=load)",
                     px + 15, py + 160, scale=0.55, color=(255, 255, 255), thick=2)

            # sequence explanation
            sy = py + 190
            put_text(show, "APPLY SEQUENCE:", px + 15, sy, scale=0.60, color=(255, 255, 255), thick=2)
            sy += 22
            seq = [
                "1) YOLO -> pixel center (u,v)",
                "2) pixel -> ray d_cam (newK)",
                "3) rotate ray only: d_base = (R_base_tool @ R_fix) @ d_cam",
                "4) RAW = intersect(cam_pos, d_base, z=plane_z)",
                "5) CAPTURE(1~N): store RAW only (pre-affine)",
                "6) SOLVE(g): learn affine measXY->trueXY",
                "7) APPLY: press 'a' repeatedly to raise blend toward 1.0",
            ]
            for s in seq:
                put_text(show, s, px + 20, sy, scale=0.52, color=(220, 220, 220), thick=2)
                sy += 18

            sy += 6
            aff_color = (0, 220, 0) if aff.last_ok else (0, 180, 255)
            put_text(show, f"AFFINE SOLVE: {aff.last_info}", px + 15, sy, scale=0.58, color=aff_color, thick=2)
            sy += 22

            cap_cnt = count_captured()
            put_text(show,
                     f"CAPTURE STATUS: {cap_cnt}/{N_TRUE} captured  (keys: 1~{N_TRUE}, g=solve, s=save, l=load)",
                     px + 15, sy, scale=0.58, color=(255, 255, 255), thick=2)
            sy += 22

            for i in range(N_TRUE):
                tp = true_pts[i]
                mp = meas_pts[i]
                k = i + 1
                if mp is None:
                    msg = f"[ ] key {k}: TRUE=({tp[0]:.0f},{tp[1]:.0f},{tp[2]:.0f})  measRAW=None"
                    col = (180, 180, 180)
                else:
                    msg = f"[✓] key {k}: TRUE=({tp[0]:.0f},{tp[1]:.0f},{tp[2]:.0f})  measRAW=({mp[0]:.0f},{mp[1]:.0f},{mp[2]:.0f})"
                    col = (255, 255, 0)
                put_text(show, msg, px + 18, sy, scale=0.52, color=col, thick=2)
                sy += 18
                if sy > py + panel_h - 14:
                    break

            hint = "keys: 1~9 cap | g solve | a apply+ | A off | s save | l load | c clear | r reset | space pause | q quit"
            cv2.putText(show, hint, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("YOLO ray-plane + Affine FB", show)

            key = (cv2.waitKey(1) & 0xFF)

            if key == ord("q"):
                break
            elif key == 32:
                paused = not paused

            elif key == ord("a"):
                if not affine_apply_runtime:
                    affine_apply_runtime = True
                    affine_blend_runtime = apply_step if affine_blend_runtime < 1e-9 else float(
                        np.clip(affine_blend_runtime + apply_step, 0.0, 1.0)
                    )
                else:
                    affine_blend_runtime = float(np.clip(affine_blend_runtime + apply_step, 0.0, 1.0))
                print(f"[APPLY] ON, blend={affine_blend_runtime:.2f} (step={apply_step:.2f})")

            elif key == ord("A"):
                affine_apply_runtime = False
                affine_blend_runtime = 0.0
                print("[APPLY] OFF, blend reset to 0")

            elif key == ord("g"):
                okA, infoA = aff.fit_ema(meas_pts, true_pts)
                print("[AFF] " + infoA)

            elif key == ord("s"):
                if not aff.last_ok:
                    print("[AFF] save skipped: affine not ready (solve with 'g' first)")
                    continue
                meta = {
                    "plane_z": float(args.plane_z),
                    "cam_offset_xyz": [float(x) for x in cam_offset_xyz.tolist()],
                    "calib_path": str(calib_path),
                    "cap_W": int(W),
                    "cap_H": int(H),
                    "R_fix_src": str(R_fix_src),
                    "use_serial_pose_as_camera": bool(args.use_serial_pose_as_camera),
                    "use_serial_rpy_as_camera": bool(args.use_serial_rpy_as_camera),
                    "rpy_order": str(args.rpy_order),
                    "affine_backup_dir": str(affine_backup_dir),
                }
                try:
                    backup_path = backup_affine_file(affine_file, affine_backup_dir)
                    if backup_path is not None:
                        print(f"[AFF] backup -> {backup_path}")
                    aff.save_npz(affine_file, true_pts, meas_pts, meta=meta)
                    print(f"[AFF] saved -> {affine_file}")
                except Exception as e:
                    print(f"[AFF] save FAIL: {e}")

            elif key == ord("l"):
                try:
                    if not os.path.isfile(affine_file):
                        print(f"[AFF] load FAIL: not found: {affine_file}")
                    else:
                        meta_loaded = aff.load_npz(affine_file)
                        print(f"[AFF] loaded <- {affine_file}")
                        if meta_loaded:
                            print("[AFF] meta:", meta_loaded)
                except Exception as e:
                    print(f"[AFF] load FAIL: {e}")

            elif key == ord("c"):
                meas_pts = [None] * N_TRUE
                print("[CAP] cleared meas points.")

            elif key == ord("r"):
                meas_pts = [None] * N_TRUE
                cam_fb_xyz[:] = 0.0
                aff.reset()
                affine_apply_runtime = False
                affine_blend_runtime = 0.0
                print("[RESET] affine=I, apply OFF, blend=0, cam_fb=0, meas cleared.")

            elif key in [ord(str(d)) for d in range(1, 10)]:
                idx = int(chr(key)) - 1
                if idx < 0 or idx >= N_TRUE:
                    print(f"[CAP] key {idx + 1}: ignored (only 1~{N_TRUE} true points configured)")
                    continue

                best = pick_best_detection(last_dets, target_name=args.target_name)
                if best is None:
                    print(f"[CAP] P{idx + 1}: no valid detection (target='{args.target_name}')")
                    continue

                meas_pts[idx] = best["P_raw"].copy()
                print(f"[CAP] measRAW P{idx + 1} <= {best['name']} conf={best['conf']:.2f} : "
                      f"({meas_pts[idx][0]:.2f},{meas_pts[idx][1]:.2f},{meas_pts[idx][2]:.2f})")

    finally:
        try:
            if worker is not None:
                worker.stop()
        except Exception:
            pass
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
