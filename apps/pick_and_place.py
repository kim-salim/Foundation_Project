# -*- coding: utf-8 -*-
# Python 3.8+
# Linux: YOLOv5 + ray-plane -> RAW base(mm) -> affine_fb.npz 적용
# + 정리모드(색상별 자동 정리/스택)
# + ✅ 시작/마무리 고정 포즈: (X=189, Y=0, Z=230)

import os, re, time, math, threading, platform
from glob import glob
from typing import Tuple, List, Optional, Dict
import pathlib

if platform.system().lower() != "windows":
    pathlib.WindowsPath = pathlib.PosixPath
    pathlib.PureWindowsPath = pathlib.PurePosixPath

import cv2
import numpy as np
import torch
import serial
from serial.tools import list_ports

# =========================
# 프로젝트 경로
# =========================
ROOT_DIR = os.path.abspath(os.path.expandvars(os.path.expanduser(
    os.getenv("VISION_ROBOT_DIR", str(pathlib.Path(__file__).resolve().parents[1]))
)))


def _first_existing(*paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return paths[0]


CALIB_PATH = _first_existing(
    os.path.join(ROOT_DIR, "data", "calib", "camera_intrinsics_1280x720.npz"),
    os.path.join(ROOT_DIR, "camera_intrinsics_1280x720.npz"),
)
AFFINE_PATH = _first_existing(
    os.path.join(ROOT_DIR, "data", "calib", "affine_fb.npz"),
    os.path.join(ROOT_DIR, "affine_fb.npz"),
)
YOLO_REPO   = os.path.join(ROOT_DIR, "yolov5")
YOLO_WEIGHT = os.path.join(ROOT_DIR, "weights", "best.pt")

IS_LINUX = (platform.system().lower() == "linux")

# =========================
# 작업 파라미터
# =========================
CONF_THRES = 0.5
IOU_THRES  = 0.45

PLANE_Z = 0.0
CAM_OFFSET_XYZ = np.array([40.0, 0.0, 100.0], dtype=np.float64)

R_CAM_FIX = np.array([
    [ 0.0, -1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0]
], dtype=np.float64)

# 카메라
CAM_ID  = 2
W, H = 1280, 720
FPS = 30
FOURCC = "MJPG"
ALPHA = 0.0
WARMUP = 10

# =========================
# Pick & Place Z
# =========================
Vec3 = Tuple[float, float, float]
Z_PICK   = 55.0
Z_OFFSET = 50.0
Z_TRAVEL = Z_PICK + Z_OFFSET

# ✅ 시작/마무리 고정 포즈
HOME_XYZ = (189.0, 0.0, 230.0)

# 정리모드(스택)
STACK_DZ = 28.0
Y_PICK_BIAS = 0.0
N_BEZIER = 25

DROP_XY_MAP: Dict[str, Tuple[float, float]] = {
    "red-cube":   (140.0, -90.0),
    "green-cube": (140.0,   0.0),
    "blue-cube":  (140.0,  90.0),
}

# =========================
# Serial (로봇/펌프)
# =========================
EOL = "\r\n"
BAUDS = [115200, 38400]

RE_STATUS = re.compile(
    r"Cartesian coordinate\(XYZ RxRyRz\):"
    r"\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*,\s*([-\d\.]+)",
    re.IGNORECASE,
)

def guess_ports():
    ports = []
    ports.extend(sorted(glob("/dev/serial/by-id/*")))
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

    seen, out = set(), []
    for p in ports:
        if p not in seen:
            out.append(p); seen.add(p)
    if not out:
        out = [p.device for p in plist]
    return out

def open_serial(port: str, baud: int, timeout=0.15, toggle_dtr_rts=True):
    ser = serial.Serial(
        port=port, baudrate=baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
        write_timeout=1.0,
        xonxoff=False, rtscts=False, dsrdtr=False,
    )
    if toggle_dtr_rts:
        try:
            ser.setDTR(False); ser.setRTS(False); time.sleep(0.05)
            ser.setDTR(True);  ser.setRTS(True)
        except Exception:
            pass
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    return ser

def ser_write_line(ser: serial.Serial, cmd: str):
    ser.write((cmd + EOL).encode("utf-8"))
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

def parse_state_token(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("<"):
        return s[1:].split(",", 1)[0].strip()
    return "UNKNOWN"

def query_status_pose(ser: serial.Serial):
    try:
        ser_drain(ser, 0.01)
        ser_write_line(ser, "?")
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
    """pose 전용 폴링 스레드(정리/모션 중에는 stop 필수)"""
    def __init__(self, ser: serial.Serial, pose_hz: float = 10.0):
        super().__init__(daemon=True)
        self.ser = ser
        self.pose_dt = 1.0 / max(1e-6, pose_hz)
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.last_ok = False
        self.last_state = "UNKNOWN"
        self.last_pose = (0.0, 0.0, 0.0)
        self.last_rpy  = (0.0, 0.0, 0.0)

    def run(self):
        t_next = 0.0
        while not self.stop_evt.is_set():
            now = time.time()
            if now >= t_next:
                t_next = now + self.pose_dt
                ok, state, xyz, rpy, _ = query_status_pose(self.ser)
                with self.lock:
                    self.last_ok = ok
                    self.last_state = state
                    if ok and xyz is not None:
                        self.last_pose = xyz
                        if rpy is not None:
                            self.last_rpy = rpy
            time.sleep(0.005)

    def snapshot(self):
        with self.lock:
            return dict(ok=self.last_ok, state=self.last_state, pose=self.last_pose, rpy=self.last_rpy)

    def stop(self):
        self.stop_evt.set()

def rx_all(ser, delay=0.05):
    time.sleep(delay)
    try:
        n = ser.in_waiting
    except Exception:
        n = 0
    return ser.read(n).decode(errors="ignore") if n else ""

def tx(ser, cmd):
    print("TX:", cmd)
    ser.write((cmd + EOL).encode())
    ser.flush()

def wait_ok(ser, timeout=3.0):
    """ok OR <idle 포함이면 성공"""
    t0 = time.time()
    buf = ""
    while time.time() - t0 < timeout:
        buf += rx_all(ser, delay=0.02)
        low = buf.lower()
        if "\nok" in ("\n" + low) or low.endswith("ok"):
            return True, buf
        if "<idle" in low:
            return True, buf
        if "error" in low or "alarm" in low or "lock" in low:
            return False, buf
    if "<idle" in buf.lower():
        return True, buf
    return False, buf

def wait_idle_by_query(ser, timeout=12.0):
    t0 = time.time()
    buf_all = ""
    while time.time() - t0 < timeout:
        ser_write_line(ser, "?")
        raw = ser_read_until(ser, end_char=b">", max_wait=0.55)
        buf_all += raw
        if parse_state_token(raw).upper() == "IDLE":
            return True, buf_all
        time.sleep(0.05)
    return False, buf_all

def send_g1_and_wait_idle(ser, g1_cmd: str, timeout=12.0):
    tx(ser, g1_cmd)
    ok, buf = wait_ok(ser, timeout=min(2.0, timeout))
    if ok:
        return True, buf
    ok2, buf2 = wait_idle_by_query(ser, timeout=timeout)
    return ok2, (buf + "\n" + buf2)

def pump_on(ser, pwm: int = 1000):
    for cmd in (f"M3S{int(pwm)}", f"M3 S{int(pwm)}"):
        tx(ser, cmd)
        ok, _ = wait_ok(ser, 2.0)
        if ok:
            return True
    print("[WARN] pump_on: no-ok (but may still work)")
    return False

def pump_off(ser):
    for cmd in ("M3S0", "M3 S0"):
        tx(ser, cmd)
        ok, _ = wait_ok(ser, 2.0)
        if ok:
            return True
    print("[WARN] pump_off: no-ok (but may still work)")
    return False


# =========================
# ray-plane
# =========================
def normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n

def pixel_to_ray_cam(u, v, K):
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    x = (float(u) - cx) / fx
    y = (float(v) - cy) / fy
    d = np.array([x, y, 1.0], dtype=np.float64)
    return normalize(d)

def intersect_ray_plane_z(cam_pos_base, ray_dir_base, plane_z):
    dz = float(ray_dir_base[2])
    if abs(dz) < 1e-9:
        return None
    s = (float(plane_z) - float(cam_pos_base[2])) / dz
    if s <= 0:
        return None
    P = np.asarray(cam_pos_base, dtype=np.float64) + s * np.asarray(ray_dir_base, dtype=np.float64)
    return P


# =========================
# Affine load/apply
# =========================
class Affine2D:
    def __init__(self):
        self.M = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]], dtype=np.float64)

    def load(self, path: str):
        d = np.load(path, allow_pickle=False)
        M = d["M"]
        if M.shape != (2,3):
            raise ValueError(f"Invalid affine M shape {M.shape}")
        self.M = M.astype(np.float64)

    def apply(self, x, y):
        x = float(x); y = float(y)
        xx = self.M[0,0]*x + self.M[0,1]*y + self.M[0,2]
        yy = self.M[1,0]*x + self.M[1,1]*y + self.M[1,2]
        return float(xx), float(yy)


# =========================
# YOLO
# =========================
def load_yolo():
    model = torch.hub.load(YOLO_REPO, "custom", path=YOLO_WEIGHT, source="local")
    model.conf = float(CONF_THRES)
    model.iou = float(IOU_THRES)
    return model

def pick_detections(df, target_names: Optional[List[str]] = None):
    out = []
    for _, row in df.iterrows():
        conf = float(row["confidence"])
        if conf < CONF_THRES:
            continue
        name = str(row["name"])
        if target_names and name not in target_names:
            continue
        out.append(row)
    return out


# =========================
# Bezier + pick&place
# =========================
def make_bezier_arc_xy_z(p_start: Vec3, p_end: Vec3, n_points: int = 25,
                         h_min: float = 40.0, k: float = 0.3,
                         z_min: float = 40.0, z_max: float = 300.0, margin: float = 10.0) -> List[Vec3]:
    x1,y1,z1 = p_start
    x2,y2,z2 = p_end
    d = math.hypot(x2-x1, y2-y1)
    h = max(h_min, k*d)
    z_mid = max(z1, z2) + h
    z_mid = min(z_mid, z_max - margin)
    z_mid = max(z_mid, z_min + margin)
    xm, ym = (x1+x2)/2.0, (y1+y2)/2.0
    P0, P1, P2 = (x1,y1,z1), (xm,ym,z_mid), (x2,y2,z2)

    path = []
    for i in range(n_points):
        t = i/(n_points-1)
        s = 1.0 - t
        x = s*s*P0[0] + 2*s*t*P1[0] + t*t*P2[0]
        y = s*s*P0[1] + 2*s*t*P1[1] + t*t*P2[1]
        z = s*s*P0[2] + 2*s*t*P1[2] + t*t*P2[2]
        path.append((x,y,z))
    return path

def init_motion_mode(ser):
    rx_all(ser, 0.15)
    for c in ("M21", "M20", "G90"):
        tx(ser, c); _ = wait_ok(ser, 1.2)
    tx(ser, "M50"); _ = wait_ok(ser, 2.0)

def move_to_home(ser, feed: float = 2500.0, timeout: float = 20.0) -> bool:
    """✅ 시작/마무리 고정 포즈로 이동"""
    try:
        init_motion_mode(ser)
        x, y, z = HOME_XYZ
        ok, resp = send_g1_and_wait_idle(ser, f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed:.1f}", timeout=timeout)
        if not ok:
            print("[HOME] FAIL:", resp.strip())
            return False
        print(f"[HOME] moved to ({x:.1f},{y:.1f},{z:.1f})")
        return True
    except Exception as e:
        print("[HOME] EXC:", e)
        return False

def move_with_pump_between_points(ser, p_pick: Vec3, p_place: Vec3,
                                  feed: float = 2000.0, n_points: int = 25) -> bool:
    x1, y1, _ = p_pick
    x2, y2, z2 = p_place

    p_top_start: Vec3 = (x1, y1, Z_TRAVEL)
    p_top_end:   Vec3 = (x2, y2, z2 + Z_OFFSET)
    path_top = make_bezier_arc_xy_z(p_top_start, p_top_end, n_points=n_points)

    try:
        init_motion_mode(ser)

        ok, resp = send_g1_and_wait_idle(ser, f"G1 X{p_top_start[0]:.3f} Y{p_top_start[1]:.3f} Z{p_top_start[2]:.3f} F{feed:.1f}", 15.0)
        if not ok: print("[FAIL] pick top:", resp.strip()); return False

        ok, resp = send_g1_and_wait_idle(ser, f"G1 X{p_top_start[0]:.3f} Y{p_top_start[1]:.3f} Z{Z_PICK:.3f} F{feed:.1f}", 15.0)
        if not ok: print("[FAIL] pick down:", resp.strip()); return False

        pump_on(ser, 1000); time.sleep(1.0)

        ok, resp = send_g1_and_wait_idle(ser, f"G1 X{p_top_start[0]:.3f} Y{p_top_start[1]:.3f} Z{Z_TRAVEL:.3f} F{feed:.1f}", 15.0)
        if not ok: print("[FAIL] pick up:", resp.strip()); return False

        for (x, y, z) in path_top[1:]:
            ok, resp = send_g1_and_wait_idle(ser, f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{feed:.1f}", 10.0)
            if not ok: print("[FAIL] bezier:", resp.strip()); return False
            time.sleep(0.01)

        ok, resp = send_g1_and_wait_idle(ser, f"G1 X{p_top_end[0]:.3f} Y{p_top_end[1]:.3f} Z{z2:.3f} F{feed:.1f}", 15.0)
        if not ok: print("[FAIL] place down:", resp.strip()); return False

        time.sleep(0.3); pump_off(ser)

        ok, resp = send_g1_and_wait_idle(ser, f"G1 X{p_top_end[0]:.3f} Y{p_top_end[1]:.3f} Z{(z2+Z_OFFSET):.3f} F{feed:.1f}", 15.0)
        if not ok: print("[FAIL] place up:", resp.strip()); return False

        try:
            tx(ser, "M400"); _ = wait_ok(ser, 10.0)
        except Exception:
            pass

        print("✅ pick&place done")
        return True
    except Exception as e:
        print("[pick_place_fail]", e)
        return False


# =========================
# 정리모드 알고리즘
# =========================
def sort_and_stack_cleanup(ser, detections: List[dict], feed=2000.0, n_points=25) -> bool:
    remain = [d for d in detections if d.get("cls") in DROP_XY_MAP]
    if not remain:
        print("[CLEANUP] supported detections none")
        return False

    stack_counts: Dict[str, int] = {}
    ref_x, ref_y = 0.0, 0.0
    total = len(remain)
    print(f"[CLEANUP] total={total}")

    step = 1
    while remain:
        best_i, best_d = None, float("inf")
        for i, det in enumerate(remain):
            dx = float(det["Xo"]) - ref_x
            dy = float(det["Yo"]) - ref_y
            d = math.hypot(dx, dy)
            if d < best_d:
                best_d, best_i = d, i

        tgt = remain.pop(best_i)
        cls = tgt["cls"]
        drop_x, drop_y = DROP_XY_MAP[cls]
        cnt = stack_counts.get(cls, 0)
        z_place = Z_PICK + STACK_DZ * cnt
        stack_counts[cls] = cnt + 1

        x_pick = float(tgt["Xo"])
        y_pick = float(tgt["Yo"]) + float(Y_PICK_BIAS)
        p_pick = (x_pick, y_pick, Z_PICK)
        p_place = (float(drop_x), float(drop_y), float(z_place))

        print(f"[CLEANUP] {step}/{total} {cls}  pick=({p_pick[0]:.1f},{p_pick[1]:.1f},{p_pick[2]:.1f})"
              f" -> place=({p_place[0]:.1f},{p_place[1]:.1f},{p_place[2]:.1f})  dist={best_d:.1f}")

        ok = move_with_pump_between_points(ser, p_pick, p_place, feed=feed, n_points=n_points)
        if not ok:
            print("[CLEANUP] FAIL -> stop")
            return False

        ref_x, ref_y = float(drop_x), float(drop_y)
        step += 1

    print("[CLEANUP] DONE")
    return True


# =========================
# Main
# =========================
def main():
    for p in [CALIB_PATH, AFFINE_PATH, YOLO_WEIGHT]:
        if not os.path.isfile(p):
            print("[-] file not found:", p); return
    if not os.path.isdir(YOLO_REPO):
        print("[-] yolo repo not found:", YOLO_REPO); return

    data = np.load(CALIB_PATH)
    K = data["K"]; dist = data["dist"]
    print("[INFO] calib:", CALIB_PATH)

    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2 if IS_LINUX else 0)
    if not cap.isOpened():
        print("[-] camera open fail:", CAM_ID); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    for _ in range(max(0, int(WARMUP))):
        cap.grab()

    Wc = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hc = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] CAP size: {Wc}x{Hc}")

    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (Wc, Hc), ALPHA, (Wc, Hc))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (Wc, Hc), cv2.CV_16SC2)

    torch.set_grad_enabled(False)
    model = load_yolo()
    print("[INFO] YOLO loaded")

    aff = Affine2D()
    aff.load(AFFINE_PATH)
    print("[INFO] affine loaded:", AFFINE_PATH)

    ser = None
    worker = None
    try:
        for dev in guess_ports():
            for b in BAUDS:
                try:
                    ser = open_serial(dev, b, timeout=0.15, toggle_dtr_rts=True)
                    print(f"[INFO] serial CONNECTED (shared): {dev}@{b}")
                    break
                except Exception:
                    ser = None
            if ser is not None:
                break

        if ser is None:
            print("[WARN] serial 연결 실패 -> pick&place 불가(비전만)")
        else:
            worker = SerialWorker(ser, pose_hz=10.0)
            worker.start()

            # ✅ 프로그램 시작 시 홈 포즈로 이동
            if worker is not None:
                worker.stop()
                worker = None
                time.sleep(0.25)
                rx_all(ser, 0.15)
            move_to_home(ser)
            worker = SerialWorker(ser, pose_hz=10.0)
            worker.start()

        detections: List[dict] = []
        need_detect = False

        print("[INFO] key: w=detect(all)  e=cleanup(auto)  q=quit")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            view = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            show = view.copy()

            if worker is not None:
                snap = worker.snapshot()
                pose_ok = bool(snap["ok"])
            else:
                snap = dict(ok=False, pose=(0.0, 0.0, 500.0))
                pose_ok = False

            ee_pos_base = np.array(snap["pose"], dtype=np.float64) if pose_ok else np.array([0.0, 0.0, 500.0], dtype=np.float64)
            cam_pos_base = ee_pos_base + CAM_OFFSET_XYZ
            R_base_cam = R_CAM_FIX

            if need_detect:
                frame_rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                df = results.pandas().xyxy[0]
                rows = pick_detections(df, target_names=list(DROP_XY_MAP.keys()))

                detections = []
                for row in rows:
                    cls_name = str(row["name"])
                    conf = float(row["confidence"])
                    xmin, ymin = float(row["xmin"]), float(row["ymin"])
                    xmax, ymax = float(row["xmax"]), float(row["ymax"])
                    u = (xmin + xmax) * 0.5
                    v = (ymin + ymax) * 0.5

                    d_cam = pixel_to_ray_cam(u, v, newK)
                    d_base = R_base_cam @ d_cam
                    P_raw = intersect_ray_plane_z(cam_pos_base, d_base, PLANE_Z)
                    if P_raw is None:
                        continue
                    Xr, Yr = float(P_raw[0]), float(P_raw[1])
                    Xo, Yo = aff.apply(Xr, Yr)

                    detections.append({
                        "cls": cls_name,
                        "conf": conf,
                        "Xr": Xr, "Yr": Yr,
                        "Xo": Xo, "Yo": Yo,
                        "bbox": (xmin, ymin, xmax, ymax)
                    })

                print(f"[DETECT] n={len(detections)}")
                for i, d in enumerate(detections):
                    print(f"  [{i}] {d['cls']} conf={d['conf']:.2f}  OUT=({d['Xo']:.1f},{d['Yo']:.1f})")

                need_detect = False

            for d in detections:
                xmin, ymin, xmax, ymax = d["bbox"]
                cv2.rectangle(show, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,255,255), 2)
                cv2.putText(show, f"{d['cls']} {d['conf']:.2f}", (int(xmin), int(ymin)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.putText(show, "w:detect(all)  e:cleanup(auto)  q:quit", (20, Hc-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("ray-plane + affine + cleanup", show)
            key = (cv2.waitKey(1) & 0xFF)

            if key == ord("q"):
                break

            elif key == ord("w"):
                need_detect = True
                print("[INFO] detect requested")

            elif key == ord("e"):
                if ser is None:
                    print("[INFO] serial 미연결 -> cleanup 불가")
                    continue
                if not detections:
                    print("[INFO] 먼저 w로 감지하세요.")
                    continue

                if worker is not None:
                    print("[INFO] stop pose worker for cleanup")
                    worker.stop()
                    worker = None
                    time.sleep(0.25)
                    rx_all(ser, 0.15)

                # ✅ cleanup 시작 전 홈 포즈
                if not move_to_home(ser):
                    print("[INFO] home move failed -> cleanup abort")
                    worker = SerialWorker(ser, pose_hz=10.0)
                    worker.start()
                    continue

                ok_clean = sort_and_stack_cleanup(ser, detections, feed=2000.0, n_points=N_BEZIER)
                print("[RESULT] cleanup:", ok_clean)

                # ✅ cleanup 끝난 후 홈 포즈
                move_to_home(ser)

                detections = []

                try:
                    worker = SerialWorker(ser, pose_hz=10.0)
                    worker.start()
                    print("[INFO] pose worker restarted")
                except Exception:
                    worker = None

    finally:
        try:
            # ✅ 프로그램 종료 시 홈 포즈
            if ser is not None:
                try:
                    if worker is not None:
                        worker.stop()
                        worker = None
                        time.sleep(0.25)
                        rx_all(ser, 0.15)
                except Exception:
                    pass
                move_to_home(ser)
        except Exception:
            pass

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
