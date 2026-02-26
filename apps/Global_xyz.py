#version python 3.8
#coding=utf-8
import time, serial
from serial.tools import list_ports

EOL   = "\r\n"
BAUDS = [115200, 38400]

INIT_SAFE_Z = 120     # 초기화 시 올려둘 안전 Z(mm)
INIT_FEED   = 2000    # 이동 속도 mm/min

# ----------------------------
# 포트 탐색/선택 (리스트 1회 고정)
# ----------------------------
def get_ports():
    plist = list(list_ports.comports())
    pri = [p for p in plist if ("CH340" in (p.description or "") or "USB-SERIAL" in (p.description or ""))]
    return pri or plist

def print_ports(ps):
    print("\n[감지된 포트]")
    for i, p in enumerate(ps, 1):
        print(f"  {i}) {p.device}  |  {p.description}")

def pick_two_ports_same_menu():
    ps = get_ports()
    if not ps:
        print("✗ 포트를 찾지 못했습니다. (USB/권한/드라이버 확인)")
        return None, None

    print_ports(ps)

    s1 = input("\n로봇1(=World) 포트 번호 선택 (예: 1, 빈칸=자동) = ").strip()
    s2 = input("로봇2(=변환 후 이동) 포트 번호 선택 (예: 2, 빈칸=자동) = ").strip()

    def parse_choice(s):
        if s == "":
            return None
        try:
            idx = int(s)
            if not (1 <= idx <= len(ps)):
                raise ValueError
            return ps[idx - 1].device
        except:
            print("✗ 잘못된 번호입니다. → 자동 모드로 진행(None)")
            return None

    port1 = parse_choice(s1)
    port2 = parse_choice(s2)

    # 같은 포트를 두 로봇에 지정하면 곤란하니 경고만(막진 않음)
    if port1 is not None and port2 is not None and port1 == port2:
        print("⚠️ 경고: 로봇1과 로봇2가 같은 포트를 가리킵니다. (선택 확인 필요)")

    return port1, port2

# ----------------------------
# 시리얼 유틸
# ----------------------------
def open_try(port, baud):
    try:
        ser = serial.Serial(
            port, baudrate=baud,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.2, write_timeout=1.5,
            xonxoff=False, rtscts=False, dsrdtr=False
        )
        try:
            ser.setDTR(False); ser.setRTS(False); time.sleep(0.05)
            ser.setDTR(True);  ser.setRTS(True)
        except:
            pass
        ser.reset_input_buffer(); ser.reset_output_buffer(); time.sleep(0.2)
        return ser
    except Exception as e:
        print(f"[open_fail] {port}@{baud}: {e}")
        return None

def rx_all(ser):
    time.sleep(0.05)
    n = ser.in_waiting
    return ser.read(n).decode(errors="ignore") if n else ""

def tx(ser, cmd):
    print("TX:", cmd)
    ser.write((cmd + EOL).encode())
    ser.flush()

def wait_ok(ser, timeout=3.0):
    t0 = time.time()
    buf = ""
    while time.time() - t0 < timeout:
        buf += rx_all(ser)
        low = buf.lower()
        if "\nok" in ("\n" + low) or low.endswith("ok"):
            return True, buf
        if "error" in low or "alarm" in low or "lock" in low:
            return False, buf
        time.sleep(0.02)
    return False, buf

def connect_any(port_hint=None):
    """
    port_hint가 있으면 그 포트만 시도.
    없으면 감지된 포트 전부 + BAUDS 조합으로 스캔.
    """
    candidate_ports = []
    if port_hint:
        candidate_ports = [port_hint]
    else:
        candidate_ports = [p.device for p in get_ports()]

    for port in candidate_ports:
        for b in BAUDS:
            ser = open_try(port, b)
            if ser:
                return ser, port, b
    return None, None, None

# ----------------------------
# 로봇 초기화/이동
# ----------------------------
def init_robot(port_hint=None, name="robot"):
    ser, port, b = connect_any(port_hint)
    if not ser:
        print(f"✗ {name} 연결 실패")
        return False

    try:
        rx_all(ser)

        # 모드 세팅
        for c in ("M21", "M20", "G90"):
            tx(ser, c); _ = wait_ok(ser, 1.2)

        # unlock
        tx(ser, "M50"); time.sleep(0.2)
        _ = rx_all(ser)

        # homing (펌웨어에 따라 다를 수 있음)
        tx(ser, "G28")
        ok, resp = wait_ok(ser, 12.0)
        if not ok:
            print(f"[hint] {name} G28 ok 미수신—펌웨어/상태 확인 필요")
            # 여기서 중단할지 계속할지는 현장 상황에 따라. 일단 계속 진행.

        # 안전 Z + 자세 정렬
        tx(ser, f"G0 Z{INIT_SAFE_Z}")
        wait_ok(ser, 3.0)

        tx(ser, f"G1 A0 B0 C0 F{INIT_FEED}")
        wait_ok(ser, 4.0)

        print(f"✅ {name} Initialized on {port} @ {b} (Z={INIT_SAFE_Z}, A=B=C=0)")
        return True

    except Exception as e:
        print(f"[init_fail:{name}]", e)
        return False
    finally:
        try: ser.close()
        except: pass

def move_cartesian(port_hint, x=None, y=None, z=None, feed=2000, name="robot"):
    ser, port, b = connect_any(port_hint)
    if not ser:
        print(f"✗ {name} 연결 실패")
        return False

    try:
        rx_all(ser)

        for c in ("M21", "M20", "G90"):
            tx(ser, c); _ = wait_ok(ser, 1.2)

        parts = ["G1"]
        if x is not None: parts.append(f"X{float(x):.3f}")
        if y is not None: parts.append(f"Y{float(y):.3f}")
        if z is not None: parts.append(f"Z{float(z):.3f}")
        parts.append(f"F{feed}")
        cmd = " ".join(parts)

        tx(ser, cmd)
        ok, resp = wait_ok(ser, 10.0)
        if not ok:
            print(f"[FAIL:{name}] 이동 거부:", resp.strip())
            return False

        # 지원 시 동작 완료 대기
        try:
            tx(ser, "M400")
            wait_ok(ser, 10.0)
        except:
            pass

        print(f"🎯 {name} Sent: {cmd}  (port={port}, baud={b})")
        return True

    except Exception as e:
        print(f"[move_fail:{name}]", e)
        return False
    finally:
        try: ser.close()
        except: pass

# ----------------------------
# 좌표 변환 (World -> Robot2 Base)
# Robot1(Base1) == World (0,0,0)
# Robot2: World에서 (400,200,0), yaw=180deg
# ----------------------------
def world_to_robot2(xw, yw, zw, tx=400.0, ty=200.0, tz=0.0):
    # yaw=180 => R=diag(-1,-1,1), and p2 = R^T (pw - t) = R (pw - t)
    x2 = -(xw - tx)   # = tx - xw
    y2 = -(yw - ty)   # = ty - yw
    z2 =  (zw - tz)
    return x2, y2, z2

def read_float(prompt):
    s = input(prompt).strip()
    return float(s)

# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    port_r1, port_r2 = pick_two_ports_same_menu()

    print("\n[선택 결과]")
    print(" - 로봇1(World) 포트:", port_r1 if port_r1 else "자동")
    print(" - 로봇2(이동)  포트:", port_r2 if port_r2 else "자동")

    # 초기화: 로봇2는 필수(이동 대상), 로봇1은 옵션
    if port_r1 is not None:
        init_robot(port_r1, name="Robot1(World)")

    ok = init_robot(port_r2, name="Robot2")
    if not ok:
        print("✗ Robot2 초기화 실패")
        raise SystemExit(1)

    # World 좌표 입력
    print("\n[입력] World 좌표 (Robot1 Base와 동일) mm")
    try:
        Xw = read_float("World X = ")
        Yw = read_float("World Y = ")
        Zw = read_float("World Z = ")
    except Exception:
        print("✗ 숫자로 입력해주세요.")
        raise SystemExit(1)

    # 변환
    X2, Y2, Z2 = world_to_robot2(Xw, Yw, Zw)
    print(f"\n[변환] World({Xw:.3f},{Yw:.3f},{Zw:.3f}) -> Robot2({X2:.3f},{Y2:.3f},{Z2:.3f})")

    # Robot2 이동
    moved = move_cartesian(port_r2, X2, Y2, Z2, feed=INIT_FEED, name="Robot2")
    if not moved:
        raise SystemExit(1)#version python 3.8
#coding=utf-8
import time, serial
from serial.tools import list_ports

EOL   = "\r\n"
BAUDS = [115200, 38400]

INIT_SAFE_Z = 120     # 초기화 시 올려둘 안전 Z(mm)
INIT_FEED   = 2000    # 이동 속도 mm/min

# ----------------------------
# 포트 탐색/선택 (리스트 1회 고정)
# ----------------------------
def get_ports():
    plist = list(list_ports.comports())
    pri = [p for p in plist if ("CH340" in (p.description or "") or "USB-SERIAL" in (p.description or ""))]
    return pri or plist

def print_ports(ps):
    print("\n[감지된 포트]")
    for i, p in enumerate(ps, 1):
        print(f"  {i}) {p.device}  |  {p.description}")

def pick_two_ports_same_menu():
    ps = get_ports()
    if not ps:
        print("✗ 포트를 찾지 못했습니다. (USB/권한/드라이버 확인)")
        return None, None

    print_ports(ps)

    s1 = input("\n로봇1(=World) 포트 번호 선택 (예: 1, 빈칸=자동) = ").strip()
    s2 = input("로봇2(=변환 후 이동) 포트 번호 선택 (예: 2, 빈칸=자동) = ").strip()

    def parse_choice(s):
        if s == "":
            return None
        try:
            idx = int(s)
            if not (1 <= idx <= len(ps)):
                raise ValueError
            return ps[idx - 1].device
        except:
            print("✗ 잘못된 번호입니다. → 자동 모드로 진행(None)")
            return None

    port1 = parse_choice(s1)
    port2 = parse_choice(s2)

    # 같은 포트를 두 로봇에 지정하면 곤란하니 경고만(막진 않음)
    if port1 is not None and port2 is not None and port1 == port2:
        print("⚠️ 경고: 로봇1과 로봇2가 같은 포트를 가리킵니다. (선택 확인 필요)")

    return port1, port2

# ----------------------------
# 시리얼 유틸
# ----------------------------
def open_try(port, baud):
    try:
        ser = serial.Serial(
            port, baudrate=baud,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.2, write_timeout=1.5,
            xonxoff=False, rtscts=False, dsrdtr=False
        )
        try:
            ser.setDTR(False); ser.setRTS(False); time.sleep(0.05)
            ser.setDTR(True);  ser.setRTS(True)
        except:
            pass
        ser.reset_input_buffer(); ser.reset_output_buffer(); time.sleep(0.2)
        return ser
    except Exception as e:
        print(f"[open_fail] {port}@{baud}: {e}")
        return None

def rx_all(ser):
    time.sleep(0.05)
    n = ser.in_waiting
    return ser.read(n).decode(errors="ignore") if n else ""

def tx(ser, cmd):
    print("TX:", cmd)
    ser.write((cmd + EOL).encode())
    ser.flush()

def wait_ok(ser, timeout=3.0):
    t0 = time.time()
    buf = ""
    while time.time() - t0 < timeout:
        buf += rx_all(ser)
        low = buf.lower()
        if "\nok" in ("\n" + low) or low.endswith("ok"):
            return True, buf
        if "error" in low or "alarm" in low or "lock" in low:
            return False, buf
        time.sleep(0.02)
    return False, buf

def connect_any(port_hint=None):
    """
    port_hint가 있으면 그 포트만 시도.
    없으면 감지된 포트 전부 + BAUDS 조합으로 스캔.
    """
    candidate_ports = []
    if port_hint:
        candidate_ports = [port_hint]
    else:
        candidate_ports = [p.device for p in get_ports()]

    for port in candidate_ports:
        for b in BAUDS:
            ser = open_try(port, b)
            if ser:
                return ser, port, b
    return None, None, None

# ----------------------------
# 로봇 초기화/이동
# ----------------------------
def init_robot(port_hint=None, name="robot"):
    ser, port, b = connect_any(port_hint)
    if not ser:
        print(f"✗ {name} 연결 실패")
        return False

    try:
        rx_all(ser)

        # 모드 세팅
        for c in ("M21", "M20", "G90"):
            tx(ser, c); _ = wait_ok(ser, 1.2)

        # unlock
        tx(ser, "M50"); time.sleep(0.2)
        _ = rx_all(ser)

        # homing (펌웨어에 따라 다를 수 있음)
        tx(ser, "G28")
        ok, resp = wait_ok(ser, 12.0)
        if not ok:
            print(f"[hint] {name} G28 ok 미수신—펌웨어/상태 확인 필요")
            # 여기서 중단할지 계속할지는 현장 상황에 따라. 일단 계속 진행.

        # 안전 Z + 자세 정렬
        tx(ser, f"G0 Z{INIT_SAFE_Z}")
        wait_ok(ser, 3.0)

        tx(ser, f"G1 A0 B0 C0 F{INIT_FEED}")
        wait_ok(ser, 4.0)

        print(f"✅ {name} Initialized on {port} @ {b} (Z={INIT_SAFE_Z}, A=B=C=0)")
        return True

    except Exception as e:
        print(f"[init_fail:{name}]", e)
        return False
    finally:
        try: ser.close()
        except: pass

def move_cartesian(port_hint, x=None, y=None, z=None, feed=2000, name="robot"):
    ser, port, b = connect_any(port_hint)
    if not ser:
        print(f"✗ {name} 연결 실패")
        return False

    try:
        rx_all(ser)

        for c in ("M21", "M20", "G90"):
            tx(ser, c); _ = wait_ok(ser, 1.2)

        parts = ["G1"]
        if x is not None: parts.append(f"X{float(x):.3f}")
        if y is not None: parts.append(f"Y{float(y):.3f}")
        if z is not None: parts.append(f"Z{float(z):.3f}")
        parts.append(f"F{feed}")
        cmd = " ".join(parts)

        tx(ser, cmd)
        ok, resp = wait_ok(ser, 10.0)
        if not ok:
            print(f"[FAIL:{name}] 이동 거부:", resp.strip())
            return False

        # 지원 시 동작 완료 대기
        try:
            tx(ser, "M400")
            wait_ok(ser, 10.0)
        except:
            pass

        print(f"🎯 {name} Sent: {cmd}  (port={port}, baud={b})")
        return True

    except Exception as e:
        print(f"[move_fail:{name}]", e)
        return False
    finally:
        try: ser.close()
        except: pass

# ----------------------------
# 좌표 변환 (World -> Robot2 Base)
# Robot1(Base1) == World (0,0,0)
# Robot2: World에서 (400,200,0), yaw=180deg
# ----------------------------
def world_to_robot2(xw, yw, zw, tx=400.0, ty=200.0, tz=0.0):
    # yaw=180 => R=diag(-1,-1,1), and p2 = R^T (pw - t) = R (pw - t)
    x2 = -(xw - tx)   # = tx - xw
    y2 = -(yw - ty)   # = ty - yw
    z2 =  (zw - tz)
    return x2, y2, z2

def read_float(prompt):
    s = input(prompt).strip()
    return float(s)

# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    port_r1, port_r2 = pick_two_ports_same_menu()

    print("\n[선택 결과]")
    print(" - 로봇1(World) 포트:", port_r1 if port_r1 else "자동")
    print(" - 로봇2(이동)  포트:", port_r2 if port_r2 else "자동")

    # 초기화: 로봇2는 필수(이동 대상), 로봇1은 옵션
    if port_r1 is not None:
        init_robot(port_r1, name="Robot1(World)")

    ok = init_robot(port_r2, name="Robot2")
    if not ok:
        print("✗ Robot2 초기화 실패")
        raise SystemExit(1)

    # World 좌표 입력
    print("\n[입력] World 좌표 (Robot1 Base와 동일) mm")
    try:
        Xw = read_float("World X = ")
        Yw = read_float("World Y = ")
        Zw = read_float("World Z = ")
    except Exception:
        print("✗ 숫자로 입력해주세요.")
        raise SystemExit(1)

    # 변환
    X2, Y2, Z2 = world_to_robot2(Xw, Yw, Zw)
    print(f"\n[변환] World({Xw:.3f},{Yw:.3f},{Zw:.3f}) -> Robot2({X2:.3f},{Y2:.3f},{Z2:.3f})")

    # Robot2 이동
    moved = move_cartesian(port_r2, X2, Y2, Z2, feed=INIT_FEED, name="Robot2")
    if not moved:
        raise SystemExit(1)