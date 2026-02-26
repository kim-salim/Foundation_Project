#version python 3.8
#coding=utf-8
import time, serial
from serial.tools import list_ports

EOL   = "\r\n"
BAUDS = [115200, 38400]

# 조그 파라미터
JOG_MM = 20      # 각 축을 이동할 거리 (mm)
PAUSE  = 1.0     # 각 이동 사이 대기 시간 (초)

INIT_SAFE_Z = 120     # 초기화 시 올려둘 안전 Z(mm)
INIT_FEED   = 2000    # 자세 정렬(G1) 속도 mm/min

def ports():
    plist = list(list_ports.comports())
    pri = [p for p in plist if ("CH340" in (p.description or "") or "USB-SERIAL" in (p.description or ""))]
    return pri or plist

def pick_port_by_index():
    print("\n 감지된 포트")
    ps= ports()
    for i,p in enumerate(ps,1):
        print(f"  {i}) {p.device}  |  {p.description}")

    s = input("\n어느 로봇을 제어할까요? (예: 1 or 2, 빈칸=자동) = ").strip()
    if s=="":
        return None
    try:
        idx=int(s)
    except:
        print("✗ 숫자(1,2,...)로 입력해주세요. → 자동 모드로 진행")
        return None

    return f"/dev/ttyUSB{idx - 1}"

def open_try(port, baud):
    try:
        ser = serial.Serial(port, baudrate=baud,
                            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE, timeout=0.2, write_timeout=1.5,
                            xonxoff=False, rtscts=False, dsrdtr=False)
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
    ser.write((cmd+EOL).encode()); ser.flush()

def wait_ok(ser, timeout=3.0):
    t0 = time.time()
    buf = ""
    while time.time() - t0 < timeout:
        buf += rx_all(ser)
        low = buf.lower()
        if "\nok" in ("\n"+low) or low.endswith("ok"):
            return True, buf
        if "error" in low or "alarm" in low or "lock" in low:
            return False, buf
        time.sleep(0.02)
    return False, buf



def try_connect_and_jog(selected_port=None):
    candidate_ports = []
    if selected_port:
        candidate_ports = [selected_port]
    else:
        candidate_ports = [p.device for p in ports()]

    for port in candidate_ports:
        for b in BAUDS:
            print(f"\n=== Probing {port} @ {b} ===")
            ser = open_try(port, b)
            if not ser:
                continue
            try:
                rx_all(ser)

                for c in ("M21", "M20", "G90"):
                    tx(ser, c); _ = wait_ok(ser, 1.2)

                tx(ser, "M50"); time.sleep(0.2)
                print("RX:", rx_all(ser).strip())

                tx(ser, "G28"); ok, _ = wait_ok(ser, 12.0)
                if not ok:
                    print("[hint] G28 ok 미수신—펌웨어 미지원일 수 있음")

                tx(ser, f"G0 Z{INIT_SAFE_Z}"); wait_ok(ser, 3.0)
                tx(ser, f"G1 A0 B0 C0 F{INIT_FEED}"); wait_ok(ser, 4.0)

                print(f"✅ Initialized on {port} @ {b} (Z={INIT_SAFE_Z}, A=B=C=0)")
                ser.close()
                return True

            except Exception as e:
                print("[run_fail]", e)
            finally:
                try: ser.close()
                except: pass
    return False
# ★ 절대 좌표로 (x,y,z) 이동하는 함수: 내부에서 직접 연결 → 이동 → 종료
def Control_Cartesian(x=None, y=None, z=None, feed=2000, selected_port=None):
    candidate_ports = []
    if selected_port:
        candidate_ports = [selected_port]
    else:
        candidate_ports = [p.device for p in ports()]

    for port in candidate_ports:
        for b in BAUDS:
            print(f"\n=== Connect for Cartesian move: {port} @ {b} ===")
            ser = open_try(port, b)
            if not ser:
                continue
            try:
                rx_all(ser)
                for c in ("M21","M20","G90"):
                    tx(ser, c); _ = wait_ok(ser, 1.2)

                parts = ["G1"]
                if x is not None: parts.append(f"X{float(x):.3f}")
                if y is not None: parts.append(f"Y{float(y):.3f}")
                if z is not None: parts.append(f"Z{float(z):.3f}")
                parts.append(f"F{feed}")
                cmd = " ".join(parts)

                tx(ser, cmd)
                ok, resp = wait_ok(ser, 8.0)
                if not ok:
                    print("[FAIL] 이동 거부:", resp.strip())
                    return False

                try:
                    tx(ser, "M400"); _ = wait_ok(ser, 10.0)
                except:
                    pass

                print(f"🎯 Sent: {cmd}")
                ser.close()
                return True
            except Exception as e:
                print("[cartesian_fail]", e)
            finally:
                try: ser.close()
                except: pass

    print("✗ 연결 실패: 포트/보드레이트 조합이 모두 실패했습니다.")
    return False

def read_axis(name):
    s = input(f"{name} (빈칸=유지) = ").strip()
    return None if s == "" else float(s)

if __name__ == "__main__":
    selected_port = pick_port_by_index()
    if selected_port:
        print(f"선택된 포트를 제어합니다.")
    else:
        print("\n[자동] 감지된 포트 중 먼저 성공하는 장치를 제어합니다.\n")



    ok = try_connect_and_jog(selected_port=selected_port)
    if  ok:
        print("\n[입력] x,y,z 좌표를 입력하세요 (mm, 공백 또는 줄바꿈 구분)")
        try:
            X = read_axis("X")
            Y = read_axis("Y")
            Z = read_axis("Z")
        except Exception:
            print("✗ 숫자로 입력해주세요.")
        else:
            Control_Cartesian(X, Y, Z)
