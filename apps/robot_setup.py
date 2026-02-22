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



def try_connect_and_jog():  # ← 이름은 그대로 두고 동작만 '초기화'로 변경
    for p in ports():
        for b in BAUDS:
            print(f"\n=== Probing {p.device} @ {b} ===")
            ser = open_try(p.device, b)
            if not ser:
                continue
            try:
                rx_all(ser)  # 드레인

                # ── 모드 정리: 조인트→카테시안, 절대좌표 ──
                for c in ("M21", "M20", "G90"):
                    tx(ser, c); _ = wait_ok(ser, 1.2)

                # ── 핑/정보 (펌웨어 따라 다름) ──
                tx(ser, "M50"); time.sleep(0.2)
                print("RX:", rx_all(ser).strip())

                # ── 홈 ── (미지원이면 힌트만 출력)
                tx(ser, "G28"); ok, _ = wait_ok(ser, 12.0)
                if not ok:
                    print("[hint] G28 ok 미수신—펌웨어 미지원일 수 있음")

                # ── 초기화 자세 ──
                # 1) 안전 Z로 올리기
                tx(ser, f"G0 Z{INIT_SAFE_Z}"); wait_ok(ser, 3.0)

                # 2) 손목 오리엔테이션 정렬(0,0,0)
                #    (X/Y는 현 위치 유지. 필요하면 X0 Y0도 추가 가능)
                tx(ser, f"G1 A0 B0 C0 F{INIT_FEED}"); wait_ok(ser, 4.0)

                # 필요 시 절대 좌표로 기준점 이동까지 하고 싶다면 아래 주석 해제:
                # tx(ser, f"G1 X0 Y0 F{INIT_FEED}"); wait_ok(ser, 6.0)

                print(f"✅ Initialized on {p.device} @ {b} (Z={INIT_SAFE_Z}, A=B=C=0)")
                ser.close()
                return True

            except Exception as e:
                print("[run_fail]", e)
            finally:
                try: ser.close()
                except: pass
    return False
# ★ 절대 좌표로 (x,y,z) 이동하는 함수: 내부에서 직접 연결 → 이동 → 종료
def Control_Cartesian(x=None, y=None, z=None, feed=2000):
    # 첫 번째로 열리는 포트/보드레이트 조합을 사용 (네 기존 로직 유지)
    for p in ports():
        for b in BAUDS:
            print(f"\n=== Connect for Cartesian move: {p.device} @ {b} ===")
            ser = open_try(p.device, b)
            if not ser:
                continue
            try:
                rx_all(ser)
                for c in ("M21","M20","G90"):
                    tx(ser, c); _ = wait_ok(ser, 1.2)

                # 값이 주어진 축만 포함해서 G1 생성
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

                # (지원 시) 모션 완료까지 대기
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
    ok = try_connect_and_jog()
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
