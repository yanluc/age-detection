import time
import struct
import threading
import serial
import cv2
import numpy as np

# ---------------- CONFIG (hard-coded) ----------------
PORT = "/dev/ttyACM0"
BAUD = 921600
OPEN_DELAY_S = 1.0

# Camera choice (hard-coded)
CAMERA_INDEX = 0              # change to 1/2/etc if needed
CAMERA_BACKEND = None         # e.g. cv2.CAP_DSHOW on Windows, or None for default
FLIP_HORIZONTAL = True        # mirror view like a selfie camera

# Model input crop
CROP_SIZE = 200               # middle 200x200 will be used as input
DRAW_CROP_BOX_THICKNESS = 2

# UI pumping (keeps window responsive even during long inference)
UI_PUMP_DELAY_MS = 1          # keep small; we also sleep a bit below
UI_IDLE_SLEEP_S = 0.01        # reduce CPU usage

# Try to reduce camera buffering (best-effort)
CAMERA_BUFFER_SIZE = 1
DROP_GRABS_BEFORE_READ = 5    # drop buffered frames before reading a fresh one

# Serial/protocol timeouts
TIMEOUT_S = 10.0
TIMEOUT_CHUNK_S = 25.0        # IMPORTANT: > ESP inactivity timeout (15s)
TIMEOUT_INFER_S = 180.0
# -----------------------------------------------------

MAGIC_REQ = b"AGE1"
MAGIC_ACK = b"ACK1"
MAGIC_OKA = b"OKA1"
MAGIC_RES = b"RES1"
MAGIC_ERR = b"ERR1"
MAGIC_LOG = b"LOG1"

PROTO_VER = 1
FMT_RAW_RGB24 = 0


def fnv1a32_update(h: int, data: bytes) -> int:
    for b in data:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def fnv1a32(data: bytes) -> int:
    return fnv1a32_update(2166136261, data)


def read_exact(ser: serial.Serial, n: int, timeout_s: float) -> bytes:
    deadline = time.time() + timeout_s
    out = bytearray()
    while len(out) < n:
        if time.time() > deadline:
            raise TimeoutError(f"Timeout reading {n} bytes (got {len(out)})")
        chunk = ser.read(n - len(out))
        if chunk:
            out.extend(chunk)
        else:
            time.sleep(0.001)
    return bytes(out)


def drain_input(ser: serial.Serial, max_bytes: int = 1_000_000):
    total = 0
    while True:
        waiting = ser.in_waiting
        if not waiting:
            break
        chunk = ser.read(min(waiting, 4096))
        if not chunk:
            break
        total += len(chunk)
        if total >= max_bytes:
            break


def ping(ser: serial.Serial):
    drain_input(ser)
    ser.write(b"PING")
    ser.flush()
    return read_exact(ser, 4, TIMEOUT_S)


def parse_log1(ser: serial.Serial, timeout_s: float):
    body = read_exact(ser, 1 + 1 + 2, timeout_s)
    ver = body[0]
    lvl = body[1]
    ln = struct.unpack_from("<H", body, 2)[0]
    msg = read_exact(ser, ln, timeout_s)
    csum = struct.unpack("<I", read_exact(ser, 4, timeout_s))[0]

    calc = 2166136261
    calc = fnv1a32_update(calc, body)
    calc = fnv1a32_update(calc, msg)

    level_name = {0: "I", 1: "W", 2: "E"}.get(lvl, str(lvl))
    text = msg.decode("utf-8", errors="replace")
    print(f"[ESP][{level_name}] {text}")
    if ver != PROTO_VER:
        print(f"[PC] Warning: LOG ver {ver} != {PROTO_VER}")
    if (calc & 0xFFFFFFFF) != csum:
        print("[PC] Warning: LOG checksum mismatch")


def recv_magic_skipping_logs(ser: serial.Serial, timeout_s: float) -> bytes:
    deadline = time.time() + timeout_s
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Timeout waiting for frame magic")
        m = read_exact(ser, 4, remaining)
        if m == MAGIC_LOG:
            parse_log1(ser, timeout_s=TIMEOUT_S)
            continue
        return m


def recv_err(ser: serial.Serial, timeout_s: float) -> int:
    body = read_exact(ser, 1 + 1 + 2 + 4, timeout_s)
    ver = body[0]
    status = body[1]
    errcode = struct.unpack_from("<H", body, 2)[0]
    checksum = struct.unpack_from("<I", body, 4)[0]
    calc = fnv1a32(body[:4])
    if checksum != calc:
        raise ValueError("Bad ERR checksum")
    if ver != PROTO_VER or status != 1:
        raise ValueError("Bad ERR header")
    return errcode


def recv_ack(ser: serial.Serial, timeout_s: float) -> int:
    body = read_exact(ser, 1 + 1 + 2 + 4 + 4, timeout_s)
    ver = body[0]
    status = body[1]
    reserved = struct.unpack_from("<H", body, 2)[0]
    chunk_hint = struct.unpack_from("<I", body, 4)[0]
    checksum = struct.unpack_from("<I", body, 8)[0]
    calc = fnv1a32(body[:8])
    if checksum != calc:
        raise ValueError("Bad ACK checksum")
    if ver != PROTO_VER or status != 0 or reserved != 0:
        raise ValueError("Bad ACK header")
    return chunk_hint


def recv_oka(ser: serial.Serial, timeout_s: float) -> int:
    body = read_exact(ser, 1 + 1 + 2 + 4 + 4, timeout_s)
    ver = body[0]
    status = body[1]
    reserved = struct.unpack_from("<H", body, 2)[0]
    received_total = struct.unpack_from("<I", body, 4)[0]
    checksum = struct.unpack_from("<I", body, 8)[0]
    calc = fnv1a32(body[:8])
    if checksum != calc:
        raise ValueError("Bad OKA checksum")
    if ver != PROTO_VER or status != 0 or reserved != 0:
        raise ValueError("Bad OKA header")
    return received_total


def recv_res(ser: serial.Serial, timeout_s: float):
    body = read_exact(ser, 1 + 1 + 2 + 4 + 4 + 4 + 4, timeout_s)
    ver = body[0]
    status = body[1]
    reserved = struct.unpack_from("<H", body, 2)[0]
    y_norm = struct.unpack_from("<f", body, 4)[0]
    age_years = struct.unpack_from("<f", body, 8)[0]
    infer_ms = struct.unpack_from("<I", body, 12)[0]
    checksum = struct.unpack_from("<I", body, 16)[0]
    calc = fnv1a32(body[:16])
    if checksum != calc:
        raise ValueError("Bad RES checksum")
    if ver != PROTO_VER or status != 0 or reserved != 0:
        raise ValueError("Bad RES header")
    return y_norm, age_years, infer_ms


class ESPAgeClient:
    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial(
            port, baud,
            timeout=0.1,
            write_timeout=None,
            xonxoff=False, rtscts=False, dsrdtr=False
        )
        time.sleep(OPEN_DELAY_S)

        print("[PC] PING...")
        pong = ping(self.ser)
        print(f"[PC] Got: {pong!r}")
        if pong != b"PONG":
            raise RuntimeError("Not PONG -> wrong firmware/port.")
        print("[PC] Serial ready.")

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def infer_rgb24(self, rgb_bytes: bytes, w: int, h: int):
        ser = self.ser
        drain_input(ser)

        checksum = fnv1a32(rgb_bytes)
        header = MAGIC_REQ + struct.pack("<BBHHII", PROTO_VER, FMT_RAW_RGB24, w, h, len(rgb_bytes), checksum)

        ser.write(header)
        ser.flush()

        m = recv_magic_skipping_logs(ser, TIMEOUT_S)
        if m == MAGIC_ERR:
            err = recv_err(ser, TIMEOUT_S)
            raise RuntimeError(f"ESP ERR before payload, errcode={err}")
        if m != MAGIC_ACK:
            raise RuntimeError(f"Expected ACK1, got {m!r}")

        chunk_hint = recv_ack(ser, TIMEOUT_S)
        if chunk_hint < 256 or chunk_hint > 65535:
            chunk_hint = 2048

        m = recv_magic_skipping_logs(ser, TIMEOUT_S)
        if m == MAGIC_ERR:
            err = recv_err(ser, TIMEOUT_S)
            raise RuntimeError(f"ESP ERR after ACK, errcode={err}")
        if m != MAGIC_OKA:
            raise RuntimeError(f"Expected OKA1 right after ACK, got {m!r}")
        got0 = recv_oka(ser, TIMEOUT_S)
        if got0 != 0:
            print(f"[PC] Warning: OKA initial total={got0} (expected 0)")

        mv = memoryview(rgb_bytes)
        n = len(mv)
        off = 0
        while off < n:
            end = min(off + chunk_hint, n)
            chunk = mv[off:end]
            clen = len(chunk)

            ser.write(struct.pack("<H", clen))
            ser.write(chunk)
            ser.flush()

            m = recv_magic_skipping_logs(ser, TIMEOUT_CHUNK_S)
            if m == MAGIC_ERR:
                err = recv_err(ser, TIMEOUT_S)
                raise RuntimeError(f"ESP ERR during payload, errcode={err}")
            if m != MAGIC_OKA:
                raise RuntimeError(f"Expected OKA1, got {m!r}")

            got_total = recv_oka(ser, TIMEOUT_S)
            off = end
            if got_total != off:
                raise RuntimeError(f"Device total mismatch: device={got_total} host={off}")

        ser.write(struct.pack("<H", 0))
        ser.flush()

        m = recv_magic_skipping_logs(ser, TIMEOUT_INFER_S)
        if m == MAGIC_ERR:
            err = recv_err(ser, TIMEOUT_S)
            raise RuntimeError(f"ESP ERR after payload, errcode={err}")
        if m != MAGIC_RES:
            raise RuntimeError(f"Expected RES1, got {m!r}")

        y_norm, age_years, infer_ms = recv_res(ser, TIMEOUT_INFER_S)
        return y_norm, age_years, infer_ms


def center_crop_coords(w: int, h: int, size: int):
    half = size // 2
    cx, cy = w // 2, h // 2
    x1 = cx - half
    y1 = cy - half
    x2 = x1 + size
    y2 = y1 + size

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return x1, y1, x2, y2


def draw_label_bottom_left(frame, text: str):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = 10
    y = h - 10
    pad = 6

    x1 = x - pad
    y1 = y - th - baseline - pad
    x2 = x + tw + pad
    y2 = y + pad
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def open_camera():
    if CAMERA_BACKEND is None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    except Exception:
        pass

    return cap


def read_latest_frame(cap: cv2.VideoCapture):
    for _ in range(max(0, int(DROP_GRABS_BEFORE_READ))):
        cap.grab()
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def prepare_frame_and_crop(frame_bgr: np.ndarray):
    if FLIP_HORIZONTAL:
        frame_bgr = cv2.flip(frame_bgr, 1)

    h, w = frame_bgr.shape[:2]
    if h < CROP_SIZE or w < CROP_SIZE:
        scale = max(CROP_SIZE / max(1, w), CROP_SIZE / max(1, h))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = center_crop_coords(w, h, CROP_SIZE)
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb_bytes = crop_rgb.tobytes()
    return frame_bgr, (x1, y1, x2, y2), rgb_bytes


def main():
    print(f"[PC] Opening serial: {PORT} @ {BAUD}")
    client = ESPAgeClient(PORT, BAUD)

    cap = None
    try:
        print(f"[PC] Opening camera index: {CAMERA_INDEX}")
        cap = open_camera()

        win_name = "ESP Age Inference (responsive, 1 frame per inference)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # Shared state
        state_lock = threading.Lock()
        current_frame = None          # frozen display frame (BGR)
        current_box = None            # (x1,y1,x2,y2) for current_frame
        last_age = None
        last_infer_ms = None
        last_err = None

        busy = False
        infer_start_t = 0.0
        done_event = threading.Event()
        worker_exc = None
        worker_result = None          # (age_years, infer_ms) on success

        # After inference completes, show result on the frozen frame for at least one UI tick
        completed = False
        shown_completed_once = False

        def infer_worker(rgb_bytes: bytes):
            nonlocal worker_exc, worker_result
            try:
                _, age_years, infer_ms = client.infer_rgb24(rgb_bytes, CROP_SIZE, CROP_SIZE)
                worker_result = (float(age_years), int(infer_ms))
                worker_exc = None
            except Exception as e:
                worker_result = None
                worker_exc = str(e)
            finally:
                done_event.set()

        def start_next_inference():
            nonlocal current_frame, current_box, busy, infer_start_t, completed, shown_completed_once
            nonlocal worker_exc, worker_result

            # Get a fresh frame (dropping buffered ones)
            fr = read_latest_frame(cap)
            if fr is None:
                raise RuntimeError("Camera read failed")

            fr, box, rgb_bytes = prepare_frame_and_crop(fr)

            with state_lock:
                current_frame = fr
                current_box = box

            worker_exc = None
            worker_result = None
            done_event.clear()

            busy = True
            infer_start_t = time.time()
            completed = False
            shown_completed_once = False

            t = threading.Thread(target=infer_worker, args=(rgb_bytes,), daemon=True)
            t.start()

        print("[PC] Press 'q' or ESC to quit.")

        # Kick off first inference immediately
        start_next_inference()

        while True:
            # Pump camera a bit while busy to avoid huge latency when starting next inference
            # (doesn't change displayed frame; only keeps the capture buffer fresh)
            if busy:
                try:
                    cap.grab()
                except Exception:
                    pass

            # If worker finished, update result state (no blocking)
            if busy and done_event.is_set():
                busy = False
                completed = True
                shown_completed_once = False

                with state_lock:
                    if worker_exc is not None:
                        last_err = worker_exc
                    else:
                        last_err = None
                        if worker_result is not None:
                            last_age, last_infer_ms = worker_result

            # Build the display image from the frozen frame
            with state_lock:
                fr = None if current_frame is None else current_frame.copy()
                box = current_box

            if fr is None or box is None:
                # Shouldn't happen after first start, but keep UI alive
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.imshow(win_name, blank)
            else:
                x1, y1, x2, y2 = box
                cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 0), DRAW_CROP_BOX_THICKNESS)

                if last_err:
                    label = f"ERR: {last_err}"
                elif last_age is None:
                    label = "Age: ..."
                else:
                    label = f"Age: {last_age:.1f}"
                    if last_infer_ms is not None:
                        label += f"  ({last_infer_ms} ms)"

                if busy:
                    elapsed = time.time() - infer_start_t
                    label += f"  [inferring {elapsed:.1f}s]"

                draw_label_bottom_left(fr, label)
                cv2.imshow(win_name, fr)

            # Pump UI events frequently (this prevents "Not Responding")
            key = cv2.waitKey(UI_PUMP_DELAY_MS) & 0xFF
            if key == 27 or key == ord("q"):
                break

            # If the window was closed via X button, exit cleanly
            try:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass

            # Ensure result is visible at least once before starting next inference
            if completed and not busy:
                if not shown_completed_once:
                    shown_completed_once = True
                else:
                    # Start next inference: this is what makes "feed fps = inference speed"
                    start_next_inference()

            time.sleep(UI_IDLE_SLEEP_S)

        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
