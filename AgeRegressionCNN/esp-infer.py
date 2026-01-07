import time
import struct
import serial
from PIL import Image

# ---------------- CONFIG (hard-coded) ----------------
PORT = "/dev/ttyACM0"
BAUD = 921600
IMAGE_PATH = "face.png"
OPEN_DELAY_S = 1.0

TIMEOUT_S = 10.0
TIMEOUT_CHUNK_S = 25.0     # IMPORTANT: > ESP inactivity timeout (15s)
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

def ping(ser: serial.Serial):
    ser.reset_input_buffer()
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

    level_name = {0:"I",1:"W",2:"E"}.get(lvl, str(lvl))
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

def load_image_as_raw_rgb24(path: str):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    rgb = img.tobytes()
    return rgb, w, h

def main():
    print(f"[PC] Loading image: {IMAGE_PATH}")
    rgb, w, h = load_image_as_raw_rgb24(IMAGE_PATH)
    print(f"[PC] Image size: {w}x{h}, payload={len(rgb)} bytes")

    checksum = fnv1a32(rgb)
    header = MAGIC_REQ + struct.pack("<BBHHII", PROTO_VER, FMT_RAW_RGB24, w, h, len(rgb), checksum)

    print(f"[PC] Opening serial: {PORT} @ {BAUD}")
    with serial.Serial(
        PORT, BAUD,
        timeout=0.1,
        write_timeout=None,
        xonxoff=False, rtscts=False, dsrdtr=False
    ) as ser:
        time.sleep(OPEN_DELAY_S)

        print("[PC] PING...")
        pong = ping(ser)
        print(f"[PC] Got: {pong!r}")
        if pong != b"PONG":
            print("[PC] Not PONG -> wrong firmware/port.")
            return

        ser.reset_input_buffer()

        print("[PC] Sending AGE header...")
        ser.write(header)
        ser.flush()

        print("[PC] Waiting for ACK1...")
        m = recv_magic_skipping_logs(ser, TIMEOUT_S)
        if m == MAGIC_ERR:
            err = recv_err(ser, TIMEOUT_S)
            print(f"[PC] ESP ERR before payload, errcode={err}")
            return
        if m != MAGIC_ACK:
            raise RuntimeError(f"Expected ACK1, got {m!r}")

        chunk_hint = recv_ack(ser, TIMEOUT_S)
        if chunk_hint < 256 or chunk_hint > 65535:
            chunk_hint = 2048
        print(f"[PC] ACK ok, chunk_hint={chunk_hint}")

        # Expect immediate OKA(0)
        m = recv_magic_skipping_logs(ser, TIMEOUT_S)
        if m == MAGIC_ERR:
            err = recv_err(ser, TIMEOUT_S)
            print(f"[PC] ESP ERR after ACK, errcode={err}")
            return
        if m != MAGIC_OKA:
            raise RuntimeError(f"Expected OKA1 right after ACK, got {m!r}")
        got0 = recv_oka(ser, TIMEOUT_S)
        print(f"[PC] OKA initial total={got0}")

        print("[PC] Sending payload with per-chunk OK...")
        mv = memoryview(rgb)
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
                print(f"[PC] ESP ERR during payload, errcode={err}")
                return
            if m != MAGIC_OKA:
                raise RuntimeError(f"Expected OKA1, got {m!r}")

            got_total = recv_oka(ser, TIMEOUT_S)
            off = end
            if got_total != off:
                raise RuntimeError(f"Device total mismatch: device={got_total} host={off}")

        ser.write(struct.pack("<H", 0))
        ser.flush()

        print("[PC] Payload done. Waiting for result...")
        m = recv_magic_skipping_logs(ser, TIMEOUT_INFER_S)
        if m == MAGIC_ERR:
            err = recv_err(ser, TIMEOUT_S)
            print(f"[PC] ESP ERR after payload, errcode={err}")
            return
        if m != MAGIC_RES:
            raise RuntimeError(f"Expected RES1, got {m!r}")

        y_norm, age_years, infer_ms = recv_res(ser, TIMEOUT_INFER_S)
        print(f"[PC] y_norm    = {y_norm:.6f}")
        print(f"[PC] age_years = {age_years:.3f}")
        print(f"[PC] infer_ms  = {infer_ms}")
        print(f"[PC] infer_s  = {infer_ms/1000}")

if __name__ == "__main__":
    main()
