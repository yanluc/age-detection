#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "esp_heap_caps.h"
#include "esp_system.h"

#include "weights.h"

// -------------------- Model constants --------------------
static constexpr float AGE_MIN = 1.0f;
static constexpr float AGE_MAX = 90.0f;
static constexpr float AGE_RANGE = (AGE_MAX - AGE_MIN);

// -------------------- Protocol --------------------
static const uint8_t MAGIC_PING[4] = {'P','I','N','G'};
static const uint8_t MAGIC_PONG[4] = {'P','O','N','G'};

static const uint8_t MAGIC_REQ[4]  = {'A','G','E','1'};
static const uint8_t MAGIC_ACK[4]  = {'A','C','K','1'};
static const uint8_t MAGIC_OKA[4]  = {'O','K','A','1'};
static const uint8_t MAGIC_RES[4]  = {'R','E','S','1'};
static const uint8_t MAGIC_ERR[4]  = {'E','R','R','1'};

static const uint8_t MAGIC_LOG[4]  = {'L','O','G','1'};

static constexpr uint8_t  PROTO_VER     = 1;
static constexpr uint8_t  FMT_RAW_RGB24 = 0;

static constexpr uint32_t MAX_PAYLOAD   = 3u * 1024u * 1024u;
static constexpr uint32_t CHUNK_HINT    = 2048;

// Error codes
static constexpr uint16_t ERR_BAD_MAGIC      = 1;
static constexpr uint16_t ERR_BAD_HEADER     = 2;
static constexpr uint16_t ERR_BAD_LENGTH     = 3;
static constexpr uint16_t ERR_TOO_LARGE      = 4;
static constexpr uint16_t ERR_OOM            = 5;
static constexpr uint16_t ERR_BAD_CHECKSUM   = 6;
static constexpr uint16_t ERR_INFER_NAN      = 7;
static constexpr uint16_t ERR_INTERNAL       = 100;

// -------------------- RGB LED --------------------
#if defined(ARDUINO_ARCH_ESP32)
extern "C" void neopixelWrite(uint8_t pin, uint8_t red, uint8_t green, uint8_t blue);
#endif

#if defined(RGB_BUILTIN)
static constexpr int LED_PIN = RGB_BUILTIN;
#elif defined(PIN_NEOPIXEL)
static constexpr int LED_PIN = PIN_NEOPIXEL;
#else
static constexpr int LED_PIN = 48;
#endif

static constexpr uint8_t LED_BRIGHT = 24;

static inline uint8_t scale8(uint8_t v) {
  return (uint8_t)((uint16_t)v * LED_BRIGHT / 255u);
}

static void led_set(uint8_t r, uint8_t g, uint8_t b) {
#if defined(ARDUINO_ARCH_ESP32)
  neopixelWrite((uint8_t)LED_PIN, scale8(r), scale8(g), scale8(b));
#else
  (void)r; (void)g; (void)b;
#endif
}

static constexpr uint8_t LED_IDLE  = 0; // blue
static constexpr uint8_t LED_HDR   = 1; // yellow
static constexpr uint8_t LED_RX    = 2; // purple
static constexpr uint8_t LED_PREP  = 3; // cyan
static constexpr uint8_t LED_INFER = 4; // green
static constexpr uint8_t LED_TX    = 5; // white
static constexpr uint8_t LED_ERR   = 6; // red

static void led_state(uint8_t s) {
  switch (s) {
    case LED_IDLE:  led_set(0, 0, 60);   break;
    case LED_HDR:   led_set(80, 60, 0);  break;
    case LED_RX:    led_set(80, 0, 80);  break;
    case LED_PREP:  led_set(0, 80, 80);  break;
    case LED_INFER: led_set(0, 120, 0);  break;
    case LED_TX:    led_set(80, 80, 80); break;
    default:        led_set(140, 0, 0);  break;
  }
}

// -------------------- Utils --------------------
static void* psram_or_heap_malloc(size_t nbytes) {
  if (psramFound()) {
    void* p = heap_caps_malloc(nbytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (p) return p;
  }
  return heap_caps_malloc(nbytes, MALLOC_CAP_8BIT);
}
static void psram_or_heap_free(void* p) { heap_caps_free(p); }

static uint32_t fnv1a32_update(uint32_t h, const uint8_t* data, size_t n) {
  for (size_t i = 0; i < n; ++i) { h ^= (uint32_t)data[i]; h *= 16777619u; }
  return h;
}
static uint32_t fnv1a32(const uint8_t* data, size_t n) {
  return fnv1a32_update(2166136261u, data, n);
}

static void write_u16_le(uint8_t* p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xFF);
  p[1] = (uint8_t)((v >> 8) & 0xFF);
}
static void write_u32_le(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xFF);
  p[1] = (uint8_t)((v >> 8) & 0xFF);
  p[2] = (uint8_t)((v >> 16) & 0xFF);
  p[3] = (uint8_t)((v >> 24) & 0xFF);
}
static uint16_t read_u16_le(const uint8_t* p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static uint32_t read_u32_le(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Inactivity-timeout reader (USB CDC friendly)
static bool read_exact(uint8_t* dst, size_t n, uint32_t inactivity_timeout_ms) {
  uint32_t last_progress = millis();
  size_t got = 0;

  while (got < n) {
    int avail = Serial.available();
    if (avail > 0) {
      int to_read = avail;
      if ((size_t)to_read > (n - got)) to_read = (int)(n - got);
      int r = Serial.readBytes((char*)(dst + got), to_read);
      if (r > 0) { got += (size_t)r; last_progress = millis(); }
    } else {
      if ((millis() - last_progress) > inactivity_timeout_ms) return false;
      delay(1);
    }
    yield();
  }
  return true;
}

// -------------------- LOG1 (NON-BLOCKING) --------------------
// If host isn't reading, we do NOT block. We drop the log instead of deadlocking.
static bool write_all_noblock(const uint8_t* p, size_t n, uint32_t max_wait_ms) {
  uint32_t start = millis();
  size_t off = 0;
  while (off < n) {
    int can = Serial.availableForWrite();
    if (can <= 0) {
      if ((millis() - start) > max_wait_ms) return false;
      delay(1);
      yield();
      continue;
    }
    size_t chunk = (size_t)can;
    if (chunk > (n - off)) chunk = (n - off);
    size_t w = Serial.write(p + off, chunk);
    off += w;
    yield();
  }
  return true;
}

static void log_send(uint8_t level, const char* fmt, ...) {
  char msg[512];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);

  uint16_t len = (uint16_t)strnlen(msg, sizeof(msg));

  uint8_t hdr[4 + 1 + 1 + 2];
  memcpy(hdr + 0, MAGIC_LOG, 4);
  hdr[4] = PROTO_VER;
  hdr[5] = level;
  write_u16_le(hdr + 6, len);

  uint32_t h = 2166136261u;
  h = fnv1a32_update(h, &hdr[4], 1 + 1 + 2);
  h = fnv1a32_update(h, (const uint8_t*)msg, len);

  uint8_t cbuf[4];
  write_u32_le(cbuf, h);

  // total bytes to write
  // If we can't write quickly, just drop the log (do not block protocol)
  const uint32_t MAX_WAIT_MS = 20;

  if (!write_all_noblock(hdr, sizeof(hdr), MAX_WAIT_MS)) return;
  if (!write_all_noblock((const uint8_t*)msg, len, MAX_WAIT_MS)) return;
  (void)write_all_noblock(cbuf, 4, MAX_WAIT_MS);
}

static void log_mem(const char* where) {
  size_t heap_free = heap_caps_get_free_size(MALLOC_CAP_8BIT);
  size_t heap_largest = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
  size_t psram_free = psramFound() ? heap_caps_get_free_size(MALLOC_CAP_SPIRAM) : 0;
  size_t psram_largest = psramFound() ? heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) : 0;

  log_send(0, "[MEM] %s | heap_free=%u heap_largest=%u | psramFound=%d psram_free=%u psram_largest=%u",
           where,
           (unsigned)heap_free, (unsigned)heap_largest,
           (int)psramFound(), (unsigned)psram_free, (unsigned)psram_largest);
}

// -------------------- Binary replies --------------------
static void send_err_frame(uint16_t errcode) {
  led_state(LED_ERR);
  uint8_t pkt[4 + 1 + 1 + 2 + 4];
  memcpy(pkt + 0, MAGIC_ERR, 4);
  pkt[4] = PROTO_VER;
  pkt[5] = 1;
  write_u16_le(pkt + 6, errcode);
  uint32_t c = fnv1a32(pkt + 4, 1 + 1 + 2);
  write_u32_le(pkt + 8, c);
  Serial.write(pkt, sizeof(pkt));
}

static void send_err(uint16_t errcode, const char* reason) {
  // Send ERR first (so host gets it even if logs are dropped)
  send_err_frame(errcode);
  log_send(2, "[ERR] code=%u reason=%s", (unsigned)errcode, reason ? reason : "(null)");
}

static void send_ack() {
  led_state(LED_TX);
  uint8_t pkt[4 + 1 + 1 + 2 + 4 + 4];
  memcpy(pkt + 0, MAGIC_ACK, 4);
  pkt[4] = PROTO_VER;
  pkt[5] = 0;
  write_u16_le(pkt + 6, 0);
  write_u32_le(pkt + 8, CHUNK_HINT);
  uint32_t c = fnv1a32(pkt + 4, 1 + 1 + 2 + 4);
  write_u32_le(pkt + 12, c);
  Serial.write(pkt, sizeof(pkt));
}

static void send_oka(uint32_t received_total) {
  led_state(LED_TX);
  uint8_t pkt[4 + 1 + 1 + 2 + 4 + 4];
  memcpy(pkt + 0, MAGIC_OKA, 4);
  pkt[4] = PROTO_VER;
  pkt[5] = 0;
  write_u16_le(pkt + 6, 0);
  write_u32_le(pkt + 8, received_total);
  uint32_t c = fnv1a32(pkt + 4, 1 + 1 + 2 + 4);
  write_u32_le(pkt + 12, c);
  Serial.write(pkt, sizeof(pkt));
}

static void send_res(float y_norm, float age_years, uint32_t infer_ms) {
  led_state(LED_TX);
  uint8_t pkt[4 + 1 + 1 + 2 + 4 + 4 + 4 + 4];
  memcpy(pkt + 0, MAGIC_RES, 4);
  pkt[4] = PROTO_VER;
  pkt[5] = 0;
  write_u16_le(pkt + 6, 0);
  memcpy(pkt + 8,  &y_norm,    4);
  memcpy(pkt + 12, &age_years, 4);
  write_u32_le(pkt + 16, infer_ms);
  uint32_t c = fnv1a32(pkt + 4, 1 + 1 + 2 + 4 + 4 + 4);
  write_u32_le(pkt + 20, c);
  Serial.write(pkt, sizeof(pkt));
}

// -------------------- NN (same math as before) --------------------
static float sigmoid_fast(float x) {
  if (x > 12.0f) return 0.999994f;
  if (x < -12.0f) return 0.000006f;
  return 1.0f / (1.0f + expf(-x));
}
static inline void relu_inplace(float* x, int n) {
  for (int i = 0; i < n; ++i) if (x[i] < 0.0f) x[i] = 0.0f;
}

// CHW: idx = c*H*W + y*W + x
static void conv2d_3x3_same(const float* in, int inC, int H, int W,
                            const float* w, const float* b, int outC,
                            float* out) {
  const int HW = H * W;

  for (int oc = 0; oc < outC; ++oc) {
    float* outc = out + oc * HW;
    float bias = b[oc];
    for (int i = 0; i < HW; ++i) outc[i] = bias;
  }

  for (int oc = 0; oc < outC; ++oc) {
    float* outc = out + oc * HW;
    const float* w_oc = w + (oc * inC * 9);

    for (int ic = 0; ic < inC; ++ic) {
      const float* in_ic = in + ic * HW;
      const float* k = w_oc + ic * 9;

      const float k00 = k[0], k01 = k[1], k02 = k[2];
      const float k10 = k[3], k11 = k[4], k12 = k[5];
      const float k20 = k[6], k21 = k[7], k22 = k[8];

      for (int y = 1; y < H - 1; ++y) {
        const float* row0 = in_ic + (y - 1) * W;
        const float* row1 = in_ic + y * W;
        const float* row2 = in_ic + (y + 1) * W;
        float* outrow = outc + y * W;
        for (int x = 1; x < W - 1; ++x) {
          float acc = outrow[x];
          acc += row0[x - 1] * k00 + row0[x] * k01 + row0[x + 1] * k02;
          acc += row1[x - 1] * k10 + row1[x] * k11 + row1[x + 1] * k12;
          acc += row2[x - 1] * k20 + row2[x] * k21 + row2[x + 1] * k22;
          outrow[x] = acc;
        }
        if ((y & 31) == 0) yield();
      }

      // borders (bounds-checked)
      {
        int y = 0;
        float* outrow = outc + y * W;
        for (int x = 0; x < W; ++x) {
          float acc = outrow[x];
          for (int ky = 0; ky < 3; ++ky) {
            int iy = y + ky - 1;
            if ((unsigned)iy >= (unsigned)H) continue;
            const float* inrow = in_ic + iy * W;
            for (int kx = 0; kx < 3; ++kx) {
              int ix = x + kx - 1;
              if ((unsigned)ix >= (unsigned)W) continue;
              acc += inrow[ix] * k[ky * 3 + kx];
            }
          }
          outrow[x] = acc;
        }
      }
      {
        int y = H - 1;
        float* outrow = outc + y * W;
        for (int x = 0; x < W; ++x) {
          float acc = outrow[x];
          for (int ky = 0; ky < 3; ++ky) {
            int iy = y + ky - 1;
            if ((unsigned)iy >= (unsigned)H) continue;
            const float* inrow = in_ic + iy * W;
            for (int kx = 0; kx < 3; ++kx) {
              int ix = x + kx - 1;
              if ((unsigned)ix >= (unsigned)W) continue;
              acc += inrow[ix] * k[ky * 3 + kx];
            }
          }
          outrow[x] = acc;
        }
      }
      for (int y = 1; y < H - 1; ++y) {
        float* outrow = outc + y * W;
        // x=0
        {
          int x = 0;
          float acc = outrow[x];
          for (int ky = 0; ky < 3; ++ky) {
            int iy = y + ky - 1;
            const float* inrow = in_ic + iy * W;
            for (int kx = 0; kx < 3; ++kx) {
              int ix = x + kx - 1;
              if ((unsigned)ix >= (unsigned)W) continue;
              acc += inrow[ix] * k[ky * 3 + kx];
            }
          }
          outrow[x] = acc;
        }
        // x=W-1
        {
          int x = W - 1;
          float acc = outrow[x];
          for (int ky = 0; ky < 3; ++ky) {
            int iy = y + ky - 1;
            const float* inrow = in_ic + iy * W;
            for (int kx = 0; kx < 3; ++kx) {
              int ix = x + kx - 1;
              if ((unsigned)ix >= (unsigned)W) continue;
              acc += inrow[ix] * k[ky * 3 + kx];
            }
          }
          outrow[x] = acc;
        }
      }
    }
  }
}

static void maxpool2d_2x2_s2(const float* in, int C, int H, int W, float* out) {
  const int outH = H / 2;
  const int outW = W / 2;
  const int inHW = H * W;
  const int outHW = outH * outW;

  for (int c = 0; c < C; ++c) {
    const float* inCptr = in + c * inHW;
    float* outCptr = out + c * outHW;

    for (int y = 0; y < outH; ++y) {
      int iy = y * 2;
      const float* row0 = inCptr + iy * W;
      const float* row1 = inCptr + (iy + 1) * W;
      float* outrow = outCptr + y * outW;

      for (int x = 0; x < outW; ++x) {
        int ix = x * 2;
        float m = row0[ix];
        float v = row0[ix + 1]; if (v > m) m = v;
        v = row1[ix];           if (v > m) m = v;
        v = row1[ix + 1];       if (v > m) m = v;
        outrow[x] = m;
      }
    }
    yield();
  }
}

static void global_avgpool_HxW(const float* in, int C, int H, int W, float* out_vec) {
  const int HW = H * W;
  for (int c = 0; c < C; ++c) {
    const float* p = in + c * HW;
    double sum = 0.0;
    for (int i = 0; i < HW; ++i) sum += p[i];
    out_vec[c] = (float)(sum / (double)HW);
  }
}

static void dense_local(const float* x, int inF, const float* w, const float* b, int outF, float* y) {
  for (int o = 0; o < outF; ++o) {
    const float* wrow = w + o * inF;
    double acc = (double)b[o];
    for (int i = 0; i < inF; ++i) acc += (double)x[i] * (double)wrow[i];
    y[o] = (float)acc;
  }
}

static void normalize_rgb_u8_to_chw_float(const uint8_t* rgb_hwc, int H, int W, float* out_chw) {
  const float inv127_5 = 1.0f / 127.5f;
  const int HW = H * W;

  float* outR = out_chw + 0 * HW;
  float* outG = out_chw + 1 * HW;
  float* outB = out_chw + 2 * HW;

  for (int i = 0; i < HW; ++i) {
    uint8_t r = rgb_hwc[i * 3 + 0];
    uint8_t g = rgb_hwc[i * 3 + 1];
    uint8_t b = rgb_hwc[i * 3 + 2];
    outR[i] = (float)r * inv127_5 - 1.0f;
    outG[i] = (float)g * inv127_5 - 1.0f;
    outB[i] = (float)b * inv127_5 - 1.0f;
  }
}

static void resize_rgb24_nn_to_200(const uint8_t* src, uint16_t srcW, uint16_t srcH, uint8_t* dst200) {
  for (int y = 0; y < 200; ++y) {
    uint32_t sy = (uint32_t)y * (uint32_t)srcH / 200u;
    const uint8_t* srcRow = src + (size_t)sy * (size_t)srcW * 3u;
    uint8_t* dstRow = dst200 + (size_t)y * 200u * 3u;
    for (int x = 0; x < 200; ++x) {
      uint32_t sx = (uint32_t)x * (uint32_t)srcW / 200u;
      const uint8_t* p = srcRow + (size_t)sx * 3u;
      dstRow[x * 3 + 0] = p[0];
      dstRow[x * 3 + 1] = p[1];
      dstRow[x * 3 + 2] = p[2];
    }
    if ((y & 31) == 0) yield();
  }
}

static void log_image_stats_200(const uint8_t* rgb200) {
  uint8_t rmin=255, gmin=255, bmin=255;
  uint8_t rmax=0,   gmax=0,   bmax=0;
  uint64_t rsum=0,  gsum=0,  bsum=0;

  const int HW = 200 * 200;
  for (int i = 0; i < HW; ++i) {
    uint8_t r = rgb200[i*3+0];
    uint8_t g = rgb200[i*3+1];
    uint8_t b = rgb200[i*3+2];
    if (r < rmin) rmin = r; if (r > rmax) rmax = r;
    if (g < gmin) gmin = g; if (g > gmax) gmax = g;
    if (b < bmin) bmin = b; if (b > bmax) bmax = b;
    rsum += r; gsum += g; bsum += b;
    if ((i & 8191) == 0) yield();
  }

  float rmean = (float)rsum / (float)HW;
  float gmean = (float)gsum / (float)HW;
  float bmean = (float)bsum / (float)HW;

  log_send(0, "[IMG] stats R[min=%u max=%u mean=%.2f] G[min=%u max=%u mean=%.2f] B[min=%u max=%u mean=%.2f]",
           (unsigned)rmin,(unsigned)rmax,rmean,
           (unsigned)gmin,(unsigned)gmax,gmean,
           (unsigned)bmin,(unsigned)bmax,bmean);

  auto pix = [&](int x, int y) {
    int idx = (y*200 + x)*3;
    uint8_t r = rgb200[idx+0], g = rgb200[idx+1], b = rgb200[idx+2];
    log_send(0, "[IMG] pix(%d,%d)=%u %u %u", x, y, (unsigned)r, (unsigned)g, (unsigned)b);
  };
  pix(0,0); pix(199,0); pix(0,199); pix(199,199); pix(100,100);
}

static float run_age_cnn_200x200_rgb(const uint8_t* rgb200_hwc) {
  led_state(LED_INFER);

  const int base = kBaseChannels;
  const int C1 = base;
  const int C2 = base * 2;
  const int C3 = base * 4;
  const int C4 = base * 8;

  const int H0 = 200, W0 = 200;
  const int H1 = 100, W1 = 100;
  const int H2 = 50,  W2 = 50;
  const int H3 = 25,  W3 = 25;
  const int H4 = 12,  W4 = 12;

  const size_t input_floats = (size_t)3 * H0 * W0;
  const size_t max_floats   = (size_t)C1 * H0 * W0;

  log_send(0, "[NN] base=%d C1=%d C2=%d C3=%d C4=%d", base, C1, C2, C3, C4);
  log_mem("before NN alloc");

  float* in_chw = (float*)psram_or_heap_malloc(input_floats * sizeof(float));
  float* bufA   = (float*)psram_or_heap_malloc(max_floats * sizeof(float));
  float* bufB   = (float*)psram_or_heap_malloc(max_floats * sizeof(float));
  float* pooled = (float*)psram_or_heap_malloc((size_t)C4 * sizeof(float));

  if (!in_chw || !bufA || !bufB || !pooled) {
    if (in_chw) psram_or_heap_free(in_chw);
    if (bufA)   psram_or_heap_free(bufA);
    if (bufB)   psram_or_heap_free(bufB);
    if (pooled) psram_or_heap_free(pooled);
    log_mem("NN alloc failed");
    return NAN;
  }

  log_mem("after NN alloc");

  normalize_rgb_u8_to_chw_float(rgb200_hwc, H0, W0, in_chw);

  conv2d_3x3_same(in_chw, 3, H0, W0, CONV0_W, CONV0_B, C1, bufA);
  relu_inplace(bufA, C1 * H0 * W0);
  conv2d_3x3_same(bufA, C1, H0, W0, CONV1_W, CONV1_B, C1, bufB);
  relu_inplace(bufB, C1 * H0 * W0);
  maxpool2d_2x2_s2(bufB, C1, H0, W0, bufA);

  conv2d_3x3_same(bufA, C1, H1, W1, CONV2_W, CONV2_B, C2, bufB);
  relu_inplace(bufB, C2 * H1 * W1);
  conv2d_3x3_same(bufB, C2, H1, W1, CONV3_W, CONV3_B, C2, bufA);
  relu_inplace(bufA, C2 * H1 * W1);
  maxpool2d_2x2_s2(bufA, C2, H1, W1, bufB);

  conv2d_3x3_same(bufB, C2, H2, W2, CONV4_W, CONV4_B, C3, bufA);
  relu_inplace(bufA, C3 * H2 * W2);
  conv2d_3x3_same(bufA, C3, H2, W2, CONV5_W, CONV5_B, C3, bufB);
  relu_inplace(bufB, C3 * H2 * W2);
  maxpool2d_2x2_s2(bufB, C3, H2, W2, bufA);

  conv2d_3x3_same(bufA, C3, H3, W3, CONV6_W, CONV6_B, C4, bufB);
  relu_inplace(bufB, C4 * H3 * W3);
  conv2d_3x3_same(bufB, C4, H3, W3, CONV7_W, CONV7_B, C4, bufA);
  relu_inplace(bufA, C4 * H3 * W3);
  maxpool2d_2x2_s2(bufA, C4, H3, W3, bufB);

  global_avgpool_HxW(bufB, C4, H4, W4, pooled);

  float fc0[128];
  float fc1[64];
  float fc2[1];

  dense_local(pooled, C4, FC0_W, FC0_B, 128, fc0);
  relu_inplace(fc0, 128);
  dense_local(fc0, 128, FC1_W, FC1_B, 64, fc1);
  relu_inplace(fc1, 64);
  dense_local(fc1, 64, FC2_W, FC2_B, 1, fc2);

  float y = sigmoid_fast(fc2[0]);

  psram_or_heap_free(in_chw);
  psram_or_heap_free(bufA);
  psram_or_heap_free(bufB);
  psram_or_heap_free(pooled);

  log_mem("after NN free");
  return y;
}

// -------------------- AGE request --------------------
static bool process_age_request_after_magic() {
  led_state(LED_HDR);
  log_send(0, "[AGE] start");
  log_mem("start AGE");

  uint8_t hdr[1 + 1 + 2 + 2 + 4 + 4];
  if (!read_exact(hdr, sizeof(hdr), 3000)) {
    send_err(ERR_BAD_HEADER, "timeout reading header");
    return true;
  }

  const uint8_t  ver = hdr[0];
  const uint8_t  fmt = hdr[1];
  const uint16_t w   = read_u16_le(hdr + 2);
  const uint16_t h   = read_u16_le(hdr + 4);
  const uint32_t payload_len   = read_u32_le(hdr + 6);
  const uint32_t want_checksum = read_u32_le(hdr + 10);

  log_send(0, "[AGE] header ver=%u fmt=%u w=%u h=%u payload_len=%u want_checksum=0x%08X",
           (unsigned)ver, (unsigned)fmt, (unsigned)w, (unsigned)h,
           (unsigned)payload_len, (unsigned)want_checksum);

  if (ver != PROTO_VER || fmt != FMT_RAW_RGB24 || w == 0 || h == 0) {
    send_err(ERR_BAD_HEADER, "bad ver/fmt/w/h");
    return true;
  }

  const uint32_t expected_len = (uint32_t)w * (uint32_t)h * 3u;
  if (payload_len != expected_len) {
    send_err(ERR_BAD_LENGTH, "payload_len != w*h*3");
    return true;
  }

  if (payload_len > MAX_PAYLOAD) {
    send_err(ERR_TOO_LARGE, "payload too large");
    return true;
  }

  log_mem("before payload alloc");
  uint8_t* payload = (uint8_t*)psram_or_heap_malloc(payload_len);
  if (!payload) {
    log_mem("payload alloc failed");
    send_err(ERR_OOM, "payload malloc failed");
    return true;
  }
  log_mem("after payload alloc");

  send_ack();
  send_oka(0);            // must be immediate, host expects it
  led_state(LED_RX);

  uint32_t received = 0;
  uint32_t running_checksum = 2166136261u;
  bool sent_any_rx_log = false;

  while (true) {
    uint8_t len2[2];
    if (!read_exact(len2, 2, 15000)) {
      psram_or_heap_free(payload);
      send_err(ERR_BAD_LENGTH, "timeout waiting chunk_len");
      return true;
    }
    uint16_t clen = read_u16_le(len2);

    if (clen == 0) break;

    if (clen > (uint16_t)CHUNK_HINT) {
      psram_or_heap_free(payload);
      send_err(ERR_BAD_LENGTH, "chunk too big");
      return true;
    }
    if ((uint32_t)clen > (payload_len - received)) {
      psram_or_heap_free(payload);
      send_err(ERR_BAD_LENGTH, "chunk exceeds remaining");
      return true;
    }

    if (!read_exact(payload + received, clen, 15000)) {
      psram_or_heap_free(payload);
      send_err(ERR_BAD_LENGTH, "timeout reading chunk bytes");
      return true;
    }

    running_checksum = fnv1a32_update(running_checksum, payload + received, clen);
    received += (uint32_t)clen;

    // IMPORTANT: OKA first, logs after (logs must never delay OKA)
    send_oka(received);
    led_state(LED_RX);

    if (!sent_any_rx_log) {
      sent_any_rx_log = true;
      log_send(0, "[RX] first chunk ok, total=%u/%u", (unsigned)received, (unsigned)payload_len);
    }

    if ((received & 16383u) == 0) {
      log_send(0, "[RX] progress %u/%u", (unsigned)received, (unsigned)payload_len);
      log_mem("during RX");
    }
  }

  if (received != payload_len) {
    psram_or_heap_free(payload);
    send_err(ERR_BAD_LENGTH, "received != payload_len");
    return true;
  }

  log_send(0, "[RX] checksum running=0x%08X expected=0x%08X", (unsigned)running_checksum, (unsigned)want_checksum);

  if (running_checksum != want_checksum) {
    psram_or_heap_free(payload);
    send_err(ERR_BAD_CHECKSUM, "checksum mismatch");
    return true;
  }

  led_state(LED_PREP);
  log_send(0, "[PREP] allocating rgb200");
  log_mem("before rgb200 alloc");

  uint8_t* rgb200 = (uint8_t*)psram_or_heap_malloc(200u * 200u * 3u);
  if (!rgb200) {
    psram_or_heap_free(payload);
    send_err(ERR_OOM, "rgb200 malloc failed");
    return true;
  }
  log_mem("after rgb200 alloc");

  memcpy(rgb200, payload, 200u * 200u * 3u);
  psram_or_heap_free(payload);
  log_mem("after payload free");

  log_image_stats_200(rgb200);

  uint32_t t0 = millis();
  float y_norm = run_age_cnn_200x200_rgb(rgb200);
  uint32_t t1 = millis();

  psram_or_heap_free(rgb200);
  log_mem("after rgb200 free");

  if (!isfinite(y_norm)) {
    send_err(ERR_INFER_NAN, "y_norm not finite");
    return true;
  }

  float age_years = y_norm * AGE_RANGE + AGE_MIN;
  log_send(0, "[RES] y_norm=%.7f age_years=%.3f infer_ms=%u", y_norm, age_years, (unsigned)(t1 - t0));

  send_res(y_norm, age_years, (uint32_t)(t1 - t0));
  led_state(LED_IDLE);
  log_send(0, "[AGE] done");
  return true;
}

// -------------------- Stream scanner (resync) --------------------
static bool scan_and_process_one() {
  if (Serial.available() <= 0) return false;

  static uint8_t win[4] = {0,0,0,0};
  static uint8_t filled = 0;

  int b = Serial.read();
  if (b < 0) return false;

  led_state(LED_HDR);

  if (filled < 4) {
    win[filled++] = (uint8_t)b;
    if (filled < 4) return true;
  } else {
    win[0] = win[1];
    win[1] = win[2];
    win[2] = win[3];
    win[3] = (uint8_t)b;
  }

  if (memcmp(win, MAGIC_PING, 4) == 0) {
    led_state(LED_TX);
    Serial.write(MAGIC_PONG, 4);
    led_state(LED_IDLE);
    filled = 0;
    return true;
  }

  if (memcmp(win, MAGIC_REQ, 4) == 0) {
    filled = 0;
    return process_age_request_after_magic();
  }

  yield();
  return true;
}

void setup() {
  Serial.begin(921600);
  Serial.setTimeout(10);
  // Bigger buffers help CDC robustness
  Serial.setRxBufferSize(8192);
  Serial.setTxBufferSize(8192);

  pinMode(LED_PIN, OUTPUT);
  led_state(LED_IDLE);

  log_send(0, "[BOOT] sdk=%s", ESP.getSdkVersion());
  log_send(0, "[BOOT] psramFound=%d", (int)psramFound());
  log_mem("boot");
}

void loop() {
  while (scan_and_process_one()) {}
  delay(1);
}