# Implementation Notes: Video Over HF SSB

These notes provide a blueprint for a coding agent or developer to build a first prototype of the system described in [DESIGN.md](DESIGN.md). The target platforms are:

- **NVIDIA DGX Spark** (ARM Linux, NVIDIA GPU)
- **Apple Silicon Mac** (macOS, Metal/ANE acceleration)

The system runs as Docker containers with a shared audio I/O interface.

---

## 1. Technology Stack

### 1.1 Languages and Rationale

| Component | Language | Rationale |
|-----------|----------|-----------|
| OFDM Modem | C (libcodec2) | Proven, real-time, cross-platform, directly reuses FreeDV/Codec 2 |
| FEC (LDPC) | C (libcodec2) | Integrated with modem in Codec 2 |
| Video Codec (H.265) | C/C++ (FFmpeg/libx265) | Mature, hardware-accelerated, cross-platform |
| Video Codec (AV1) | C/C++ (FFmpeg/libaom or SVT-AV1) | Optional, better compression at cost of CPU |
| Keypoint Extraction (Mode B) | Python (PyTorch) | ML model inference, well-supported on both platforms |
| Generative Decoder (Mode B) | Python (PyTorch) | FOMM/NIER models are PyTorch-native |
| Super-Resolution | Python (PyTorch) or C++ (ONNX Runtime) | Real-ESRGAN available in both |
| Frame Interpolation | Python (PyTorch) | RIFE is PyTorch-native |
| Packet Framing / Glue | Rust or Python | Rust for performance; Python for rapid prototyping |
| Audio I/O | C/C++ (PortAudio) or Rust (cpal) | Cross-platform audio capture/playback |
| Configuration / CLI | Python (Click or argparse) | User interface and configuration |

### 1.2 Key Dependencies

```
# Core
libcodec2          >= 1.2    # OFDM modem + LDPC FEC (https://github.com/drowe67/codec2)
ffmpeg/libavcodec  >= 6.0    # H.265/AV1 encoding/decoding
portaudio          >= 19     # Cross-platform audio I/O

# Python ML (for Mode B and receiver enhancement)
torch              >= 2.2    # PyTorch for neural models
torchvision        >= 0.17
onnxruntime        >= 1.17   # Optional: ONNX inference for optimised models
numpy              >= 1.26
opencv-python      >= 4.9    # Image processing

# Pre-trained Models (download at build time)
real-esrgan                  # Super-resolution (https://github.com/xinntao/Real-ESRGAN)
rife                         # Frame interpolation (https://github.com/hzwer/ECCV2022-RIFE)
first-order-model            # Keypoint extraction/generation (https://github.com/AliaksandrSiarohin/first-order-model)
```

---

## 2. Docker Architecture

### 2.1 Container Layout

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                         │
│                                                           │
│  ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   hf-modem      │    │   video-engine              │  │
│  │                  │    │                             │  │
│  │  - OFDM mod/dem │◄──►│  - Camera capture           │  │
│  │  - LDPC enc/dec │    │  - Video encode (H.265/AV1) │  │
│  │  - Audio I/O    │    │  - Keypoint extract (Mode B)│  │
│  │  - BPF + AGC    │    │  - Video decode             │  │
│  │                  │    │  - Super-resolution         │  │
│  │  C + PortAudio  │    │  - Frame interpolation      │  │
│  │                  │    │  - Generative decode (B)    │  │
│  │                  │    │  - Display output           │  │
│  └────────┬─────────┘    │                             │  │
│           │ Unix socket  │  C++/Python + PyTorch       │  │
│           │ or shared    └──────────────┬──────────────┘  │
│           │ memory (IPC)               │                  │
│           └────────────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

Two containers communicate via Unix domain socket or shared memory:
- **hf-modem**: Handles all radio-side processing (modem, FEC, audio I/O)
- **video-engine**: Handles all video-side processing (capture, encode/decode, ML inference)

This separation allows the modem to run at hard-real-time priority while ML inference runs best-effort.

### 2.2 Dockerfile — hf-modem

```dockerfile
FROM ubuntu:24.04 AS modem-build

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libportaudio2 portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Build Codec 2 from source (includes OFDM modem + LDPC)
RUN git clone https://github.com/drowe67/codec2.git /opt/codec2 && \
    cd /opt/codec2 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install

# Build modem application
COPY modem/ /opt/hf-modem/
RUN cd /opt/hf-modem && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

FROM ubuntu:24.04
COPY --from=modem-build /usr/local/lib/ /usr/local/lib/
COPY --from=modem-build /opt/hf-modem/build/hf_modem /usr/local/bin/
RUN ldconfig

# Audio device access
ENV AUDIODEV=default
CMD ["hf_modem", "--config", "/etc/hf-modem/config.toml"]
```

### 2.3 Dockerfile — video-engine

```dockerfile
FROM nvidia/cuda:12.4-runtime-ubuntu24.04 AS base
# For Apple Silicon: FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libx265-dev libavcodec-dev libavformat-dev \
    python3 python3-pip python3-venv \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch (CUDA or CPU/MPS depending on platform)
# For DGX Spark (NVIDIA GPU):
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# For Apple Silicon: pip install torch torchvision (uses MPS backend)

RUN pip install numpy opencv-python-headless onnxruntime realesrgan

# Download pre-trained models
RUN python3 -c "from realesrgan import RealESRGANer; print('Real-ESRGAN ready')"
# TODO: Download RIFE and FOMM model weights

COPY video_engine/ /opt/video-engine/
CMD ["python3", "/opt/video-engine/main.py", "--config", "/etc/video-engine/config.toml"]
```

### 2.4 Docker Compose

```yaml
version: '3.8'
services:
  hf-modem:
    build:
      context: .
      dockerfile: Dockerfile.modem
    volumes:
      - /dev/snd:/dev/snd          # ALSA audio devices
      - ./config:/etc/hf-modem
      - modem-ipc:/tmp/hf-ipc
    devices:
      - /dev/snd
    privileged: false
    cap_add:
      - SYS_NICE                   # For real-time audio priority

  video-engine:
    build:
      context: .
      dockerfile: Dockerfile.video
    volumes:
      - ./config:/etc/video-engine
      - modem-ipc:/tmp/hf-ipc
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia        # For DGX Spark
              count: 1
              capabilities: [gpu]
    environment:
      - DISPLAY=${DISPLAY}
      - PYTORCH_MPS_HIGH_WATERMARK=0  # For Apple Silicon MPS

volumes:
  modem-ipc:
```

### 2.5 Platform-Specific Notes

#### NVIDIA DGX Spark (ARM Linux + NVIDIA GPU)
- Use CUDA 12.x for PyTorch inference
- H.265 hardware encoding via NVENC (if available on Spark's GPU)
- LDPC decoding and OFDM can run on CPU (low compute load)
- GPU primarily for ML inference (SR, keypoint extraction, generative decoding)

#### Apple Silicon (macOS)
- Docker on macOS cannot access audio devices directly. **Two options**:
  1. Run the modem container natively (not in Docker) with direct CoreAudio access, video-engine in Docker
  2. Run both natively, using Docker only for CI/CD and reproducible builds
- PyTorch uses MPS (Metal Performance Shaders) backend for GPU acceleration
- CoreML can be used for optimised SR inference via `coremltools`
- ANE (Apple Neural Engine) acceleration possible for ONNX models via CoreML
- PortAudio works on macOS for audio I/O in native builds
- **Recommended for macOS**: Native build for real-time operation, Docker for CI

---

## 3. Component Implementation Details

### 3.1 OFDM Modem (C, using libcodec2)

The Codec 2 library provides a complete, tested OFDM modem via the `freedv_api`. Key functions from the [FreeDV API](https://github.com/drowe67/codec2/blob/main/README_freedv.md):

```c
#include "freedv_api.h"

// Initialise for data mode
struct freedv *freedv = freedv_open(FREEDV_MODE_DATAC1);

// TX: Encode data into audio samples
int n_nom_modem_samples = freedv_get_n_nom_modem_samples(freedv);
short mod_out[n_nom_modem_samples];

// Prepare payload (up to freedv_get_bits_per_modem_frame(freedv) / 8 bytes)
unsigned char payload[512];  // DATAC1: 510 bytes payload per frame
int payload_len = /* fill with video packet data */;

// Modulate: payload → audio samples
freedv_rawdatatx(freedv, mod_out, payload);
// Write mod_out to audio device (8000 Hz, 16-bit PCM)

// RX: Decode audio samples into data
short demod_in[n_nom_modem_samples];
// Read from audio device into demod_in

unsigned char rx_payload[512];
int rx_len;
int valid = freedv_rawdatarx(freedv, rx_payload, demod_in);
if (valid) {
    // CRC passed, rx_payload contains decoded data
    // Forward to video decoder via IPC
}
```

**Custom modem mode**: For maximum flexibility, use the `FSK_LDPC` mode or build a custom OFDM configuration by modifying `ofdm_internal.h` parameters (carrier count, cyclic prefix, modulation order). See the [OFDM modem README](https://github.com/drowe67/codec2/blob/main/README_ofdm.md) for details on configuring carriers and FEC.

**Audio I/O wrapper** (PortAudio):

```c
#include <portaudio.h>

// Open audio stream at 8000 Hz, 16-bit mono
PaStream *stream;
Pa_OpenDefaultStream(&stream, 1, 1, paInt16, 8000,
                     n_nom_modem_samples, NULL, NULL);
Pa_StartStream(stream);

// TX: Write modulated audio
Pa_WriteStream(stream, mod_out, n_nom_modem_samples);

// RX: Read audio
Pa_ReadStream(stream, demod_in, n_nom_modem_samples);
```

### 3.2 Video Encoder — Mode A (C/C++, FFmpeg)

```c
// Initialise H.265 encoder for ultra-low bitrate
AVCodecContext *enc_ctx;
const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H265);
enc_ctx = avcodec_alloc_context3(codec);

enc_ctx->width = 160;       // Transmit resolution
enc_ctx->height = 120;
enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
enc_ctx->time_base = (AVRational){1, 2};  // 2 fps
enc_ctx->framerate = (AVRational){2, 1};
enc_ctx->bit_rate = 3000;   // 3 kbps target
enc_ctx->rc_max_rate = 5000;
enc_ctx->rc_buffer_size = 10000;
enc_ctx->gop_size = 10;     // I-frame every 5 seconds at 2 fps
enc_ctx->max_b_frames = 0;  // No B-frames for lowest latency
enc_ctx->thread_count = 1;

// Ultra-low latency presets
av_opt_set(enc_ctx->priv_data, "preset", "ultrafast", 0);
av_opt_set(enc_ctx->priv_data, "tune", "zerolatency", 0);
av_opt_set(enc_ctx->priv_data, "x265-params",
           "crf=52:keyint=10:min-keyint=10:bframes=0:repeat-headers=1", 0);

avcodec_open2(enc_ctx, codec, NULL);
```

### 3.3 Video Encoder — Mode B (Python, PyTorch)

Keypoint extraction using a FOMM-variant model:

```python
import torch
import numpy as np
from first_order_model import KeypointDetector  # or custom variant

class KeypointEncoder:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.kp_detector = KeypointDetector().to(self.device)
        self.kp_detector.load_state_dict(torch.load(model_path))
        self.kp_detector.eval()
        self.prev_kp = None
    
    def extract_keypoints(self, frame: np.ndarray) -> bytes:
        """Extract keypoints from a video frame.
        
        Args:
            frame: RGB image, shape (H, W, 3), uint8
            
        Returns:
            Serialised keypoint data (bytes)
        """
        # Preprocess
        x = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            kp = self.kp_detector(x)
        
        # kp['value']: (1, 15, 2) - keypoint coordinates
        # kp['jacobian']: (1, 15, 2, 2) - local affine transforms
        
        coords = kp['value'][0].cpu().numpy()       # (15, 2) float32
        jacobians = kp['jacobian'][0].cpu().numpy()  # (15, 2, 2) float32
        
        # Quantise to int8 for transmission
        coords_q = np.clip(coords * 127, -128, 127).astype(np.int8)      # 30 bytes
        jacobians_q = np.clip(jacobians * 64, -128, 127).astype(np.int8) # 60 bytes
        
        # Delta encoding against previous frame
        if self.prev_kp is not None:
            delta_coords = coords_q - self.prev_kp['coords']
            delta_jac = jacobians_q - self.prev_kp['jacobians']
            self.prev_kp = {'coords': coords_q, 'jacobians': jacobians_q}
            
            # Pack delta-encoded data
            payload = b'\x01'  # Delta flag
            payload += delta_coords.tobytes()   # 30 bytes
            payload += delta_jac.tobytes()       # 60 bytes
            return payload  # 91 bytes total
        else:
            self.prev_kp = {'coords': coords_q, 'jacobians': jacobians_q}
            payload = b'\x00'  # Full keypoint flag
            payload += coords_q.tobytes()
            payload += jacobians_q.tobytes()
            return payload  # 91 bytes total
    
    def encode_reference_frame(self, frame: np.ndarray, quality: int = 60) -> bytes:
        """Encode a reference frame as JPEG for initial transmission."""
        import cv2
        _, jpeg_data = cv2.imencode('.jpg', frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpeg_data.tobytes()
```

### 3.4 Receiver — Super-Resolution (Python)

```python
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2
import numpy as np

class SuperResolver:
    def __init__(self, scale: int = 4, device: str = "cuda"):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=6, num_grow_ch=32, scale=scale)
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path='weights/realesrgan-x4-animevideo.pth',
            model=model,
            tile=0,  # No tiling needed for tiny inputs
            device=device
        )
    
    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a low-resolution frame.
        
        Args:
            frame: BGR image, e.g., 160×120
            
        Returns:
            Upscaled BGR image, e.g., 640×480
        """
        output, _ = self.upsampler.enhance(frame, outscale=4)
        # Resize to exact target if needed
        output = cv2.resize(output, (400, 300), interpolation=cv2.INTER_LANCZOS4)
        return output
```

### 3.5 Receiver — Frame Interpolation (Python)

```python
import torch
from rife_model import RIFE  # https://github.com/hzwer/ECCV2022-RIFE

class FrameInterpolator:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = RIFE().to(self.device)
        self.model.eval()
        self.prev_frame = None
    
    def interpolate(self, frame: np.ndarray, factor: int = 2) -> list:
        """Interpolate between previous frame and current frame.
        
        Args:
            frame: Current decoded frame (BGR, uint8)
            factor: Interpolation factor (2 = double fps, 4 = quadruple)
            
        Returns:
            List of interpolated frames including endpoints
        """
        if self.prev_frame is None:
            self.prev_frame = frame
            return [frame]
        
        frames = [self.prev_frame]
        
        img0 = torch.from_numpy(self.prev_frame).permute(2,0,1).float()/255.0
        img1 = torch.from_numpy(frame).permute(2,0,1).float()/255.0
        img0 = img0.unsqueeze(0).to(self.device)
        img1 = img1.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for i in range(1, factor):
                t = i / factor
                mid = self.model.inference(img0, img1, timestep=t)
                mid_np = (mid[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                frames.append(mid_np)
        
        frames.append(frame)
        self.prev_frame = frame
        return frames
```

### 3.6 IPC Between Containers

The modem and video engine communicate via a Unix domain socket with a simple framing protocol:

```python
# Protocol: [4-byte length (big-endian)] [payload]
import struct
import socket

class IPCClient:
    def __init__(self, socket_path: str = "/tmp/hf-ipc/modem.sock"):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(socket_path)
    
    def send_packet(self, data: bytes):
        header = struct.pack('>I', len(data))
        self.sock.sendall(header + data)
    
    def recv_packet(self) -> bytes:
        header = self._recv_exact(4)
        length = struct.unpack('>I', header)[0]
        return self._recv_exact(length)
    
    def _recv_exact(self, n: int) -> bytes:
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Socket closed")
            data += chunk
        return data
```

---

## 4. Build and Test Plan

### 4.1 Phase 1: Modem Loopback (Week 1–2)

**Goal**: Verify data can be reliably sent and received through the Codec 2 OFDM modem.

1. Build libcodec2 for target platform
2. Write a simple loopback test: encode random data → modulate → add simulated HF channel noise → demodulate → decode → verify
3. Use `freedv_data_raw_tx` and `freedv_data_raw_rx` CLI tools for initial testing
4. Measure actual throughput and packet error rates at various simulated SNR levels
5. Test with real audio loopback (soundcard output → input) to verify timing

```bash
# CLI loopback test with DATAC1 mode
echo "Hello Video" | ./freedv_data_raw_tx --bursts 1 DATAC1 /dev/stdin - 2>/dev/null | \
  ./ch - - --No -20 --fading_dir ../../raw/ | \
  ./freedv_data_raw_rx --framesperburst 1 DATAC1 - /dev/stdout 2>/dev/null
```

### 4.2 Phase 2: Mode A Video Pipeline (Week 3–4)

**Goal**: End-to-end Mode A (H.265) video over simulated channel.

1. Implement camera capture → downscale → H.265 encode pipeline
2. Implement packetiser: split NAL units into modem-frame-sized packets
3. Connect to modem via IPC
4. Implement depacketiser and H.265 decoder on RX side
5. Test with audio file I/O first (no real-time audio), then with PortAudio

**Test matrix**:
| Resolution | FPS | Bitrate | Simulated SNR | Expected Result |
|---|---|---|---|---|
| 80×60 | 2 | 1000 bps | 5 dB | Recognisable faces |
| 80×60 | 1 | 500 bps | 0 dB | Blurry but identifiable |
| 160×120 | 2 | 3000 bps | 10 dB | Good low-res video |
| 160×120 | 4 | 5000 bps | 15 dB | Smooth low-res video |

### 4.3 Phase 3: Receiver Enhancement (Week 5–6)

**Goal**: Add neural super-resolution and frame interpolation.

1. Integrate Real-ESRGAN for 4× upscaling
2. Integrate RIFE for 2–4× frame interpolation
3. Measure inference latency on both target platforms
4. Optimise: try ONNX Runtime, CoreML (macOS), TensorRT (DGX)
5. Validate perceived quality improvement

### 4.4 Phase 4: Mode B Generative Pipeline (Week 7–8)

**Goal**: Keypoint-based face video compression.

1. Train or fine-tune FOMM-variant keypoint detector on face dataset
2. Implement keypoint quantisation and delta encoding
3. Implement reference frame segmentation and progressive JPEG transmission
4. Implement generative decoder at receiver
5. Test end-to-end with face video content

### 4.5 Phase 5: Real Radio Testing (Week 9+)

**Goal**: On-air testing with real HF SSB transceivers.

1. Connect system to SSB transceiver via audio interface (e.g., Digirig, SignaLink)
2. Test on local VHF/UHF first (controlled environment)
3. Graduate to HF bands (e.g., 40m, 20m)
4. Measure real-world performance: packet error rate, throughput, visual quality
5. Tune AGC, notch filter, and adaptive rate parameters based on real conditions

---

## 5. Project File Structure

```
amateur_video_over_hf/
├── README.md
├── DESIGN.md
├── ANALYSIS.md
├── IMPLEMENTATION.md
├── docker-compose.yml
├── Dockerfile.modem
├── Dockerfile.video
├── config/
│   ├── modem.toml              # Modem configuration
│   └── video.toml              # Video engine configuration
├── modem/
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── main.c              # Modem main loop
│   │   ├── ofdm_wrapper.c      # Codec 2 OFDM interface
│   │   ├── audio_io.c          # PortAudio wrapper
│   │   ├── packet_framing.c    # Packet assembly/disassembly
│   │   ├── ipc_server.c        # Unix socket server
│   │   └── bpf.c               # Band-pass filter
│   └── include/
│       └── *.h
├── video_engine/
│   ├── main.py                 # Video engine entry point
│   ├── capture.py              # Camera capture
│   ├── encoder_mode_a.py       # H.265/AV1 encoding
│   ├── encoder_mode_b.py       # Keypoint extraction
│   ├── decoder_mode_a.py       # H.265/AV1 decoding
│   ├── decoder_mode_b.py       # Generative reconstruction
│   ├── super_resolution.py     # Real-ESRGAN upscaling
│   ├── frame_interpolation.py  # RIFE interpolation
│   ├── packetiser.py           # Packet framing
│   ├── ipc_client.py           # Unix socket client
│   └── models/
│       ├── download_models.sh  # Download pre-trained weights
│       └── README.md           # Model sources and licences
├── tests/
│   ├── test_modem_loopback.sh
│   ├── test_video_encode.py
│   ├── test_e2e_simulated.py
│   └── test_sr_quality.py
└── tools/
    ├── channel_simulator.py    # HF channel simulation (Watterson model)
    ├── snr_estimator.py        # Estimate SNR from audio recording
    └── benchmark.py            # Throughput and latency benchmarks
```

---

## 6. Configuration

### 6.1 modem.toml

```toml
[modem]
mode = "DATAC1"              # DATAC0, DATAC1, DATAC3, or custom
sample_rate = 8000           # Audio sample rate (Hz)
interleave_depth = 4         # OFDM frames to interleave (4-16)
adaptive_rate = true         # Enable adaptive rate switching
min_snr_threshold = 2.0      # dB, below this switch to RATE_LOW

[audio]
device = "default"           # PortAudio device name
buffer_size = 1024           # Audio buffer size (samples)

[filter]
bpf_low = 200                # Band-pass filter low cutoff (Hz)
bpf_high = 2700              # Band-pass filter high cutoff (Hz)
bpf_order = 6                # Filter order
agc_enabled = true
notch_enabled = true

[ipc]
socket_path = "/tmp/hf-ipc/modem.sock"
```

### 6.2 video.toml

```toml
[general]
role = "tx"                  # "tx" or "rx" or "both" (loopback)
mode = "A"                   # "A" (traditional) or "B" (generative)

[capture]
device = 0                   # Camera device index
native_fps = 30              # Capture FPS (decimated for TX)

[encoder]
codec = "h265"               # "h265" or "av1"
width = 160                  # Transmit resolution
height = 120
fps = 2                      # Transmit FPS
crf = 52                     # Constant Rate Factor
preset = "ultrafast"
max_bitrate = 3000           # bps

[encoder_mode_b]
model = "fomm"               # Keypoint model variant
num_keypoints = 15
quantisation = "int8"        # "float16" or "int8"
delta_encoding = true
reference_quality = 60       # JPEG quality for reference frame
reference_refresh_interval = 60  # Seconds between reference refreshes

[decoder]
super_resolution = true
sr_model = "realesrgan"
sr_scale = 4                 # Upscale factor
frame_interpolation = true
fi_model = "rife"
fi_factor = 2                # Frame rate multiplication factor
output_width = 400
output_height = 300
output_fps = 4               # Target display FPS

[ipc]
socket_path = "/tmp/hf-ipc/modem.sock"
```

---

## 7. Platform Build Notes

### 7.1 Apple Silicon (macOS 14+)

```bash
# Install dependencies
brew install cmake portaudio ffmpeg libx265

# Build Codec 2
git clone https://github.com/drowe67/codec2.git
cd codec2 && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
make -j$(sysctl -n hw.ncpu)
sudo make install

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision  # Will use MPS backend
pip install numpy opencv-python realesrgan

# Verify MPS acceleration
python3 -c "import torch; print(torch.backends.mps.is_available())"  # Should print True
```

### 7.2 NVIDIA DGX Spark (ARM Linux + NVIDIA GPU)

```bash
# System dependencies
sudo apt install -y cmake build-essential portaudio19-dev \
    ffmpeg libx265-dev libavcodec-dev libavformat-dev

# Build Codec 2
git clone https://github.com/drowe67/codec2.git
cd codec2 && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install && sudo ldconfig

# Python environment (CUDA)
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy opencv-python-headless realesrgan

# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

---

## 8. Testing with Simulated HF Channel

Before real radio testing, use the Codec 2 channel simulator:

```bash
# Simulate MultiPath Poor channel (1Hz Doppler, 2ms delay) at 5dB SNR
./ofdm_mod --in /dev/zero --ldpc --testframes 60 --txbpf | \
  ./ch - - --No -22 -f -10 --mpp | \
  ./ofdm_demod --out /dev/null --testframes --verbose 1 --ldpc

# The 'ch' tool applies:
#   --No: noise level (dBHz)
#   -f: frequency offset (Hz)
#   --mpp: MultiPath Poor model
#   --fast: fast fading (auroral)
```

For end-to-end video testing:
```bash
# Generate test video → encode → modulate → simulate channel → demodulate → decode → display
python3 tools/channel_simulator.py \
    --input test_video.mp4 \
    --snr 5 \
    --channel mpp \
    --modem datac1 \
    --output received_video.mp4
```

---

## 9. Sources

- Codec 2 OFDM modem and data modes: https://github.com/drowe67/codec2
- FreeDV API documentation: https://github.com/drowe67/codec2/blob/main/README_freedv.md
- Codec 2 data mode README: https://github.com/drowe67/codec2/blob/main/README_data.md
- Real-ESRGAN super-resolution: https://github.com/xinntao/Real-ESRGAN
- RIFE frame interpolation: https://github.com/hzwer/ECCV2022-RIFE
- First Order Motion Model: https://github.com/AliaksandrSiarohin/first-order-model
- NIER practical DIA system: https://dl.acm.org/doi/10.1145/3718958.3750518
- DCVC-RT neural video codec: https://github.com/microsoft/DCVC
- PortAudio cross-platform audio: http://www.portaudio.com/
- FFmpeg / libx265: https://ffmpeg.org/
