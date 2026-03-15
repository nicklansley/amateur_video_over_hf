# System Design: Video Over HF SSB

## 1. Recommended Overall Approach

### 1.1 Design Philosophy

The fundamental constraint is information-theoretic: a 2800 Hz channel at typical HF SNR (5–15 dB) offers roughly 1,000–7,000 bits/second of usable throughput after forward error correction. Transmitting 400×300 video at 4–8 fps would require 10,000–20,000+ bps even with state-of-the-art neural compression — far beyond what HF SSB can deliver.

**The recommended approach is therefore a "transmit small, reconstruct big" architecture:**

1. **Transmit** at severely reduced resolution (80×60 to 160×120) and low frame rate (1–4 fps)
2. **Reconstruct** at the receiver using neural super-resolution (upscaling to 400×300) and frame interpolation (boosting to 4–8 fps perceived)
3. **Adapt** the compression strategy based on content type — general scenes use traditional codecs at tiny resolution; face/person content uses keypoint-based generative compression

This mirrors the approach taken by academic systems like [NIER](https://dl.acm.org/doi/10.1145/3718958.3750518) (SIGCOMM 2025, 10–100 Kbps video conferencing) and [FVC-3K2M](https://doi.org/10.1109/TIP.2024.3518100) (2 kbps face video), but adapted for the unique challenges of HF radio: multipath fading, Doppler spread, impulsive noise, and no ARQ (the channel is unidirectional during transmit).

### 1.2 Dual-Mode Video Compression

#### Mode A: General Scene (Traditional Codec + Super-Resolution)

For arbitrary video content (landscapes, equipment, diagrams, whiteboards):

- **Encoder**: Capture at native resolution → downscale to transmit resolution (e.g., 160×120 or 80×60) → encode with H.265 (libx265) or AV1 (SVT-AV1/libaom) at ultra-low bitrate (CRF 50–55)
- **Bitstream**: Packetise encoded NAL units into modem frames
- **Decoder**: Decode H.265/AV1 → apply neural super-resolution (Real-ESRGAN, SwinIR, or similar) to upscale to 400×300 → apply frame interpolation (RIFE or similar) to boost fps

**Why H.265 over AV1 at these resolutions**: Both codecs operate well at ultra-low bitrates. H.265 via libx265 has lower encoding latency and mature real-time encoding support. AV1 via SVT-AV1 offers ~10–15% better compression efficiency at the cost of higher encoding complexity. For a real-time system, H.265 is recommended for the initial prototype; AV1 can be offered as an option for pre-recorded/non-real-time content. At resolutions below 160×120, the efficiency gap between codecs narrows significantly — see discussions on [reddit.com/r/AV1](https://www.reddit.com/r/AV1/comments/1gyiire/is_av1_a_good_choice_for_extremely_low_bitrates/).

**Bitrate budget (Mode A)**:
- Keyframes (I-frames): One every 2–5 seconds, ~500–2000 bytes each at 160×120
- Inter frames (P-frames): ~20–100 bytes each for low-motion content
- Total: 1,000–5,000 bps depending on motion and resolution

#### Mode B: Face/Person (Keypoint-Based Generative Compression)

For video conferencing-style content (talking head, upper body):

Based on the [First Order Motion Model (FOMM)](https://arxiv.org/abs/2003.00196) and its successors, the approach is:

1. **Session initialisation**: Transmit a high-quality reference frame (JPEG, ~3–10 KB) at session start. At 1000 bps, this takes 24–80 seconds — acceptable as a one-time setup cost, analogous to SSTV. The reference frame can be progressively transmitted using scalable JPEG (JPEG 2000) to provide a usable low-quality reference within seconds.
2. **Ongoing transmission**: For each frame, extract 10–15 learned keypoints (2D coordinates + local affine transformations). This is approximately 50–100 bytes per frame.
3. **Receiver reconstruction**: Animate the reference frame using the received keypoints via the pre-trained generative model to produce full frames at the reference resolution (e.g., 256×192 or 320×240).

**Bitrate budget (Mode B)**:
- Per-frame keypoints: 50–100 bytes → at 1000 bps, supports 1–2.5 fps; at 5000 bps, supports 6–12 fps
- Periodic reference frame updates: Every 30–60 seconds when scene changes significantly
- Total: 500–5,000 bps for the ongoing stream

**Both modes share the same OFDM modem and FEC layer** — mode selection is indicated in the packet header.

### 1.3 System Architecture Diagram

```
TX Pipeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌───────────┐
  │  Camera  │──►│  Downscaler  │──►│  Encoder  │──►│ Packetiser│
  │ Capture  │   │ + Preprocess │   │ Mode A/B  │   │ + CRC     │
  └──────────┘   └──────────────┘   └───────────┘   └─────┬─────┘
                                                          │
                     ┌──────────┐   ┌───────────┐         │
                     │ OFDM Mod │◄──│LDPC Encode│◄────────┘
                     │ + BPF    │   │+ Interleave│
                     └────┬─────┘   └───────────┘
                          │
                     Audio Out ──► SSB Radio Mic Input
                   (8000 Hz PCM)

RX Pipeline:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SSB Radio Speaker Out ──► Audio In (8000 Hz PCM)
                                │
                     ┌──────────▼─────┐   ┌───────────────┐
                     │ OFDM Demod     │──►│ LDPC Decode   │
                     │ + Sync + AGC   │   │ + Deinterleave│
                     └────────────────┘   └──────┬────────┘
                                                 │
    ┌──────────┐   ┌────────────────┐   ┌────────▼─────┐
    │ Display  │◄──│ Post-Process   │◄──│  Decoder     │
    │ 400×300  │   │ SR + FrameInterp│  │  Mode A/B    │
    └──────────┘   └────────────────┘   └──────────────┘
```

---

## 2. High-Level Protocol Specification

### 2.1 Protocol Layers

```
┌─────────────────────────────────────┐
│  APPLICATION LAYER                  │
│  Video frames, mode control,        │
│  session management                 │
├─────────────────────────────────────┤
│  FRAMING LAYER                      │
│  Packet structure, CRC-16,          │
│  sequence numbers, timestamps       │
├─────────────────────────────────────┤
│  FEC LAYER                          │
│  LDPC encoding, interleaving,       │
│  burst error protection             │
├─────────────────────────────────────┤
│  MODEM LAYER                        │
│  OFDM modulation/demodulation,      │
│  pilot symbols, synchronisation     │
├─────────────────────────────────────┤
│  PHYSICAL LAYER                     │
│  Audio I/O (8 kHz / 48 kHz PCM),   │
│  bandpass filtering 100–2900 Hz     │
└─────────────────────────────────────┘
```

### 2.2 Modem Layer Specification

The modem is based on the proven [FreeDV/Codec 2 OFDM architecture](https://github.com/drowe67/codec2/blob/main/README_ofdm.md), which has been extensively tested on real HF channels worldwide.

#### Modem Parameters (Default Configuration)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Modulation | OFDM, coherent QPSK | Upgradeable to 16QAM at high SNR |
| Carrier count | 29–31 | Fills ~2100 Hz RF bandwidth |
| Symbol rate | ~50 Hz per carrier | Low rate combats multipath fading |
| Symbol period | 20 ms | |
| Cyclic prefix | 4 ms | Handles up to 4 ms multipath delay spread |
| Pilot symbol rate | 1 in 8 symbols | For channel estimation and coherent demodulation |
| Frame period | 160 ms | Compromise between latency and sync overhead |
| Carrier spacing | ~75 Hz | OFDM subcarrier spacing |
| Audio sample rate | 8000 Hz | Standard soundcard rate; 48000 Hz also supported |
| Frequency range | ~200–2700 Hz | Leaves margin within 100–2900 Hz passband |

#### Adaptive Rate Control

The modem should support multiple operating points, selected based on measured SNR:

| Mode | Modulation | Carriers | Raw Rate (bps) | FEC | Payload (bps) | Min SNR |
|------|-----------|----------|-----------------|-----|---------------|---------|
| RATE_LOW | QPSK | 9 | ~450 | LDPC (2048,1024) | ~250 | -2 dB |
| RATE_MED | QPSK | 29 | ~1900 | LDPC (8192,4096) | ~980 | 3 dB |
| RATE_HIGH | QPSK | 29 | ~3000 | LDPC (504,396) | ~2400 | 8 dB |
| RATE_TURBO | 16QAM | 43 | ~8000 | LDPC (8192,4096) | ~4000 | 14 dB |

These are modelled on the [FreeDV data modes](https://github.com/drowe67/codec2/blob/main/README_data.md) (DATAC0 through DATAC3) and [ARDOP OFDM](https://ardop.groups.io/g/developers/topic/freedv_ofdm_modem/21211944) configurations.

### 2.3 FEC Layer

#### LDPC Forward Error Correction

HF channels exhibit both random errors (thermal noise, interference) and burst errors (fading, static crashes). The FEC scheme must handle both.

**Primary FEC**: LDPC codes from the Codec 2 library, which have been specifically designed and tested for HF channels:

- Rate 1/2 codes: (224,112), (2048,1024), (8192,4096)
- Rate ~3/4 codes: (504,396), (212,158) for higher throughput at good SNR
- Operating point: Corrects up to ~10% raw BER after demodulation ([Codec 2 OFDM README](https://github.com/drowe67/codec2/blob/main/README_ofdm.md))

**Interleaving**: Block interleaver across multiple OFDM frames to spread burst errors:
- Interleave depth: 4–16 frames (640 ms – 2.56 s at 160 ms frame period)
- Deeper interleaving improves burst error resilience at the cost of latency
- Configurable: shallow interleaving for lower latency, deep for difficult channels

**CRC**: 16-bit CRC per application-layer packet for error detection. Packets failing CRC are discarded (no ARQ — this is a unidirectional broadcast stream).

### 2.4 Framing Layer

#### Packet Structure

```
┌──────┬──────┬──────┬──────┬──────┬────────────┬──────┐
│ SYNC │ MODE │ SEQ  │ TS   │ LEN  │  PAYLOAD   │ CRC  │
│ 2B   │ 1B   │ 2B   │ 4B   │ 2B   │  Variable  │ 2B   │
└──────┴──────┴──────┴──────┴──────┴────────────┴──────┘
```

| Field | Size | Description |
|-------|------|-------------|
| SYNC | 2 bytes | Synchronisation pattern (0xAA55) for packet boundary detection |
| MODE | 1 byte | Video mode (0x01=Mode A H.265, 0x02=Mode A AV1, 0x10=Mode B keypoints, 0x20=Mode B reference frame, 0xF0=control) |
| SEQ | 2 bytes | Sequence number (wrapping) for ordering and loss detection |
| TS | 4 bytes | Timestamp in milliseconds (for frame timing and interpolation) |
| LEN | 2 bytes | Payload length in bytes |
| PAYLOAD | Variable | Compressed video data (codec NAL units, keypoints, or reference frame segment) |
| CRC | 2 bytes | CRC-16/CCITT over MODE through PAYLOAD |

**Header overhead**: 13 bytes per packet. At the lowest data rate (~250 bps), this represents significant overhead. For RATE_LOW, multiple frames should be packed into a single modem burst to amortise overhead.

#### Reference Frame Segmentation (Mode B)

Reference frames are segmented into chunks that fit within single modem frames:

```
Reference Frame Packet:
┌──────┬──────┬──────┬──────┬──────┬──────────────────┬──────┐
│ ...  │ MODE │ ...  │ TOTAL│ SEGNO│  JPEG Segment    │ CRC  │
│      │ 0x20 │      │ 1B   │ 1B   │  (fits modem     │      │
│      │      │      │      │      │   frame payload)  │      │
└──────┴──────┴──────┴──────┴──────┴──────────────────┴──────┘
```

If a segment is lost, the receiver can request re-send (if a back-channel exists) or use the partial reference with reduced quality. Progressive JPEG encoding ensures partial data still yields a usable (lower-quality) reference.

### 2.5 Application Layer

#### Session Lifecycle

```
1. HANDSHAKE (optional, if back-channel exists)
   TX sends: SESSION_START (mode, resolution, fps target)
   RX sends: SESSION_ACK
   
2. REFERENCE FRAME TRANSFER (Mode B only)
   TX sends: Reference frame in segments
   RX reconstructs reference frame
   
3. STREAMING
   TX sends: Video packets continuously
   RX decodes and displays with post-processing
   
4. REFERENCE REFRESH (Mode B, periodic)
   TX sends: Updated reference frame segments
   RX cross-fades to new reference
```

#### Mode A Specifics

- H.265 encoder configured for ultra-low latency: `--tune zerolatency --preset ultrafast`
- Constrained bitrate with CBR or VBV-constrained CRF
- Keyframe interval: every 2–5 seconds (I-frame provides re-sync point after errors)
- Encoder input: YUV420P at transmit resolution
- IDR frames marked in packet header for receiver re-sync

#### Mode B Specifics

- Keypoint format: 15 keypoints × (x, y, 4 affine params) = 15 × 6 × float16 = 180 bytes
- Quantised keypoint format: 15 keypoints × (x_u8, y_u8, 4 affine_i8) = 15 × 6 = 90 bytes
- Delta encoding: Transmit differences from previous frame's keypoints → typically 30–50 bytes
- Jacobian matrices truncated to 8-bit fixed point with minimal perceptual loss

---

## 3. Latency and SNR Considerations

### 3.1 End-to-End Latency Budget

| Stage | Latency | Notes |
|-------|---------|-------|
| Camera capture | 33 ms | At 30 fps capture |
| Downscale + preprocess | 5 ms | |
| Video encoding | 10–50 ms | H.265 zerolatency or keypoint extraction |
| Packetisation | 1 ms | |
| LDPC encoding | 2 ms | |
| OFDM modulation | 20 ms | One symbol period |
| Interleaver buffering | 640–2560 ms | **Dominant latency source** |
| Radio TX/RX pipeline | ~50 ms | SSB transceiver processing |
| HF propagation | 5–20 ms | Speed of light, ionospheric reflection |
| OFDM demodulation + sync | 160 ms | One modem frame period |
| LDPC decoding | 5–20 ms | Iterative decoding (50 iterations max) |
| Deinterleaving | (included in interleaver) | |
| Video decoding | 5–20 ms | |
| Super-resolution | 20–100 ms | GPU-dependent; Real-ESRGAN ~30ms on Apple M-series |
| Frame interpolation | 10–30 ms | RIFE network |
| Display | 16 ms | At 60 Hz display |
| **Total (shallow interleave)** | **~1.0–1.5 s** | |
| **Total (deep interleave)** | **~3.0–4.0 s** | |

The interleaver is the dominant latency contributor. For "live" video, shallow interleaving (4 frames, ~640 ms) is recommended, accepting reduced burst-error resilience. For non-real-time applications (recorded video, weather maps), deep interleaving maximises reliability.

### 3.2 SNR and Channel Characteristics

#### Typical HF Channel Impairments

| Impairment | Characteristics | Mitigation |
|------------|----------------|------------|
| Multipath fading | 1–4 ms delay spread, 0.1–2 Hz Doppler | OFDM cyclic prefix (4 ms), pilot-based channel estimation |
| Slow fading | 1–10 second fade duration | Interleaving, adaptive rate |
| Fast fading | >1 Hz Doppler on auroral paths | Higher pilot rate, shorter frames |
| Impulsive noise (static) | Short bursts of high noise | LDPC handles scattered bit errors; interleaving spreads bursts |
| Interference | Adjacent QSOs, powerline noise | Notch filtering pre-demodulation |
| Frequency offset | ±100 Hz typical SSB calibration | Modem acquisition compensates ±60 Hz; manual tuning for larger offsets |

#### SNR Operating Points

Based on [FreeDV modem characterisation](https://github.com/drowe67/codec2/blob/main/README_freedv.md):

| SNR (3 kHz) | Channel Quality | Recommended Mode | Expected Throughput |
|---|---|---|---|
| < -2 dB | Unusable for video | — | — |
| -2 to 2 dB | Marginal | RATE_LOW (DATAC0-like) | ~250 bps |
| 2–5 dB | Moderate | RATE_MED (DATAC1-like) | ~980 bps |
| 5–10 dB | Good | RATE_HIGH | ~2,400 bps |
| 10–15 dB | Very good | RATE_HIGH or RATE_TURBO | 2,400–4,000 bps |
| >15 dB | Excellent | RATE_TURBO (16QAM) | 4,000–7,000 bps |

The system should measure SNR during pilot symbol demodulation and adapt the modem rate accordingly. Rate changes are signalled in the MODE byte of subsequent packets.

### 3.3 Proposed Filtering Techniques

#### Transmitter-Side Filtering

1. **Band-pass filter**: Strict 100–2900 Hz BPF on the audio output to prevent out-of-band energy causing splatter on adjacent frequencies. A 6th-order Butterworth or elliptic filter is suitable.
2. **Pre-emphasis**: Optional pre-emphasis to compensate for SSB transceiver frequency response roll-off at band edges. This should be calibrated per radio.
3. **Crest factor reduction**: OFDM signals have high peak-to-average power ratio (PAPR ~9–10 dB). Clip-and-filter or tone reservation techniques can reduce PAPR by 2–3 dB to avoid transceiver ALC compression.

#### Receiver-Side Filtering

1. **AGC**: Automatic gain control to normalise audio levels from the radio receiver output.
2. **Notch filter**: Adaptive notch filter to suppress narrowband interference (e.g., from adjacent CW or digital signals within the passband).
3. **Noise blanker**: Digital impulse noise blanker to mitigate static crashes.
4. **Frequency tracking**: The OFDM modem's pilot symbols provide continuous frequency offset tracking, but a coarse AFC based on pilot tone detection can accelerate initial acquisition.

---

## 4. Realistic Resolution and Frame Rate Expectations

### 4.1 The 400×300 @ 4–8 fps Target

This target is **not directly achievable** over the channel. Here is why:

- 400×300 = 120,000 pixels per frame
- At 4 fps = 480,000 pixels/second
- Even at an extremely aggressive 0.02 bits per pixel: 480,000 × 0.02 = **9,600 bps** — this exceeds what most HF modems can deliver and is only theoretically possible at SNR > 15 dB with zero overhead

The 0.02 bpp figure is itself unrealistic for useful video — practical H.265 at tiny resolutions operates at 0.05–0.2 bpp for recognisable content.

### 4.2 What Can Actually Be Transmitted

Based on the [Shannon–Hartley theorem](https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem) and real-world modem performance:

#### Mode A (Traditional Codec)

| Throughput | Near-Static Content (0.05 bpp) | Low Motion (0.15 bpp) |
|---|---|---|
| 500 bps @ 1 fps | 115×86 | 66×49 |
| 1000 bps @ 2 fps | 115×86 | 66×49 |
| 2000 bps @ 2 fps | 163×122 | 94×70 |
| 3000 bps @ 2 fps | 200×150 | 115×86 |
| 5000 bps @ 4 fps | 182×136 | 105×78 |
| 7000 bps @ 4 fps | 216×162 | 124×93 |

#### Mode B (Keypoint-Based, 50–100 Bytes/Frame)

| Throughput | Min Keypoints (50 B) | Full Keypoints (100 B) |
|---|---|---|
| 500 bps | 1.2 fps | 0.6 fps |
| 1000 bps | 2.5 fps | 1.2 fps |
| 2000 bps | 5.0 fps | 2.5 fps |
| 5000 bps | 12.5 fps | 6.2 fps |

Mode B is dramatically more bandwidth-efficient because it transmits **motion descriptors, not pixels** — but it requires a pre-shared reference frame and a pre-trained generative model at both ends.

### 4.3 Receiver-Side Enhancement to Reach 400×300

The gap between transmitted resolution and target display resolution is bridged by neural post-processing:

1. **Super-Resolution (SR)**: Models like [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) or SwinIR can upscale 80×60 → 320×240 (4×) or 160×120 → 640×480 (4×) with impressive perceptual quality for faces and natural scenes. On Apple M-series, Real-ESRGAN runs at ~30 ms per frame for small input resolutions. Output is cropped/resampled to 400×300.

2. **Frame Interpolation**: Models like [RIFE](https://github.com/hzwer/ECCV2022-RIFE) (Real-Time Intermediate Flow Estimation) can interpolate between received frames to double or quadruple the perceived frame rate. If 2 fps is received, RIFE can produce 4–8 fps output.

3. **Generative Reconstruction (Mode B)**: The FOMM/NIER-style decoder directly produces frames at the reference resolution, which can then be further enhanced by SR if needed.

**Quality caveats**: SR and interpolation hallucinate detail — the receiver output will look plausible but is not pixel-accurate. Fine text, small numbers, and intricate patterns will be unreliable. This is acceptable for recognising people, scenes, and general situational awareness, which is the primary use case.

---

## 5. Comparison of Modulation and Digital Schemes

### 5.1 OFDM vs. Single-Carrier

| Criterion | OFDM | Single-Carrier (e.g., QPSK + RRC) |
|---|---|---|
| Multipath resilience | Excellent (cyclic prefix) | Requires complex equaliser |
| Spectral efficiency | Good (tight carrier packing) | Moderate |
| Frequency offset sensitivity | Higher | Lower |
| PAPR | High (~10 dB) | Low (~3 dB) |
| Implementation complexity | Moderate (FFT-based) | Lower |
| HF track record | Proven (FreeDV, VARA, Pactor 4) | Proven (STANAG 4285) |
| Recommendation | **Primary choice** | Fallback for very simple implementation |

OFDM is recommended because it naturally handles multipath (the dominant HF impairment) and is well-proven in amateur HF via [FreeDV](https://freedv.org/) and [VARA](https://rosmodem.wordpress.com/). The [Codec 2 OFDM modem](https://github.com/drowe67/codec2/blob/main/README_ofdm.md) is open-source, cross-platform, and already handles the hard problems of acquisition, frequency tracking, and channel estimation.

### 5.2 Modulation Comparison for 2800 Hz Channel

| Modulation | Bits/Symbol | Approx Raw Rate (29 carriers) | Required SNR | Notes |
|---|---|---|---|---|
| BPSK | 1 | ~1,450 bps | -3 dB | Most robust, lowest rate |
| QPSK | 2 | ~2,900 bps | 2 dB | Best balance for HF |
| 8PSK | 3 | ~4,350 bps | 8 dB | Marginal on fading channels |
| 16QAM | 4 | ~5,800 bps | 14 dB | Needs very good channel |
| 64QAM | 6 | ~8,700 bps | 20 dB | Rarely achievable on HF |

The [VARA modem](http://wa8lmf.net/VARA/) uses 52 QAM subcarriers and achieves 7,000 bps on good channels, but is proprietary and Windows-only. For an open implementation, QPSK is the recommended baseline with 16QAM as an adaptive upgrade.

### 5.3 Comparison with Existing Amateur Modes

| Mode | Type | Bandwidth | Rate | Content | Limitations |
|---|---|---|---|---|---|
| SSTV (Martin M1) | Analog FM | 3 kHz | 1 frame/114s | Still images | No motion, analog artifacts |
| Digital SSTV (EasyPal) | OFDM + Reed-Solomon | 3 kHz | 1 frame/30–60s | Still images | Still no motion |
| DATV (DVB-S) | Wide digital TV | 80+ kHz | 18+ kbps | Full video | Far exceeds SSB bandwidth |
| **This project** | OFDM + LDPC + neural | 2.8 kHz | 250–7000 bps | Low-res video | Requires GPU at receiver |

This project fills the gap between SSTV (still images, analog) and DATV (full video, wideband). The 29 MHz DATV experiments by [M0DTS](https://ei7gl.blogspot.com/2022/11/successful-digital-amateur-tv-tests-on.html) using DVB-S at 18 Kbps required 80 kHz bandwidth — our approach compresses into standard 2.8 kHz SSB.

---

## 6. Forward Error Correction Details

### 6.1 LDPC Code Selection

The [Codec 2 project](https://github.com/drowe67/codec2) includes several LDPC codes tested on real HF channels:

| Code | Rate | Payload Bits | Total Bits | Use Case |
|------|------|-------------|------------|----------|
| (112,56) | 1/2 | 56 | 112 | Very short control packets |
| (224,112) | 1/2 | 112 | 224 | Short data, FreeDV 700D voice |
| (256,128) | 1/2 | 128 | 256 | DATAC0 data frames |
| (2048,1024) | 1/2 | 1024 | 2048 | DATAC3 data frames |
| (8192,4096) | 1/2 | 4096 | 8192 | DATAC1 data frames (highest throughput) |
| (504,396) | ~3/4 | 396 | 504 | Higher-rate at good SNR |

**Recommendation for video**: Use (8192,4096) rate 1/2 LDPC for the primary video data stream. This provides excellent error correction at the cost of ~4 seconds per frame at DATAC1 rates. For real-time operation at higher rates, (2048,1024) or (504,396) provide lower latency with slightly less error protection.

### 6.2 Interleaving Strategy

```
Without interleaving:        With block interleaving (depth 8):
                             
Burst error:                 Burst error (spread across 8 frames):
Frame 1: ████████████████    Frame 1: █░░░░░░░█░░░░░░░
Frame 2: ░░░░░░░░░░░░░░░░   Frame 2: █░░░░░░░█░░░░░░░
Frame 3: ░░░░░░░░░░░░░░░░   Frame 3: █░░░░░░░█░░░░░░░
Frame 4: ░░░░░░░░░░░░░░░░   Frame 4: █░░░░░░░█░░░░░░░
                             Frame 5: █░░░░░░░█░░░░░░░
Result: Frame 1 unrecoverable Frame 6: █░░░░░░░█░░░░░░░
                             Frame 7: █░░░░░░░█░░░░░░░
                             Frame 8: █░░░░░░░█░░░░░░░
                             
                             Result: LDPC corrects scattered errors
                                     in all 8 frames
```

### 6.3 Unequal Error Protection (UEP)

Video data has a natural hierarchy of importance:

1. **Critical**: Mode byte, sequence number, keyframe I-slice headers → protect with rate 1/2 LDPC
2. **Important**: Motion vectors, keypoints, P-frame headers → protect with rate 1/2 LDPC
3. **Enhancement**: Residual data, texture detail → protect with rate 3/4 LDPC or no FEC

UEP allows the system to gracefully degrade: under deteriorating conditions, enhancement data is lost first while structural video information is preserved. This is conceptually similar to the NIER system's [layered key-point encoding](https://zhan6841.github.io/assets/pdf/paper/nier-sigcomm25.pdf) approach.

---

## 7. Sources and References

### Modem and HF Digital Modes
- FreeDV / Codec 2 OFDM modem: https://github.com/drowe67/codec2
- FreeDV data modes specification: https://github.com/drowe67/codec2/blob/main/README_data.md
- Codec 2 HF Data Modes Part 2 (Rowetel blog): https://www.rowetel.com/wordpress/?p=7665
- VARA HF modem specifications: https://rosmodem.wordpress.com/
- FreeDATA project: https://github.com/DJ2LS/FreeDATA
- ARDOP OFDM discussion: https://ardop.groups.io/g/developers/topic/freedv_ofdm_modem/21211944
- Zero Retries guide to HF data: https://www.zeroretries.org/p/hf-data

### Video Compression Research
- NIER low-bitrate video conferencing (SIGCOMM 2025): https://dl.acm.org/doi/10.1145/3718958.3750518
- First Order Motion Model (FOMM): https://arxiv.org/abs/2003.00196
- Ultra-low bitrate face video (FVC-3K2M): https://doi.org/10.1109/TIP.2024.3518100
- DCVC-RT real-time neural codec: https://github.com/microsoft/DCVC
- PGen scalable generative face video compression: https://arxiv.org/abs/2502.17085
- Deep image animation for compression: https://arxiv.org/abs/2012.00346
- Narrowband satellite IoT video (AVS3): https://doi.org/10.1109/IoTAAI66837.2025.11213191

### Information Theory
- Shannon–Hartley theorem: https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem

### Amateur Radio TV
- M0DTS 29 MHz DATV across North Atlantic: https://ei7gl.blogspot.com/2022/11/successful-digital-amateur-tv-tests-on.html
- SSTV overview: https://en.wikipedia.org/wiki/Slow-scan_television
