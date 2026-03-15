# Bandwidth Analysis: Video Over HF SSB

## 1. Shannon Capacity of the HF SSB Channel

The [Shannon–Hartley theorem](https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem) gives the theoretical maximum data rate for a noisy channel:

```
C = B × log₂(1 + SNR)
```

Where:
- C = channel capacity (bits/second)
- B = bandwidth (Hz) = 2800 Hz (100–2900 Hz passband)
- SNR = signal-to-noise ratio (linear)

### 1.1 Theoretical Capacity vs. SNR

| SNR (dB) | SNR (Linear) | Shannon Capacity | After Rate 1/2 FEC |
|----------|-------------|------------------|---------------------|
| -2 | 0.63 | 1,976 bps | 988 bps |
| 0 | 1.0 | 2,800 bps | 1,400 bps |
| 5 | 3.16 | 5,760 bps | 2,880 bps |
| 10 | 10.0 | 9,686 bps | 4,843 bps |
| 15 | 31.6 | 14,078 bps | 7,039 bps |
| 20 | 100.0 | 18,643 bps | 9,321 bps |

**Important**: These are theoretical upper bounds. Practical modems achieve 30–70% of Shannon capacity due to synchronisation overhead, pilot symbols, cyclic prefix, guard bands, and non-ideal coding.

### 1.2 Practical Modem Throughput

Real-world HF modems operating within 2800 Hz bandwidth, from [FreeDV documentation](https://github.com/drowe67/codec2/blob/main/README_freedv.md), [VARA specifications](https://rosmodem.wordpress.com/), and [Zero Retries guide](https://www.zeroretries.org/p/hf-data):

| Modem | Bandwidth | Payload Rate | FEC | Min SNR | Open Source |
|-------|-----------|-------------|-----|---------|-------------|
| FreeDV DATAC0 | 500 Hz | 291 bps | LDPC (256,128) | 0 dB | Yes |
| FreeDV DATAC3 | 500 Hz | 321 bps | LDPC (2048,1024) | 0 dB | Yes |
| FreeDV DATAC1 | 1700 Hz | 980 bps | LDPC (8192,4096) | 5 dB | Yes |
| FreeDV DATAC13 | 200 Hz | 64 bps | LDPC (384,128) | -4 dB | Yes |
| FreeDV DATAC14 | 250 Hz | 58 bps | LDPC (112,56) | -2 dB | Yes |
| FreeDATA | 1700 Hz | ~980 bps | LDPC | ~5 dB | Yes |
| ARDOP OFDM 16QAM | 2500 Hz | 4,600 bps | OFDM+FEC | ~10 dB | Yes |
| VARA HF (max) | 2300 Hz | 7,000 bps | Adaptive | 14.5 dB | No ($69) |
| Pactor 4 | 2400 Hz | 10,500 bps | Proprietary | ~15 dB | No (expensive) |

**Takeaway**: Open-source modems deliver 250–4,600 bps depending on bandwidth usage and SNR. Proprietary modems like VARA can reach 7,000 bps. The project should target the open-source Codec 2 OFDM modem stack for maximum flexibility and cross-platform support.

---

## 2. Resolution and Frame Rate Feasibility

### 2.1 Mode A — Traditional Codec (H.265/AV1)

The key parameter is **bits per pixel (bpp)**. At ultra-low resolution and bitrate:

- **Near-static content** (talking head barely moving, still image with minor changes): ~0.05 bpp achievable
- **Low motion** (person gesturing, slow pan): ~0.15 bpp typical
- **Moderate motion** (walking, hand movements): ~0.3 bpp or more

Based on [AV1 low-bitrate discussions](https://www.reddit.com/r/AV1/comments/1gyiire/is_av1_a_good_choice_for_extremely_low_bitrates/), H.265 and AV1 at CRF 50–55 can produce viewable results at ~100 kbps for 360p. Scaling down proportionally:

#### Achievable Resolutions at Various Throughput Levels

**Near-Static Content (0.05 bpp)**:

| Throughput | 1 fps | 2 fps | 4 fps |
|-----------|-------|-------|-------|
| 300 bps | 89×67 | 63×47 | 44×33 |
| 500 bps | 115×86 | 81×61 | 57×43 |
| 1,000 bps | 163×122 | 115×86 | 81×61 |
| 2,000 bps | 230×173 | 163×122 | 115×86 |
| 3,000 bps | 283×212 | 200×150 | 141×106 |
| 5,000 bps | 365×274 | 258×194 | 183×137 |
| 7,000 bps | 432×324 | 306×229 | 216×162 |

**Low Motion (0.15 bpp)**:

| Throughput | 1 fps | 2 fps | 4 fps |
|-----------|-------|-------|-------|
| 300 bps | 52×39 | 37×28 | 26×19 |
| 500 bps | 67×50 | 47×35 | 33×25 |
| 1,000 bps | 94×71 | 67×50 | 47×35 |
| 2,000 bps | 133×100 | 94×71 | 67×50 |
| 3,000 bps | 163×122 | 115×86 | 82×61 |
| 5,000 bps | 211×158 | 149×112 | 105×79 |
| 7,000 bps | 249×187 | 176×132 | 125×94 |

#### Recommended Operating Points for Mode A

| Channel | Throughput | TX Resolution | TX FPS | Strategy |
|---------|-----------|--------------|--------|----------|
| Poor (0 dB) | ~300 bps | 80×60 | 0.5 | Periodic stills, like fast SSTV |
| Moderate (5 dB) | ~980 bps | 80×60 | 2 | Recognisable motion |
| Good (10 dB) | ~3,000 bps | 160×120 | 2 | Useful low-fps video |
| Very good (15 dB) | ~5,000 bps | 160×120 | 4 | Good video with SR |
| Excellent (20 dB) | ~7,000 bps | 200×150 | 4 | Best achievable in 2.8 kHz |

### 2.2 Mode B — Keypoint-Based Generative Compression

Based on research from [FOMM](https://arxiv.org/abs/2003.00196), [NIER](https://dl.acm.org/doi/10.1145/3718958.3750518), and [FVC-3K2M](https://doi.org/10.1109/TIP.2024.3518100):

**Per-frame keypoint data sizes**:
- Full precision: 15 keypoints × (2 coords + 4 affine) × 2 bytes (float16) = **180 bytes**
- Quantised: 15 keypoints × 6 × 1 byte (int8) = **90 bytes**
- Delta-encoded (vs previous frame): ~**30–50 bytes** (most keypoints move very little)
- Minimal (10 keypoints, delta, 4-bit quant): ~**20–30 bytes**

| Throughput | Full (180 B/frame) | Quantised (90 B) | Delta (40 B) | Minimal (25 B) |
|---|---|---|---|---|
| 300 bps | 0.2 fps | 0.4 fps | 0.9 fps | 1.5 fps |
| 500 bps | 0.3 fps | 0.7 fps | 1.6 fps | 2.5 fps |
| 1,000 bps | 0.7 fps | 1.4 fps | 3.1 fps | 5.0 fps |
| 2,000 bps | 1.4 fps | 2.8 fps | 6.2 fps | 10.0 fps |
| 3,000 bps | 2.1 fps | 4.2 fps | 9.4 fps | 15.0 fps |
| 5,000 bps | 3.5 fps | 6.9 fps | 15.6 fps | 25.0 fps |

**Reference frame overhead**: A JPEG-compressed 256×192 reference at quality 60 ≈ 3–5 KB. At 1000 bps, this requires 24–40 seconds to transmit — done once at session start and refreshed periodically.

**Mode B is dramatically more efficient for face/person content** — at just 1000 bps (moderate HF conditions), it can deliver 3–5 fps of keypoint data, which the receiver's generative model reconstructs into full 256×192 frames.

### 2.3 Receiver-Side Enhancement

The transmitted low-resolution or keypoint-generated video is enhanced at the receiver:

| Enhancement | Input | Output | Model | Apple M-series Speed |
|-------------|-------|--------|-------|---------------------|
| Super-resolution 4× | 80×60 | 320×240 | Real-ESRGAN / SwinIR | ~5–15 ms/frame |
| Super-resolution 4× | 160×120 | 640×480 | Real-ESRGAN / SwinIR | ~15–30 ms/frame |
| Frame interpolation 2× | 2 fps | 4 fps | RIFE | ~10–20 ms/pair |
| Frame interpolation 4× | 2 fps | 8 fps | RIFE (2 passes) | ~30–50 ms/pair |
| Generative reconstruction | Keypoints | 256×192 | FOMM-variant | ~20–50 ms/frame |

**Combined output for good conditions (5 dB SNR, Mode B)**:
- Transmitted: keypoints at 3–5 fps
- Generative reconstruction: 256×192 at 3–5 fps
- Frame interpolation: boosted to 6–10 fps
- Optional SR: upscaled to 400×300 or 512×384
- **Result**: Perceptually useful face video at 400×300 @ 6–10 fps

---

## 3. Comparison with Existing Systems

| System | Bandwidth | Data Rate | Content | Latency | Resolution |
|--------|-----------|-----------|---------|---------|------------|
| SSTV Martin M1 | 3 kHz | N/A (analog) | Still image | 114 sec/frame | 320×256 |
| EasyPal (Digital SSTV) | 3 kHz | ~2 kbps | Still image | 30–60 sec/frame | 640×480 |
| DVB-S on 29 MHz ([M0DTS](https://ei7gl.blogspot.com/2022/11/successful-digital-amateur-tv-tests-on.html)) | 80 kHz | 18 kbps | Full video | Real-time | ~320×240 |
| This project (Mode A) | 2.8 kHz | 1–7 kbps | Low-res video | 1–4 sec | 80×60 to 200×150 |
| This project (Mode B) | 2.8 kHz | 1–5 kbps | Face video | 1–3 sec | 256×192 (generated) |
| This project (RX output) | — | — | Enhanced video | 1–4 sec | 400×300 @ 4–8 fps |

The project uniquely combines narrowband digital modulation with neural video compression and receiver-side enhancement to deliver video within standard SSB bandwidth.

---

## 4. Key Constraints and Trade-offs

### 4.1 Bandwidth vs. Resolution vs. Frame Rate

These three compete for the same bits. The optimal trade-off depends on content:

- **Face/person video**: Prioritise frame rate (smooth motion perception) over resolution → Mode B
- **Equipment/scenery**: Prioritise resolution over frame rate → Mode A at 1–2 fps
- **Diagrams/text**: Maximise resolution, minimise frame rate → Mode A at 0.5–1 fps, no SR (SR destroys text)

### 4.2 Latency vs. Error Resilience

- Shallow interleaving (4 frames, ~640 ms): Low latency but vulnerable to fades >500 ms
- Deep interleaving (16 frames, ~2.5 s): Robust but adds several seconds of latency
- **Recommendation**: Default to shallow interleaving for "live" video; offer deep interleaving as a config option

### 4.3 Complexity vs. Quality

- Mode A (H.265 only) is simpler to implement but delivers lower frame rates
- Mode B (generative) delivers much better results for faces but requires ML model deployment
- SR at receiver significantly improves perceived quality but requires GPU
- **Recommendation**: Implement Mode A first, add Mode B and SR as enhancements

### 4.4 Note on Content Limitations

Receiver-side super-resolution and generative reconstruction **hallucinate detail** that is not present in the transmitted data. This means:

- Fine text will be unreadable (SR will produce plausible-looking but incorrect characters)
- Small numbers on instruments/gauges will be unreliable
- Facial identity may drift over long sessions in Mode B if the reference frame is not refreshed
- Fast motion will produce smearing artifacts even with frame interpolation
- The system is best suited for **situational awareness** — recognising who/what is in frame, general activity — not for reading detail

---

## 5. Sources

- Shannon–Hartley theorem: https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem
- FreeDV mode specifications: https://github.com/drowe67/codec2/blob/main/README_freedv.md
- FreeDV data modes: https://github.com/drowe67/codec2/blob/main/README_data.md
- FreeDV OFDM modem: https://github.com/drowe67/codec2/blob/main/README_ofdm.md
- VARA HF specifications: https://rosmodem.wordpress.com/
- VARA HF modem overview: http://wa8lmf.net/VARA/
- ARDOP OFDM performance data: https://ardop.groups.io/g/developers/topic/freedv_ofdm_modem/21211944
- Zero Retries HF data guide: https://www.zeroretries.org/p/hf-data
- AV1 low bitrate discussion: https://www.reddit.com/r/AV1/comments/1gyiire/is_av1_a_good_choice_for_extremely_low_bitrates/
- NIER video conferencing: https://dl.acm.org/doi/10.1145/3718958.3750518
- FOMM image animation: https://arxiv.org/abs/2003.00196
- FVC-3K2M face video compression: https://doi.org/10.1109/TIP.2024.3518100
- M0DTS 29 MHz DATV: https://ei7gl.blogspot.com/2022/11/successful-digital-amateur-tv-tests-on.html
- SSTV overview: https://en.wikipedia.org/wiki/Slow-scan_television
- HF channel NOMA study: https://doi.org/10.1049/cmu2.12643
- Narrowband satellite IoT video (AVS3): https://doi.org/10.1109/IoTAAI66837.2025.11213191
