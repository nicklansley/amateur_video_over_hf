# Amateur Video Over HF

**Real-time low-resolution video transmission over narrowband HF SSB radio channels (100–2900 Hz)**

This project aims to transmit useful video over the ~2800 Hz audio passband of a standard amateur HF SSB transceiver — a channel originally designed for voice. The system encodes video into an audio signal that can be fed directly into the microphone input of any SSB radio, and decoded from the audio output of a receiver.

## The Core Challenge

| Parameter | Value |
|-----------|-------|
| Usable audio bandwidth | 100–2900 Hz (2800 Hz) |
| Shannon capacity at 10 dB SNR | ~9,700 bps theoretical |
| Practical modem throughput | 300–7,000 bps (SNR-dependent) |
| Target output resolution | 400×300 (via receiver-side upscaling) |
| Target output frame rate | 4–8 fps (via receiver-side interpolation) |

**400×300 at 4–8 fps cannot be transmitted directly** over a 2800 Hz channel — even at the most aggressive compression, the raw pixel data far exceeds available bandwidth. Instead, we transmit at much lower resolution and frame rate, and use neural post-processing at the receiver to reconstruct a perceptually useful 400×300 output.

## Documentation

| Document | Contents |
|----------|----------|
| [DESIGN.md](DESIGN.md) | Recommended approach, protocol specification, modulation, FEC, and complete system architecture |
| [ANALYSIS.md](ANALYSIS.md) | Bandwidth calculations, Shannon capacity analysis, resolution/framerate feasibility tables, SNR considerations |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Implementation notes for building a Docker-containerised C++/Rust/Python prototype on NVIDIA DGX Spark and Apple Silicon |

## Quick Summary of Approach

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRANSMITTER (TX)                             │
│                                                                     │
│  Camera ──► Downscale ──► Video Encoder ──► FEC ──► OFDM Modem ──► │
│             (160x120     (H.265/neural   (LDPC    (audio out to     │
│              or 80x60)    keypoints)     rate½)    SSB radio mic)   │
└─────────────────────────────────────────────────────────────────────┘
                              HF Channel
                         (100–2900 Hz SSB)
┌─────────────────────────────────────────────────────────────────────┐
│                        RECEIVER (RX)                                │
│                                                                     │
│  ◄── OFDM Demod ◄── FEC Decode ◄── Video Decode ◄── Upscale ──►   │
│     (audio in from   (LDPC)        (reconstruct     (neural SR      │
│      SSB radio)                     frames)          to 400x300     │
│                                                      + frame        │
│                                                      interpolation) │
└─────────────────────────────────────────────────────────────────────┘
```

The system operates in **two modes** depending on content type:

1. **General Scene Mode** — H.265/AV1 at ultra-low bitrate with tiny resolution, receiver-side neural super-resolution
2. **Face/Person Mode** — Keypoint-based generative compression (inspired by FOMM/NIER research) transmitting only sparse motion descriptors

## Realistic Expectations

| Channel Condition | SNR | Modem Rate | TX Resolution | TX FPS | RX Output (after upscale) |
|---|---|---|---|---|---|
| Poor | 0 dB | ~320 bps | 80×60 | 0.5–1 | Slideshow-quality stills |
| Moderate | 5 dB | ~980 bps | 80×60 | 2–3 | Recognisable motion |
| Good | 10 dB | ~3,000 bps | 160×120 | 2–4 | Usable low-fps video |
| Excellent | 15+ dB | ~5,000–7,000 bps | 160×120 | 4–8 | Good quality with SR upscale |

## Status

🔬 **Research & Design Phase** — This repository contains research, design documentation, and implementation specifications. No runnable code yet. See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the blueprint a coding agent or developer can follow to build a first prototype.

## Licence

MIT

## References

Key sources informing this design are cited inline throughout the documentation. Principal references include:

- [FreeDV / Codec 2 Project](https://github.com/drowe67/codec2) — Open-source OFDM HF modem and codec
- [First Order Motion Model (FOMM)](https://github.com/AliaksandrSiarohin/first-order-model) — Keypoint-based image animation
- [NIER: Practical Neural-enhanced Low-bitrate Video Conferencing](https://dl.acm.org/doi/10.1145/3718958.3750518) — Key-point DIA system achieving 10–100 Kbps video
- [DCVC-RT](https://github.com/microsoft/DCVC) — Real-time neural video codec outperforming H.266/VVC
- [Ultra-Low Bitrate Face Video Compression (FVC-3K2M)](https://doi.org/10.1109/TIP.2024.3518100) — 2 kbps face video via 3D keypoints
