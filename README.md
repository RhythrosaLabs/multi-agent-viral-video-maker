<div align="center">

# 🎬 AI Multi-Agent Video Creator

**Multi-agent pipeline for creating longform HD videos with music, voiceover, and cinematic visuals**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Replicate](https://img.shields.io/badge/Replicate-000000?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

🌐 **[Live Demo → viral-video-maker.streamlit.app](https://viral-video-maker.streamlit.app)**

</div>

---

A multi-agent video production pipeline. Enter a topic, configure the style, and let specialized AI agents handle every step: scriptwriting, visual scene generation, TTS narration, background music, and final video assembly — all the way to a downloadable MP4.

## ✨ Features

- **Script Generation** — Claude Sonnet writes a structured, narrated script
- **Voiceover Narration** — Minimax Speech O2 TTS with multiple voice options
- **Visual Scene Creation** — Luma Ray 2 generates cinematic video clips per scene
- **Background Music** — Google Lyria generates original background tracks
- **Audio Mixing** — optional voiceover + music blending
- **Customizable Styles** — Documentary, Cinematic, Nature, and more
- **Aspect Ratios** — 16:9, 9:16, 1:1, 4:3
- **Video Lengths** — 10s, 15s, or 20s
- **Camera Motions** — randomized zooms, pans, orbit, and more
- **Downloadable Scripts** — export the generated script separately

## 🚀 Quick Start

```bash
git clone https://github.com/RhythrosaLabs/multi-agent-viral-video-maker.git
cd multi-agent-viral-video-maker
pip install -r requirements.txt
streamlit run app.py
```

Or try the [live demo](https://viral-video-maker.streamlit.app).

## 🛠️ Tech Stack

- **Python + Streamlit** — web app and pipeline orchestration
- **Anthropic Claude** — script generation
- **Replicate** — Luma Ray 2 (video), Google Lyria (music)
- **Minimax Speech** — TTS narration
- **MoviePy** — video assembly and audio mixing

## 🤝 Contributing

PRs welcome. Open an issue first for major changes.

## 📄 License

MIT

## 💛 Support

If this pipeline helps you create content, consider supporting development:

👉 [Donate via PayPal](https://paypal.me/noodlebake) — @noodlebake

---
<div align="center">Made with ❤️ by <a href="https://github.com/RhythrosaLabs">RhythrosaLabs</a></div>
