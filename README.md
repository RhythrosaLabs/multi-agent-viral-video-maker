# 🎬 AI Multi-Agent Video Creator

A powerful and flexible web app built with **Streamlit**, **Replicate**, and **MoviePy** that enables you to generate short educational or cinematic videos automatically. Simply enter a topic, configure settings, and let AI agents handle everything from scriptwriting to video generation and narration.

---

## 🚀 Features

- ✍️ **AI Script Generation** using Replicate (Claude-4 Sonnet)
- 🎤 **Voiceover Narration** with a variety of AI voice options and emotions
- 🎥 **Visual Scene Creation** tailored to your script
- 🖼️ **Customizable Video Styles** (Documentary, Cinematic, Nature, etc.)
- 🔊 **Optional Voiceover and Audio Mixing**
- 📐 **Aspect Ratios:** 16:9, 9:16, 1:1, 4:3
- 🎚️ **Video Lengths:** 10s, 15s, or 20s
- 🕹️ **Camera Motions:** Randomized movements like zooms, pans, orbit, etc.
- 📁 **Downloadable Scripts** and assets
- 🧠 **Multi-Agent Pipeline:** Modular architecture for flexibility and future extensions

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/ai-multi-agent-video-creator.git
cd ai-multi-agent-video-creator
```

2. **Install dependencies**

We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> 📦 Make sure you have `ffmpeg` installed for MoviePy to process videos.

3. **Set up your Replicate API key**

You can get your API key from [replicate.com/account](https://replicate.com/account).

---

## 📄 Usage

Run the app with:

```bash
streamlit run app.py
```

Then:

1. Enter your **Replicate API key**.
2. Input a **video topic** (e.g., `"Why the Earth rotates"`).
3. Customize:
   - Voice
   - Emotion
   - Video style, aspect ratio, quality
   - Duration and looping
   - Camera motion
4. Click **"Generate Video"**.

The app will:
- Write a script using Claude-4 Sonnet
- Generate scenes for each script segment
- Add voiceover narration (optional)
- Concatenate video and audio into one final video
- Let you download the script and generated video

---

## 📦 Dependencies

- [Streamlit](https://streamlit.io/)
- [Replicate](https://replicate.com/)
- [MoviePy](https://zulko.github.io/moviepy/)
- [Requests](https://docs.python-requests.org/en/latest/)
- [FFmpeg](https://ffmpeg.org/) (required for video/audio encoding)

---

## 📁 File Structure

```plaintext
app.py                  # Main Streamlit app
requirements.txt        # Python dependencies
README.md               # You're here!
```

---

## 🧠 Roadmap & Ideas

- Add background music track options
- Upload custom voice files or images
- Extend to long-form or YouTube-ready videos
- Add subtitle/caption generation
- Integrate Stable Diffusion or SDXL for enhanced visuals
- Support multiple languages and TTS providers

---

## ⚠️ Disclaimer

This app uses third-party AI APIs which may generate unpredictable or biased outputs. Always review generated content before sharing or publishing.

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

Developed by [Your Name](https://github.com/yourusername).  
Feel free to open issues, fork, and contribute!
