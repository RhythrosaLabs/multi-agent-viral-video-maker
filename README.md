# 🎬 AI Multi-Agent Video Creator

Explore the UI:

https://viral-video-maker.streamlit.app


A powerful and flexible web app built with **Streamlit**, **Replicate**, and **MoviePy** that enables you to generate short educational or cinematic videos automatically. Simply enter a topic, configure settings, and let AI agents handle everything from scriptwriting to video generation and narration.

---

## 🚀 Features

- ✍️ **Script Generation** using Claude-4 Sonnet
- 🎤 **Voiceover Narration** using Minimax Speech O2
- 🎥 **Visual Scene Creation** using Luma Ray 2
- 🎹 **Background Music** using Google Lyria
- 🔊 **Optional Voiceover and Audio Mixing**
- 🖼️ **Customizable Video Styles** (Documentary, Cinematic, Nature, etc.)
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

It is typically best practice to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

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
- Compose and add music
- Concatenate video and audio into one final video
- Let you download the full video and all individual elements (such as script, voiceover, music, etc.)



---

## 📁 File Structure

```plaintext
app.py                  # Main Streamlit app
requirements.txt        # Python dependencies
README.md               # You're here!
