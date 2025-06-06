import streamlit as st
import replicate
import tempfile
import os
import requests
import re
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeAudioClip,
    concatenate_audioclips,
)
import numpy as np

# Set Streamlit page configuration for a wider layout and custom title
st.set_page_config(layout="wide", page_title="AI Multi-Agent Video Creator")

# --- Header Section ---
st.title("🎬 AI Multi-Agent Video Creator")
st.markdown("Unleash your creativity! Generate dynamic videos with AI-powered scriptwriting, voiceovers, visuals, and music.")

# --- About Section ---
st.sidebar.header("About This App")
st.sidebar.info(
    """
    This application leverages multiple AI models from Replicate to create videos from a simple text prompt.
    Here's a breakdown of the process:

    1.  **Script Generation (Text Model):** An AI model (e.g., Claude, GPT) crafts a voiceover script based on your topic and chosen video category.
    2.  **Voiceover Generation (Speech Model):** The script is converted into natural-sounding speech using a Text-to-Speech model.
    3.  **Visual Generation (Video Model):** For each segment of the script, a video generation model creates a short video clip based on a visual prompt derived from your topic and style.
    4.  **Music Generation (Music Model):** A background music track is generated to complement the video's theme.
    5.  **Video Assembly:** All generated audio and video clips are seamlessly combined, synchronized, and rendered into a final video.

    **Note:** This application requires a Replicate API key to function.
    """
)

# Helper function to sanitize string for API calls
def sanitize_for_api(text_string):
    """Encodes a string to ASCII, ignoring errors, then decodes back to string.
    This removes any non-ASCII characters that might cause UnicodeEncodeError."""
    if isinstance(text_string, str):
        return text_string.encode('ascii', 'ignore').decode('ascii')
    return str(text_string).encode('ascii', 'ignore').decode('ascii')

# --- Model Configurations and Pricing ---
# Prices are approximate and based on Replicate's public pricing as of latest search.
# Prices are typically per million tokens for text, per second or per run for others.
MODEL_CONFIGS = {
    "text": {
        "anthropic/claude-4-sonnet": {
            "name": "Claude 4 Sonnet",
            "model_id": "anthropic/claude-4-sonnet",
            "input_cost_per_million_tokens": 3.00,
            "output_cost_per_million_tokens": 15.00,
            "avg_output_tokens_per_run": 700, # Estimated avg output tokens for a script
            "parameters": {}
        },
        "openai/gpt-4.1": {
            "name": "GPT-4.1",
            "model_id": "openai/gpt-4.1",
            "input_cost_per_million_tokens": 2.00,
            "output_cost_per_million_tokens": 8.00,
            "avg_output_tokens_per_run": 700,
            "parameters": {}
        },
        "openai/gpt-4.1-nano": {
            "name": "GPT-4.1 Nano",
            "model_id": "openai/gpt-4.1-nano",
            "input_cost_per_million_tokens": 0.10,
            "output_cost_per_million_tokens": 0.40,
            "avg_output_tokens_per_run": 700,
            "parameters": {}
        },
        "meta-llama/llama-3-8b-instruct": {
            "name": "Llama 3 8B Instruct",
            "model_id": "meta-llama/llama-3-8b-instruct",
            "input_cost_per_million_tokens": 0.05,
            "output_cost_per_million_tokens": 0.25,
            "avg_output_tokens_per_run": 700,
            "parameters": {}
        },
        "anthropic/claude-opus-4": {
            "name": "Claude Opus 4",
            "model_id": "anthropic/claude-opus-4",
            "input_cost_per_million_tokens": 15.00,
            "output_cost_per_million_tokens": 75.00,
            "avg_output_tokens_per_run": 700,
            "parameters": {}
        },
        "anthropic/claude-haiku-3.5": {
            "name": "Claude Haiku 3.5",
            "model_id": "anthropic/claude-haiku-3.5",
            "input_cost_per_million_tokens": 0.80,
            "output_cost_per_million_tokens": 4.00,
            "avg_output_tokens_per_run": 700,
            "parameters": {}
        },
        "meta/llama-4-maverick-instruct": {
            "name": "Llama 4 Maverick Instruct",
            "model_id": "meta/llama-4-maverick-instruct",
            "input_cost_per_million_tokens": 0.25,
            "output_cost_per_million_tokens": 0.95,
            "avg_output_tokens_per_run": 700,
            "parameters": {}
        },
    },
    "speech": {
        "minimax/speech-02-turbo": {
            "name": "MiniMax Speech-02-Turbo",
            "model_id": "minimax/speech-02-turbo",
            # Estimated cost per second based on similar models: $0.00038 for ~2s generation
            "cost_per_second": 0.00019,
            "parameters": {
                "speed": {"type": "float", "default": 1.1, "min": 0.5, "max": 2.0, "step": 0.1},
                "pitch": {"type": "float", "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.5},
                "volume": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
            }
        },
        "jaaari/kokoro-82m": {
            "name": "Kokoro-82M",
            "model_id": "jaaari/kokoro-82m",
            "cost_per_run": 0.00038, # Direct cost per run found
            "parameters": {
                "speed": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1},
            }
        },
        "minimax/speech-02-hd": {
            "name": "MiniMax Speech-02-HD",
            "model_id": "minimax/speech-02-hd",
            # Assuming similar character/second rate as turbo, and 1M chars = $50.00
            "cost_per_second": 0.000625, # $50 / 1M chars * 12.5 chars/sec (approx)
            "parameters": {
                "speed": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1},
                "pitch": {"type": "float", "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.5},
                "volume": {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
            }
        },
        "replicate/openvoice-v2": {
            "name": "OpenVoice v2",
            "model_id": "replicate/openvoice-v2",
            "cost_per_run": 0.00833, # $8.33 / 1000 runs (approx)
            "parameters": {
                "speed": {"type": "float", "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1},
                # OpenVoice also has language parameter, but it's complex with codes. Skipping for now.
            }
        },
    },
    "video": {
        "luma/ray-flash-2-540p": {
            "name": "Luma Ray Flash 2 (540p)",
            "model_id": "luma/ray-flash-2-540p",
            "cost_per_video_segment": 0.45, # Cost per 5s video segment based on luma/ray pricing
            "parameters": {
                "num_frames": {"type": "int", "default": 120, "min": 120, "max": 200, "step": 10}, # Moved here
                "fps": {"type": "int", "default": 24, "min": 10, "max": 30, "step": 1},
                "guidance": {"type": "float", "default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1},
                "num_inference_steps": {"type": "int", "default": 30, "min": 10, "max": 100, "step": 5}
            }
        },
        "google/veo-3": {
            "name": "Google Veo 3",
            "model_id": "google/veo-3",
            "cost_per_second": 0.75, # Direct cost per second
            "parameters": {
                "fps": {"type": "int", "default": 24, "min": 10, "max": 30, "step": 1},
                "quality": {"type": "int", "default": 10, "min": 1, "max": 10, "step": 1},
            }
        },
        "minimax/video-01-director": {
            "name": "Minimax Video-01-Director",
            "model_id": "minimax/video-01-director",
            "cost_per_video_segment": 0.50, # Direct cost per video
            "parameters": {
                "fps": {"type": "int", "default": 24, "min": 10, "max": 30, "step": 1},
                "num_inference_steps": {"type": "int", "default": 50, "min": 20, "max": 100, "step": 5},
            }
        },
        "google/veo-2": {
            "name": "Google Veo 2",
            "model_id": "google/veo-2",
            "cost_per_second": 0.50, # Direct cost per second
            "parameters": {
                "fps": {"type": "int", "default": 24, "min": 10, "max": 30, "step": 1},
                "quality": {"type": "int", "default": 7, "min": 1, "max": 10, "step": 1},
            }
        },
        "wan-video/wan-2.1-1.3b": {
            "name": "WAN 2.1 1.3B",
            "model_id": "wan-video/wan-2.1-1.3b",
            "cost_per_video_segment": 0.20, # Direct cost per video (5s video)
            "parameters": {
                # No specific parameters mentioned on Replicate for WAN 2.1 1.3B beyond prompt
            }
        },
    },
    "music": {
        "google/lyria-2": {
            "name": "Google Lyria 2",
            "model_id": "google/lyria-2",
            "cost_per_second": 0.002, # Direct pricing from search
            "parameters": {}
        },
        "meta/musicgen": {
            "name": "Meta MusicGen (Melody)",
            "model_id": "meta/musicgen",
            "cost_per_run": 0.085, # Direct cost per run
            "parameters": {
                "duration": {"type": "float", "default": 10.0, "min": 1.0, "max": 30.0, "step": 1.0},
                "model_version": {"type": "str", "default": "melody", "options": ["melody", "large", "stereo-melody-large", "stereo-large"]}, # Added more options
            }
        },
        "lucataco/ace-step": {
            "name": "ACE-Step",
            "model_id": "lucataco/ace-step",
            "cost_per_run": 0.085, # Estimating similar to MusicGen, as no direct pricing found
            "parameters": {
                # No explicit parameters mentioned on Replicate for ACE-Step beyond prompt
            }
        },
    }
}

# --- User Inputs ---
st.header("1. Your Video Idea")
replicate_api_key = st.text_input("🔑 **Enter your Replicate API Key**", type="password", help="You can find your API key at replicate.com/account.")
video_topic_raw = st.text_input("💡 **What's your video about?**", placeholder="e.g., 'Why the Earth rotates' for Educational, 'New running shoes' for Advertisement, 'A dystopian future' for Movie Trailer")
video_topic = sanitize_for_api(video_topic_raw)

# --- Video Length and Category ---
col_len_cat1, col_len_cat2 = st.columns(2)

with col_len_cat1:
    video_length_option_key = "video_length_option_pre_select"
    if video_length_option_key not in st.session_state:
        st.session_state[video_length_option_key] = "20 seconds" # Default value

    video_length_option = st.selectbox(
        "⏳ **Video Length:**",
        ["10 seconds", "15 seconds", "20 seconds"],
        key=video_length_option_key,
        help="Select the desired total length of your video."
    )

total_video_duration = 0
num_segments = 0
script_prompt_template = ""
video_visual_style_prompt = ""
music_style_prompt = ""
base_script_prompt_template = ""

if video_length_option == "10 seconds":
    num_segments = 2
    total_video_duration = 10
    base_script_prompt_template = f"The video will be {total_video_duration} seconds long; divide your script into {num_segments} segments of approximately 5 seconds each. Each segment should be approximately 15-25 words, providing detailed and continuous narration to fill its 5-second duration with spoken content, not silence. Label each section clearly as '1:', and '2:'. "
elif video_length_option == "15 seconds":
    num_segments = 3
    total_video_duration = 15
    base_script_prompt_template = f"The video will be {total_video_duration} seconds long; divide your script into {num_segments} segments of approximately 5 seconds each. Each segment should be approximately 15-25 words, providing detailed and continuous narration to fill its 5-second duration with spoken content, not silence. Label each section clearly as '1:', '2:', and '3:'. "
else: # Default to 20 seconds
    num_segments = 4
    total_video_duration = 20
    base_script_prompt_template = f"The video will be {total_video_duration} seconds long; divide your script into {num_segments} segments of approximately 5 seconds each. Each segment should be approximately 15-25 words, providing detailed and continuous narration to fill its 5-second duration with spoken content, not silence. Label each section clearly as '1:', '2:', '3:', and '4:'. "

with col_len_cat2:
    video_category_options = ["Educational", "Advertisement", "Movie Trailer"]
    video_category = st.selectbox(
        "🏷️ **Video Category:**",
        video_category_options,
        index=video_category_options.index(st.session_state.get("video_category", "Educational")),
        key="video_category",
        help="Choose the genre of your video. This influences the script and visual style."
    )

# Adjust prompts based on video category
if video_category == "Educational":
    script_prompt_template = (
        f"You are an expert video scriptwriter. Write a clear, engaging, thematically consistent voiceover script for a {total_video_duration}-second educational video titled '{{video_topic}}'. "
        f"{base_script_prompt_template}"
        f"Make sure the {num_segments} segments tell a cohesive, progressive story that builds toward a compelling conclusion. "
        f"Use vivid, concrete language that translates well to visuals. Include specific details, numbers, or comparisons when relevant. "
        f"Write in an, conversational tone that keeps viewers hooked. Avoid generic statements."
    )
    video_visual_style_prompt = f"educational video about '{{video_topic}}'. Style: {{video_style.lower()}}, clean, professional, well-lit. Camera movement: smooth, purposeful. No text overlays."
    music_style_prompt = f"Background music for a cohesive, {total_video_duration}-second educational video about {{video_topic}}. Light, non-distracting, slightly cinematic tone."

elif video_category == "Advertisement":
    script_prompt_template = (
        f"You are an expert video scriptwriter. Write a compelling, persuasive script for a {total_video_duration}-second **advertisement** about '{{video_topic}}'. "
        f"{base_script_prompt_template}"
        f"Focus on benefits, problem-solution, and a clear call to action. Each segment should highlight a key feature, benefit, or evoke a positive emotion. "
        f"The final segment should include a strong call to action (e.g., 'Learn more at...', 'Buy now!', 'Visit our website!'). "
        f"Use a professional, enticing, and slightly urgent tone. Avoid generic statements."
    )
    video_visual_style_prompt = f"dynamic, visually appealing shots for a product/service advertisement about '{{video_topic}}'. Highlight features. Style: modern, vibrant, clean, commercial-ready. Camera movement: engaging, product-focused. No text overlays."
    music_style_prompt = f"Upbeat, modern, and catchy background music for a commercial advertisement about {{video_topic}}. Energetic and positive tone."

elif video_category == "Movie Trailer":
    script_prompt_template = (
        f"You are an expert video scriptwriter. Write a dramatic, suspenseful script for a {total_video_duration}-second **movie trailer** for a film titled '{{video_topic}}'. "
        f"{base_script_prompt_template}"
        f"Each segment should introduce elements of the plot, characters, or rising conflict, building suspense. "
        f"The final segment should be a compelling, open-ended hook that leaves the audience wanting more. "
        f"Use evocative language, questions, and a fast-paced, intense tone. Build anticipation."
    )
    video_visual_style_prompt = f"epic, dramatic, cinematic shots for a movie trailer about '{{video_topic}}'. Emphasize tension, conflict, character expressions. Style: dark, moody, high-contrast, blockbuster film. Camera movement: intense, sweeping, purposeful. No text overlays."
    music_style_prompt = f"Dramatic, suspenseful, and epic background music for a movie trailer about {{video_topic}}. Build tension and excitement with orchestral elements."

# Dictionary mapping display names to Replicate voice IDs for the speech models (fixed for now)
voice_options = {
    "Wise Woman": "Wise_Woman",
    "Friendly Person": "Friendly_Person",
    "Inspirational Girl": "Inspirational_girl",
    "Deep Voice Man": "Deep_Voice_Man",
    "Calm Woman": "Calm_Woman",
    "Casual Guy": "Casual_Guy",
    "Lively Girl": "Lively_Girl",
    "Patient Man": "Patient_Man",
    "Young Knight": "Young_Knight",
    "Determined Man": "Determined_Man",
    "Lovely Girl": "Lovely_Girl",
    "Decent Boy": "Decent_Boy",
    "Imposing Manner": "Imposing_Manner",
    "Elegant Man": "Elegant_Man",
    "Abbess": "Abbess",
    "Sweet Girl 2": "Sweet_Girl_2",
    "Exuberant Girl": "Exuberant_Girl"
}

# List of available emotion options for the voiceover
emotion_options = ["auto", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]

# --- Model Selection ---
st.header("2. Choose Your AI Models")
st.markdown("Select the AI models that will power your video creation. Different models offer varying capabilities and costs.")
col_model1, col_model2, col_model3, col_model4 = st.columns(4)

with col_model1:
    selected_text_model_name = st.selectbox(
        "📝 **Text Model:**",
        options=[config["name"] for config in MODEL_CONFIGS["text"].values()],
        index=0, # Default to Claude 4 Sonnet
        key="text_model_select",
        help="This model generates the video script."
    )
    selected_text_model_id = next(config["model_id"] for config in MODEL_CONFIGS["text"].values() if config["name"] == selected_text_model_name)

with col_model2:
    selected_speech_model_name = st.selectbox(
        "🗣️ **Speech Model:**",
        options=[config["name"] for config in MODEL_CONFIGS["speech"].values()],
        index=0, # Default to MiniMax Speech-02-Turbo
        key="speech_model_select",
        help="This model converts the script into a voiceover."
    )
    selected_speech_model_id = next(config["model_id"] for config in MODEL_CONFIGS["speech"].values() if config["name"] == selected_speech_model_name)

with col_model3:
    selected_video_model_name = st.selectbox(
        "🎥 **Video Model:**",
        options=[config["name"] for config in MODEL_CONFIGS["video"].values()],
        index=0, # Default to Luma Ray Flash 2 (540p)
        key="video_model_select",
        help="This model generates the visual segments for your video."
    )
    selected_video_model_id = next(config["model_id"] for config in MODEL_CONFIGS["video"].values() if config["name"] == selected_video_model_name)

with col_model4:
    selected_music_model_name = st.selectbox(
        "🎵 **Music Model:**",
        options=[config["name"] for config in MODEL_CONFIGS["music"].values()],
        index=0, # Default to Google Lyria 2
        key="music_model_select",
        help="This model generates the background music for your video."
    )
    selected_music_model_id = next(config["model_id"] for config in MODEL_CONFIGS["music"].values() if config["name"] == selected_music_model_name)

# --- Advanced Model Parameters (Dynamic) ---
st.header("3. Fine-Tune Model Parameters")
st.markdown("Adjust specific settings for your chosen AI models. These parameters can significantly impact the output.")

# Get the selected model configurations
text_model_config = next(config for config in MODEL_CONFIGS["text"].values() if config["name"] == selected_text_model_name)
speech_model_config = next(config for config in MODEL_CONFIGS["speech"].values() if config["name"] == selected_speech_model_name)
video_model_config = next(config for config in MODEL_CONFIGS["video"].values() if config["name"] == selected_video_model_name)
music_model_config = next(config for config in MODEL_CONFIGS["music"].values() if config["name"] == selected_music_model_name)

# Create dynamic parameter inputs for each model type if they have configurable parameters
advanced_params = {}

with st.expander("⚙️ Adjust Model Parameters"):
    col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)

    with col_adv1:
        st.markdown(f"**{selected_text_model_name}**")
        if text_model_config["parameters"]:
            for param_name, details in text_model_config["parameters"].items():
                if details["type"] == "float":
                    min_val = float(details["min"])
                    max_val = float(details["max"])
                    default_val = float(details["default"])
                    step_val = float(details.get("step", 0.1))
                    advanced_params[f"text_{param_name}"] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"text_{param_name}")
                elif details["type"] == "int":
                    min_val = int(details["min"])
                    max_val = int(details["max"])
                    default_val = int(details["default"])
                    step_val = int(details.get("step", 1))
                    advanced_params[f"text_{param_name}"] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"text_{param_name}")
                elif details["type"] == "str" and "options" in details:
                    advanced_params[f"text_{param_name}"] = st.selectbox(param_name, options=details["options"], index=details["options"].index(details["default"]) if details["default"] in details["options"] else 0, key=f"text_{param_name}")
                elif details["type"] == "str":
                    advanced_params[f"text_{param_name}"] = st.text_input(param_name, value=details["default"], key=f"text_{param_name}")
                elif details["type"] == "bool":
                    advanced_params[f"text_{param_name}"] = st.checkbox(param_name, value=details["default"], key=f"text_{param_name}")
        else:
            st.info("No configurable parameters.")

    with col_adv2:
        st.markdown(f"**{selected_speech_model_name}**")
        if speech_model_config["parameters"]:
            for param_name, details in speech_model_config["parameters"].items():
                if details["type"] == "float":
                    min_val = float(details["min"])
                    max_val = float(details["max"])
                    default_val = float(details["default"])
                    step_val = float(details.get("step", 0.1))
                    advanced_params[f"speech_{param_name}"] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"speech_{param_name}")
                elif details["type"] == "int":
                    min_val = int(details["min"])
                    max_val = int(details["max"])
                    default_val = int(details["default"])
                    step_val = int(details.get("step", 1))
                    advanced_params[f"speech_{param_name}"] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"speech_{param_name}")
                elif details["type"] == "str" and "options" in details:
                    advanced_params[f"speech_{param_name}"] = st.selectbox(param_name, options=details["options"], index=details["options"].index(details["default"]) if details["default"] in details["options"] else 0, key=f"speech_{param_name}")
                elif details["type"] == "str":
                    advanced_params[f"speech_{param_name}"] = st.text_input(param_name, value=details["default"], key=f"speech_{param_name}")
                elif details["type"] == "bool":
                    advanced_params[f"speech_{param_name}"] = st.checkbox(param_name, value=details["default"], key=f"speech_{param_name}")
        else:
            st.info("No configurable parameters.")

    with col_adv3:
        st.markdown(f"**{selected_video_model_name}**")
        if video_model_config["parameters"]:
            for param_name, details in video_model_config["parameters"].items():
                if details["type"] == "float":
                    min_val = float(details["min"])
                    max_val = float(details["max"])
                    default_val = float(details["default"])
                    step_val = float(details.get("step", 0.1))
                    advanced_params[f"video_{param_name}"] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"video_{param_name}")
                elif details["type"] == "int":
                    min_val = int(details["min"])
                    max_val = int(details["max"])
                    default_val = int(details["default"])
                    step_val = int(details.get("step", 1))
                    advanced_params[f"video_{param_name}"] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"video_{param_name}")
                elif details["type"] == "str" and "options" in details:
                    advanced_params[f"video_{param_name}"] = st.selectbox(param_name, options=details["options"], index=details["options"].index(details["default"]) if details["default"] in details["options"] else 0, key=f"video_{param_name}")
                elif details["type"] == "str":
                    advanced_params[f"video_{param_name}"] = st.text_input(param_name, value=details["default"], key=f"video_{param_name}")
                elif details["type"] == "bool":
                    advanced_params[f"video_{param_name}"] = st.checkbox(param_name, value=details["default"], key=f"video_{param_name}")
        else:
            st.info("No configurable parameters.")

    with col_adv4:
        st.markdown(f"**{selected_music_model_name}**")
        if music_model_config["parameters"]:
            for param_name, details in music_model_config["parameters"].items():
                if details["type"] == "float":
                    min_val = float(details["min"])
                    max_val = float(details["max"])
                    default_val = float(details["default"])
                    step_val = float(details.get("step", 0.1))
                    advanced_params[f"music_{param_name}"] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"music_{param_name}")
                elif details["type"] == "int":
                    min_val = int(details["min"])
                    max_val = int(details["max"])
                    default_val = int(details["default"])
                    step_val = int(details.get("step", 1))
                    advanced_params[f"music_{param_name}"] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"music_{param_name}")
                elif details["type"] == "str" and "options" in details:
                    advanced_params[f"music_{param_name}"] = st.selectbox(param_name, options=details["options"], index=details["options"].index(details["default"]) if details["default"] in details["options"] else 0, key=f"music_{param_name}")
                elif details["type"] == "str":
                    advanced_params[f"music_{param_name}"] = st.text_input(param_name, value=details["default"], key=f"music_{param_name}")
                elif details["type"] == "bool":
                    advanced_params[f"music_{param_name}"] = st.checkbox(param_name, value=details["default"], key=f"music_{param_name}")
        else:
            st.info("No configurable parameters.")

# --- Estimated Cost Calculation ---
def calculate_estimated_cost(total_video_duration, num_segments, selected_text_model_id, selected_speech_model_id, selected_video_model_id, selected_music_model_id, script_prompt_template, video_topic, include_voiceover_flag=True, cleaned_narration_content=""):
    cost = 0.0

    # Text Model Cost
    text_model_config = next(config for config in MODEL_CONFIGS["text"].values() if config["model_id"] == selected_text_model_id)
    # Estimate input tokens: prompt length / 4 (chars per token)
    estimated_input_tokens = len(script_prompt_template.format(video_topic=video_topic)) / 4
    # Estimate output tokens: based on average words per segment and total segments.
    # 20 words/segment * 1.3 tokens/word = 26 tokens/segment
    estimated_output_tokens = (total_video_duration / 5) * 26 
    
    cost += (estimated_input_tokens / 1_000_000) * text_model_config["input_cost_per_million_tokens"]
    cost += (estimated_output_tokens / 1_000_000) * text_model_config["output_cost_per_million_tokens"]

    # Speech Model Cost
    if include_voiceover_flag and cleaned_narration_content: # Only calculate if voiceover is included and not empty
        speech_model_config = next(config for config in MODEL_CONFIGS["speech"].values() if config["model_id"] == selected_speech_model_id)
        if "cost_per_second" in speech_model_config:
            # We estimate speech duration as total_video_duration, including the 2s lead-in
            cost += (total_video_duration) * speech_model_config["cost_per_second"]
        elif "cost_per_run" in speech_model_config:
            cost += speech_model_config["cost_per_run"]
    
    # Video Model Cost
    video_model_config = next(config for config in MODEL_CONFIGS["video"].values() if config["model_id"] == selected_video_model_id)
    if "cost_per_video_segment" in video_model_config:
        cost += num_segments * video_model_config["cost_per_video_segment"]
    elif "cost_per_second" in video_model_config:
        cost += total_video_duration * video_model_config["cost_per_second"]

    # Music Model Cost
    music_model_config = next(config for config in MODEL_CONFIGS["music"].values() if config["model_id"] == selected_music_model_id)
    if "cost_per_second" in music_model_config:
        cost += total_video_duration * music_model_config["cost_per_second"]
    elif "cost_per_run" in music_model_config:
        cost += music_model_config["cost_per_run"] # Assume one music generation per video

    return cost

# Calculate estimated cost dynamically
# Need a placeholder for cleaned_narration to pass to cost calculation
temp_cleaned_narration = "This is a placeholder for narration to estimate costs."
estimated_cost = calculate_estimated_cost(
    total_video_duration,
    num_segments,
    selected_text_model_id,
    selected_speech_model_id,
    selected_video_model_id,
    selected_music_model_id,
    script_prompt_template,
    video_topic,
    st.session_state.get("include_voiceover", True), # Pass the state of the checkbox
    temp_cleaned_narration if st.session_state.get("include_voiceover", True) else ""
)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Estimated Video Cost")
st.sidebar.metric("Total Estimated Cost", f"${estimated_cost:.4f}", help="This is an estimate based on average usage of the selected models. Actual costs may vary.")

# --- Video Settings ---
st.header("4. Customize Video Appearance")
col_vid_set1, col_vid_set2, col_vid_set3 = st.columns(3)

with col_vid_set1:
    video_style_options = ["Documentary", "Cinematic", "Educational", "Modern", "Nature", "Scientific", "Abstract", "Fantasy"]
    video_style = st.selectbox(
        "✨ **Video Style:**",
        video_style_options,
        index=video_style_options.index(st.session_state.get("video_style", "Documentary")),
        key="video_style",
        help="Choose the overall visual aesthetic for your video."
    )

    aspect_ratio_options = ["16:9", "9:16", "1:1", "4:3"]
    aspect_ratio = st.selectbox(
        "📏 **Video Dimensions:**",
        aspect_ratio_options,
        index=aspect_ratio_options.index(st.session_state.get("aspect_ratio", "16:9")),
        key="aspect_ratio",
        help="Select the aspect ratio for your video (e.g., 16:9 for widescreen, 9:16 for vertical shorts)."
    )

with col_vid_set2:
    enable_loop = st.checkbox(
        "🔄 **Loop video segments**",
        value=st.session_state.get("enable_loop", False),
        key="enable_loop",
        help="If checked, video segments will loop smoothly to match audio duration. This can help prevent awkward silence at the end of segments."
    )
    
    # Camera Movement options
    st.markdown("---")
    st.markdown("### Camera Movement (Optional)")
    camera_concepts = [
        "static", "zoom_in", "zoom_out", "pan_left", "pan_right",
        "tilt_up", "tilt_down", "orbit_left", "orbit_right",
        "push_in", "pull_out", "crane_up", "crane_down",
        "aerial", "aerial_drone", "handheld", "dolly_zoom"
    ]

    selected_concepts = st.multiselect(
        "🎬 **Choose camera movements** (applied randomly to segments):",
        options=camera_concepts,
        default=["static", "zoom_in", "pan_right"],
        help="Select desired camera movements to make your video more dynamic. These will be randomly applied to your video segments."
    )


with col_vid_set3:
    st.markdown("### Voiceover Settings")
    selected_voice = st.selectbox(
        "🎤 **Voice (for MiniMax Speech-02-Turbo):**",
        options=list(voice_options.keys()),
        index=list(voice_options.keys()).index(st.session_state.get("selected_voice", "Wise Woman")),
        key="selected_voice",
        help="Select the voice that will narrate your video. Note: This applies mainly to MiniMax Speech-02-Turbo."
    )

    selected_emotion = st.selectbox(
        "🎭 **Voice Emotion (for MiniMax Speech-02-Turbo):**",
        options=emotion_options,
        index=emotion_options.index(st.session_state.get("selected_emotion", "auto")),
        key="selected_emotion",
        help="Choose an emotion for the voiceover. 'Auto' lets the AI decide."
    )

    include_voiceover = st.checkbox(
        "🔇 **Include Voiceover**",
        value=st.session_state.get("include_voiceover", True),
        key="include_voiceover",
        help="Uncheck this if you only want a video with background music, without narration."
    )

# Placeholder for existing or new session state variables for other elements
if "narration_script_raw" not in st.session_state:
    st.session_state.narration_script_raw = ""
if "cleaned_narration_content" not in st.session_state:
    st.session_state.cleaned_narration_content = ""
if "voice_audio_paths" not in st.session_state:
    st.session_state.voice_audio_paths = []
if "video_urls" not in st.session_state:
    st.session_state.video_urls = []
if "music_audio_path" not in st.session_state:
    st.session_state.music_audio_path = None

# --- Main Video Generation Logic (placeholders for actual calls) ---
def run_model(model_id, inputs):
    # This is a placeholder for your actual Replicate API call
    # In a real app, you would use replicate.run(model_id, input=inputs)
    st.info(f"Calling model: {model_id} with inputs: {inputs}")
    # Simulate a response
    if "text" in model_id:
        return "1: This is the first segment of your video script. It introduces the topic with engaging language. 2: The second segment continues the narrative, providing more details and keeping the viewer interested. 3: Finally, the third segment concludes the video with a call to action or a summary."
    elif "speech" in model_id:
        # Simulate an audio URL
        return "https://replicate.delivery/pbxt/C4R3Yk8g1gR2z0W4m5x5X6j/output.mp3"
    elif "video" in model_id:
        # Simulate video URLs for each segment
        return ["https://replicate.delivery/pbxt/video1.mp4", "https://replicate.delivery/pbxt/video2.mp4", "https://replicate.delivery/pbxt/video3.mp4", "https://replicate.delivery/pbxt/video4.mp4"]
    elif "music" in model_id:
        # Simulate a music URL
        return "https://replicate.delivery/pbxt/music.mp3"
    return "Simulated output"

def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return filename

def generate_video_content(video_topic, total_video_duration, num_segments, script_prompt_template, video_visual_style_prompt, music_style_prompt, selected_text_model_id, selected_speech_model_id, selected_video_model_id, selected_music_model_id, selected_voice, selected_emotion, aspect_ratio, enable_loop, advanced_params, include_voiceover):
    replicate.api_token = replicate_api_key
    st.session_state.narration_script_raw = ""
    st.session_state.cleaned_narration_content = ""
    st.session_state.voice_audio_paths = []
    st.session_state.video_urls = []
    st.session_state.music_audio_path = None
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    current_progress = 0

    try:
        # 1. Generate Script
        status_text.text("Generating script...")
        text_inputs = {"prompt": script_prompt_template.format(video_topic=video_topic), **{k.replace("text_", ""): v for k, v in advanced_params.items() if k.startswith("text_")}}
        script_output = run_model(selected_text_model_id, text_inputs) # Replace with actual API call
        st.session_state.narration_script_raw = script_output
        st.write("---")
        st.subheader("Generated Script")
        st.info(st.session_state.narration_script_raw)
        
        # Extract and clean narration segments
        segments = re.findall(r'(\d+):(.*?)', st.session_state.narration_script_raw, re.DOTALL)
        if not segments:
            raise ValueError("No segments found in the generated script. Please adjust prompt or review script.")
        
        cleaned_narration_segments = [sanitize_for_api(segment[1].strip()) for segment in segments]
        st.session_state.cleaned_narration_content = " ".join(cleaned_narration_segments)
        
        current_progress += 20
        progress_bar.progress(current_progress)

        # 2. Generate Voiceover (if enabled)
        if include_voiceover:
            status_text.text("Generating voiceover...")
            for i, segment_text in enumerate(cleaned_narration_segments):
                speech_inputs = {
                    "text": segment_text,
                    "voice": voice_options[selected_voice],
                    "emotion": selected_emotion,
                    **{k.replace("speech_", ""): v for k, v in advanced_params.items() if k.startswith("speech_")}
                }
                voice_audio_url = run_model(selected_speech_model_id, speech_inputs) # Replace with actual API call
                if voice_audio_url:
                    audio_filename = os.path.join(tempfile.gettempdir(), f"voice_segment_{i}.mp3")
                    download_file(voice_audio_url, audio_filename)
                    st.session_state.voice_audio_paths.append(audio_filename)
            st.write("---")
            st.subheader("Generated Voiceover (per segment)")
            for path in st.session_state.voice_audio_paths:
                st.audio(path, format='audio/mp3')

        current_progress += 20
        progress_bar.progress(current_progress)

        # 3. Generate Video Segments
        status_text.text("Generating video segments...")
        video_prompts = [f"{video_visual_style_prompt.format(video_topic=video_topic, video_style=video_style)} {movement} scene for segment {i+1}." for i, movement in enumerate(np.random.choice(selected_concepts, num_segments, replace=True))]

        for i, prompt in enumerate(video_prompts):
            video_inputs = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                **{k.replace("video_", ""): v for k, v in advanced_params.items() if k.startswith("video_")}
            }
            video_url_list = run_model(selected_video_model_id, video_inputs) # Replace with actual API call (returns a list)
            if video_url_list and isinstance(video_url_list, list):
                st.session_state.video_urls.append(video_url_list[0]) # Assuming model returns list and we take the first
            else:
                st.session_state.video_urls.append(video_url_list) # Or just the URL if not a list

        st.write("---")
        st.subheader("Generated Video Segments (URLs)")
        for url in st.session_state.video_urls:
            st.write(url) # Display URLs for debugging or direct access

        current_progress += 30
        progress_bar.progress(current_progress)

        # 4. Generate Background Music
        status_text.text("Generating background music...")
        music_inputs = {
            "prompt": music_style_prompt.format(video_topic=video_topic),
            "duration": float(total_video_duration), # Ensure duration is float
            **{k.replace("music_", ""): v for k, v in advanced_params.items() if k.startswith("music_")}
        }
        music_audio_url = run_model(selected_music_model_id, music_inputs) # Replace with actual API call
        if music_audio_url:
            music_filename = os.path.join(tempfile.gettempdir(), "background_music.mp3")
            download_file(music_audio_url, music_filename)
            st.session_state.music_audio_path = music_filename
            st.write("---")
            st.subheader("Generated Music")
            st.audio(st.session_state.music_audio_path, format='audio/mp3')

        current_progress += 15
        progress_bar.progress(current_progress)

        # 5. Assemble Final Video
        status_text.text("Assembling final video...")
        
        # Download video files
        video_clips = []
        for i, url in enumerate(st.session_state.video_urls):
            video_path = os.path.join(tempfile.gettempdir(), f"video_segment_{i}.mp4")
            download_file(url, video_path)
            clip = VideoFileClip(video_path)
            video_clips.append(clip)

        if not video_clips:
            raise ValueError("No video clips generated to assemble.")

        final_video_clip = concatenate_videoclips(video_clips)

        # Handle audio
        audio_clips = []
        if include_voiceover and st.session_state.voice_audio_paths:
            for path in st.session_state.voice_audio_paths:
                audio_clips.append(AudioFileClip(path))
            voice_track = concatenate_audioclips(audio_clips)
        else:
            voice_track = AudioFileClip(np.zeros(1), fps=44100).set_duration(final_video_clip.duration) # Silent audio if no voiceover

        if st.session_state.music_audio_path:
            music_track = AudioFileClip(st.session_state.music_audio_path)
            # Loop or trim music to match video duration
            if music_track.duration < final_video_clip.duration:
                # Loop music
                num_loops = int(final_video_clip.duration / music_track.duration) + 1
                music_track = concatenate_audioclips([music_track] * num_loops)
            music_track = music_track.set_duration(final_video_clip.duration)
            
            # Mix voice and music
            final_audio = CompositeAudioClip([music_track.volumex(0.3), voice_track]) # Adjust music volume
        else:
            final_audio = voice_track # Only voice if no music

        final_video_clip = final_video_clip.set_audio(final_audio)

        # Export final video
        output_video_path = os.path.join(tempfile.gettempdir(), "final_ai_video.mp4")
        final_video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", fps=24)

        status_text.text("Video generation complete!")
        progress_bar.progress(100)
        st.success("Your video is ready!")
        st.video(output_video_path)

        # Clean up temporary files (optional, but good practice)
        for path in st.session_state.voice_audio_paths + [st.session_state.music_audio_path] + [os.path.join(tempfile.gettempdir(), f"video_segment_{i}.mp4") for i in range(len(st.session_state.video_urls)) if os.path.exists(os.path.join(tempfile.gettempdir(), f"video_segment_{i}.mp4"))]:
             if path and os.path.exists(path):
                 os.remove(path)

    except Exception as e:
        status_text.error(f"An error occurred: {e}")
        st.exception(e)
    finally:
        progress_bar.empty()
        status_text.empty()

# --- Generate Video Button ---
st.markdown("---")
if st.button("🚀 **Generate My Video!**", type="primary", use_container_width=True):
    if not replicate_api_key:
        st.error("Please enter your Replicate API Key to proceed.")
    elif not video_topic:
        st.error("Please enter a video topic.")
    else:
        generate_video_content(
            video_topic,
            total_video_duration,
            num_segments,
            script_prompt_template,
            video_visual_style_prompt,
            music_style_prompt,
            selected_text_model_id,
            selected_speech_model_id,
            selected_video_model_id,
            selected_music_model_id,
            selected_voice,
            selected_emotion,
            aspect_ratio,
            enable_loop,
            advanced_params,
            include_voiceover
        )

st.markdown("---")
st.caption("Powered by Replicate AI models | Developed with Streamlit")

