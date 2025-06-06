import streamlit as st
import replicate
import tempfile
import os
import requests
import re
from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclip, # Changed concatenate_videoclips to concatenate_videoclip
    AudioFileClip,
    CompositeAudioClip,
    concatenate_audioclips,
)
import numpy as np

# Set Streamlit page configuration for a wider layout and custom title
st.set_page_config(layout="wide", page_title="AI Multi-Agent Video Creator")

# Main title of the application
st.title("AI Multi-Agent Video Creator")

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

# --- Streamlit UI ---

# Input fields for Replicate API Key and video topic
replicate_api_key = st.text_input("Enter your Replicate API Key", type="password")
video_topic_raw = st.text_input("Enter a video topic (e.g., 'Why the Earth rotates' for Educational, 'New running shoes' for Advertisement, 'A dystopian future' for Movie Trailer)")
# Sanitize video_topic immediately after input to prevent UnicodeEncodeError in API calls
video_topic = sanitize_for_api(video_topic_raw)

# --- Determine video parameters (total duration and number of segments) based on selected length ---
# This block is moved up to ensure these variables are defined before being used.
video_length_option_key = "video_length_option_pre_select"
if video_length_option_key not in st.session_state:
    st.session_state[video_length_option_key] = "20 seconds" # Default value

video_length_option = st.selectbox(
    "Video Length:",
    ["10 seconds", "15 seconds", "20 seconds"],
    # Removed 'index' parameter, now relying solely on session_state for value
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

# Adjust prompts based on video category
video_category_options = ["Educational", "Advertisement", "Movie Trailer"]
video_category = st.selectbox(
    "Video Category:",
    video_category_options,
    index=video_category_options.index(st.session_state.get("video_category", "Educational")), # Default to Educational
    key="video_category"
)

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
st.subheader("Model Selection")
col_model1, col_model2, col_model3, col_model4 = st.columns(4)

with col_model1:
    selected_text_model_name = st.selectbox(
        "Text Model:",
        options=[config["name"] for config in MODEL_CONFIGS["text"].values()],
        index=0, # Default to Claude 4 Sonnet
        key="text_model_select"
    )
    selected_text_model_id = next(config["model_id"] for config in MODEL_CONFIGS["text"].values() if config["name"] == selected_text_model_name)

with col_model2:
    selected_speech_model_name = st.selectbox(
        "Speech Model:",
        options=[config["name"] for config in MODEL_CONFIGS["speech"].values()],
        index=0, # Default to MiniMax Speech-02-Turbo
        key="speech_model_select"
    )
    selected_speech_model_id = next(config["model_id"] for config in MODEL_CONFIGS["speech"].values() if config["name"] == selected_speech_model_name)

with col_model3:
    selected_video_model_name = st.selectbox(
        "Video Model:",
        options=[config["name"] for config in MODEL_CONFIGS["video"].values()],
        index=0, # Default to Luma Ray Flash 2 (540p)
        key="video_model_select"
    )
    selected_video_model_id = next(config["model_id"] for config in MODEL_CONFIGS["video"].values() if config["name"] == selected_video_model_name)

with col_model4:
    selected_music_model_name = st.selectbox(
        "Music Model:",
        options=[config["name"] for config in MODEL_CONFIGS["music"].values()],
        index=0, # Default to Google Lyria 2
        key="music_model_select"
    )
    selected_music_model_id = next(config["model_id"] for config in MODEL_CONFIGS["music"].values() if config["name"] == selected_music_model_name)

# --- Advanced Model Parameters (Dynamic) ---
st.subheader("Advanced Model Parameters")
st.write("Adjust parameters for the currently selected models below.")

# Get the selected model configurations
text_model_config = next(config for config in MODEL_CONFIGS["text"].values() if config["name"] == selected_text_model_name)
speech_model_config = next(config for config in MODEL_CONFIGS["speech"].values() if config["name"] == selected_speech_model_name)
video_model_config = next(config for config in MODEL_CONFIGS["video"].values() if config["name"] == selected_video_model_name)
music_model_config = next(config for config in MODEL_CONFIGS["music"].values() if config["name"] == selected_music_model_name)

# Create dynamic parameter inputs for each model type if they have configurable parameters
advanced_params = {}

col_adv1, col_adv2, col_adv3, col_adv4 = st.columns(4)

with col_adv1:
    st.markdown(f"**{selected_text_model_name} Parameters**")
    if text_model_config["parameters"]:
        for param_name, details in text_model_config["parameters"].items():
            if details["type"] == "float":
                min_val = float(details["min"])
                max_val = float(details["max"])
                default_val = float(details["default"])
                step_val = float(details.get("step", 0.1))
                advanced_params[param_name] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"text_{param_name}")
            elif details["type"] == "int":
                min_val = int(details["min"])
                max_val = int(details["max"])
                default_val = int(details["default"])
                step_val = int(details.get("step", 1))
                advanced_params[param_name] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"text_{param_name}")
            elif details["type"] == "str" and "options" in details:
                advanced_params[param_name] = st.selectbox(param_name, options=details["options"], index=details["options"].index(details["default"]) if details["default"] in details["options"] else 0, key=f"text_{param_name}")
            elif details["type"] == "str":
                advanced_params[param_name] = st.text_input(param_name, value=details["default"], key=f"text_{param_name}")
            elif details["type"] == "bool":
                advanced_params[param_name] = st.checkbox(param_name, value=details["default"], key=f"text_{param_name}")
    else:
        st.write("No configurable parameters.")

with col_adv2:
    st.markdown(f"**{selected_speech_model_name} Parameters**")
    if speech_model_config["parameters"]:
        for param_name, details in speech_model_config["parameters"].items():
            if details["type"] == "float":
                min_val = float(details["min"])
                max_val = float(details["max"])
                default_val = float(details["default"])
                step_val = float(details.get("step", 0.1))
                advanced_params[param_name] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"speech_{param_name}")
            elif details["type"] == "int":
                min_val = int(details["min"])
                max_val = int(details["max"])
                default_val = int(details["default"])
                step_val = int(details.get("step", 1))
                advanced_params[param_name] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"speech_{param_name}")
            elif details["type"] == "str" and "options" in details:
                advanced_params[param_name] = st.selectbox(param_name, options=details["options"], index=details["options"].index(details["default"]) if details["default"] in details["options"] else 0, key=f"speech_{param_name}")
            elif details["type"] == "str":
                advanced_params[param_name] = st.text_input(param_name, value=details["default"], key=f"speech_{param_name}")
            elif details["type"] == "bool":
                advanced_params[param_name] = st.checkbox(param_name, value=details["default"], key=f"speech_{param_name}")
    else:
        st.write("No configurable parameters.")

with col_adv3:
    st.markdown(f"**{selected_video_model_name} Parameters**")
    if video_model_config["parameters"]:
        for param_name, details in video_model_config["parameters"].items():
            if details["type"] == "float":
                min_val = float(details["min"])
                max_val = float(details["max"])
                default_val = float(details["default"])
                step_val = float(details.get("step", 0.1))
                advanced_params[param_name] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"video_{param_name}")
            elif details["type"] == "int":
                min_val = int(details["min"])
                max_val = int(details["max"])
                default_val = int(details["default"])
                step_val = int(details.get("step", 1))
                advanced_params[param_name] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"video_{param_name}")
            elif details["type"] == "str" and "options" in details:
                advanced_params[param_name] = st.selectbox(param_name, options=details["options"], index=details["options"].index(details["default"]) if details["default"] in details["options"] else 0, key=f"video_{param_name}")
            elif details["type"] == "str":
                advanced_params[param_name] = st.text_input(param_name, value=details["default"], key=f"video_{param_name}")
            elif details["type"] == "bool":
                advanced_params[param_name] = st.checkbox(param_name, value=details["default"], key=f"video_{param_name}")
    else:
        st.write("No configurable parameters.")

with col_adv4:
    st.markdown(f"**{selected_music_model_name} Parameters**")
    if music_model_config["parameters"]:
        for param_name, details in music_model_config["parameters"].items():
            if details["type"] == "float":
                min_val = float(details["min"])
                max_val = float(details["max"])
                default_val = float(details["default"])
                step_val = float(details.get("step", 0.1))
                advanced_params[param_name] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"music_{param_name}")
            elif details["type"] == "int":
                min_val = int(details["min"])
                max_val = int(details["max"])
                default_val = int(details["default"])
                step_val = int(details.get("step", 1))
                advanced_params[param_name] = st.slider(param_name, min_value=min_val, max_value=max_val, value=default_val, step=step_val, key=f"music_{param_name}")
            elif details["type"] == "str" and "options" in details:
                advanced_params[param_name] = st.selectbox(param_name, options=details["options"], index=details["options"].index(details["default"]) if details["default"] in details["options"] else 0, key=f"music_{param_name}")
            elif details["type"] == "str":
                advanced_params[param_name] = st.text_input(param_name, value=details["default"], key=f"music_{param_name}")
            elif details["type"] == "bool":
                advanced_params[param_name] = st.checkbox(param_name, value=details["default"], key=f"music_{param_name}")
    else:
        st.write("No configurable parameters.")


# --- Estimated Cost Calculation ---
def calculate_estimated_cost(total_video_duration, num_segments, selected_text_model_id, selected_speech_model_id, selected_video_model_id, selected_music_model_id, script_prompt_template, video_topic, include_voiceover_flag=True, cleaned_narration_content=""):
    cost = 0.0

    # Text Model Cost
    text_model_config = next(config for config in MODEL_CONFIGS["text"].values() if config["model_id"] == selected_text_model_id)
    # Estimate input tokens: prompt length / 4 (chars per token)
    estimated_input_tokens = len(script_prompt_template.format(video_topic=video_topic)) / 4
    # Estimate output tokens: based on average words per segment and total segments.
    # 20 words/segment * 1.3 tokens/word = 26 tokens/segment
    estimated_output_tokens = (total_video_duration / 5) * 26 # 5 seconds per segment, 26 tokens per segment
    
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
st.metric("Estimated Cost", f"${estimated_cost:.4f}") # Display estimated cost

# --- Camera Movement options ---
st.subheader("Camera Movement (Optional)")
camera_concepts = [
    "static", "zoom_in", "zoom_out", "pan_left", "pan_right",
    "tilt_up", "tilt_down", "orbit_left", "orbit_right",
    "push_in", "pull_out", "crane_up", "crane_down",
    "aerial", "aerial_drone", "handheld", "dolly_zoom"
]

selected_concepts = st.multiselect(
    "Choose camera movements (will be applied randomly to segments):",
    options=camera_concepts,
    default=["static", "zoom_in", "pan_right"],
    help="Select camera movements to make your video more dynamic"
)


# --- Video Settings --- # Moved below Model Selection and Advanced Parameters
st.subheader("Video Settings")
col1, col2, col3 = st.columns(3)

with col1:
    # video_category is defined above now
    video_style_options = ["Documentary", "Cinematic", "Educational", "Modern", "Nature", "Scientific"]
    video_style = st.selectbox(
        "Video Style:",
        video_style_options,
        index=video_style_options.index(st.session_state.get("video_style", "Documentary")),
        key="video_style",
        help="Choose the visual style for your video"
    )

    aspect_ratio_options = ["16:9", "9:16", "1:1", "4:3"]
    aspect_ratio = st.selectbox(
        "Video Dimensions:",
        aspect_ratio_options,
        index=aspect_ratio_options.index(st.session_state.get("aspect_ratio", "16:9")),
        key="aspect_ratio",
        help="Choose aspect ratio for your video"
    )

with col2:
    # num_frames_option removed from here as it's now part of Luma model's dynamic params
    enable_loop = st.checkbox(
        "Loop video segments",
        value=st.session_state.get("enable_loop", False),
        key="enable_loop",
        help="Make video segments loop smoothly"
    )

with col3:
    selected_voice = st.selectbox(
        "Voice (for MiniMax Speech-02-Turbo):",
        options=list(voice_options.keys()),
        index=list(voice_options.keys()).index(st.session_state.get("selected_voice", "Wise Woman")),
        key="selected_voice",
        help="Select the voice that will narrate your video"
    )

    selected_emotion = st.selectbox(
        "Voice emotion (for MiniMax Speech-02-Turbo):",
        options=emotion_options,
        index=emotion_options.index(st.session_state.get("selected_emotion", "auto")),
        key="selected_emotion",
        help="Select the emotional tone for the voiceover"
    )

# --- Audio Settings ---
st.subheader("Audio Settings")
col_audio1, col_audio2 = st.columns(2)
with col_audio1:
    include_voiceover = st.checkbox("Include VoiceOver", value=st.session_state.get("include_voiceover", True), key="include_voiceover", help="Check to include a generated voiceover narration in your video.")
with col_audio2:
    # video_length_option is defined above now
    pass # No need for a selectbox here as it's defined globally at the top


# --- Main Generation Logic ---
if replicate_api_key and video_topic and st.button(f"Generate {video_length_option} Video"):
    replicate_client = replicate.Client(api_token=replicate_api_key)

    def run_replicate(model_id, input_data):
        return replicate_client.run(model_id, input=input_data)

    def download_to_file(url: str, suffix: str):
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with open(tmp.name, "wb") as f:
            for chunk in resp.iter_content(1024 * 32):
                f.write(chunk)
        return tmp.name

    # Step 1: Write the cohesive script for the full video
    st.info(f"Step 1: Writing cohesive script for {total_video_duration}-second {video_category} video using {selected_text_model_name}")
    
    sanitized_script_prompt = sanitize_for_api(script_prompt_template.format(video_topic=video_topic))
    full_script = run_replicate(
        selected_text_model_id,
        {
            "prompt": sanitized_script_prompt
        },
    )

    script_text = "".join(full_script) if isinstance(full_script, list) else full_script
    script_segments = re.findall(r"\d+:\s*(.+)", script_text)

    if len(script_segments) < num_segments:
        st.error(f"Failed to extract {num_segments} clear script segments. Try adjusting your topic or refining the prompt.")
        st.stop()
    script_segments = script_segments[:num_segments]

    st.success("Script written successfully")
    script_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(script_file_path, "w") as f:
        f.write("\n\n".join(script_segments))
    st.download_button("Download Script", script_file_path, "script.txt")

    # Step 2: Generate voiceover narration directly after script
    voice_path = None
    if include_voiceover:
        st.info(f"Step 2: Generating voiceover narration with {selected_voice} voice using {selected_speech_model_name}")
        full_narration = " ".join(script_segments)
        cleaned_narration = re.sub(r'[^\w\s.,!?]', '', full_naration)
        
        if not cleaned_narration.strip():
            st.warning("Voiceover script became empty after cleaning. Skipping voiceover generation.")
            voice_path = None
        else:
            try:
                sanitized_narration_text = sanitize_for_api(cleaned_narration)
                speech_model_params = {}
                for param_name, details in speech_model_config["parameters"].items():
                    # Only include parameters if they are explicitly in advanced_params (i.e., user adjusted)
                    # or if they are essential model parameters like voice_id, emotion
                    if param_name in advanced_params:
                        speech_model_params[param_name] = advanced_params[param_name]
                
                # Fixed parameters for minimax/speech-02-turbo (as they are not exposed for dynamic change or are default)
                if selected_speech_model_id == "minimax/speech-02-turbo":
                    speech_model_params["voice_id"] = voice_options[selected_voice]
                    speech_model_params["emotion"] = selected_emotion
                    speech_model_params["bitrate"] = 128000
                    speech_model_params["channel"] = "mono"
                    speech_model_params["sample_rate"] = 32000
                    speech_model_params["language_boost"] = "English"
                    speech_model_params["english_normalization"] = True
                
                # For Kokoro-82M, it needs text and possibly speed
                if selected_speech_model_id == "jaaari/kokoro-82m":
                    speech_model_params["text"] = sanitized_narration_text # Text is required for this model
                    if "speed" in advanced_params:
                        speech_model_params["speed"] = advanced_params["speed"]
                
                # For MiniMax Speech-02-HD
                if selected_speech_model_id == "minimax/speech-02-hd":
                    speech_model_params["voice_id"] = voice_options[selected_voice]
                    speech_model_params["emotion"] = selected_emotion
                    speech_model_params["bitrate"] = 128000
                    speech_model_params["channel"] = "mono"
                    speech_model_params["sample_rate"] = 32000
                    speech_model_params["language_boost"] = "English"
                    speech_model_params["english_normalization"] = True

                # For OpenVoice v2
                if selected_speech_model_id == "replicate/openvoice-v2":
                    speech_model_params["speed"] = advanced_params.get("speed", 1.0)


                voiceover_uri = run_replicate(
                    selected_speech_model_id,
                    {
                        "text": sanitized_narration_text, # Text is always the main input
                        **speech_model_params # Merge dynamically collected parameters
                    },
                )
                voice_path = download_to_file(voiceover_uri, suffix=".mp3")

                if not os.path.exists(voice_path) or os.path.getsize(voice_path) == 0:
                    st.error("Generated voiceover file is empty or missing. It might not have generated correctly on Replicate's side.")
                    voice_path = None
                else:
                    st.audio(voice_path)
                    st.download_button("Download Voiceover", voice_path, "voiceover.mp3")

            except Exception as e:
                st.error(f"Failed to generate or download voiceover: {e}")
                voice_path = None

    temp_video_paths = []
    segment_clips = []

    # Step 3: Generate segment visuals for each script segment
    for i, segment in enumerate(script_segments):
        st.info(f"Step 3.{i+1}: Generating visuals for segment {i+1} using {selected_video_model_name}")
        if i == 0:
            shot_type = "establishing wide shot"
        elif i == 1 and num_segments > 2:
            shot_type = "medium shot with focus on key elements"
        elif i == 2 and num_segments > 3:
            shot_type = "close-up shot showing important details"
        else:
            shot_type = "dynamic concluding shot"

        video_prompt = f"Cinematic {shot_type} for {video_visual_style_prompt.format(video_topic=video_topic)}. Visual content: {segment}."
        sanitized_video_prompt = sanitize_for_api(video_prompt)
        
        video_model_params = {}
        for param_name, details in video_model_config["parameters"].items():
            if param_name in advanced_params:
                video_model_params[param_name] = advanced_params[param_name]
        
        # Default parameters for selected video model if not overridden by user
        if selected_video_model_id == "luma/ray-flash-2-540p":
            video_model_params.setdefault("num_frames", advanced_params.get("num_frames", 120)) # This 'num_frames' is now from advanced_params
            video_model_params.setdefault("fps", 24)
            video_model_params.setdefault("guidance", 3.0)
            video_model_params.setdefault("num_inference_steps", 30)

        elif selected_video_model_id == "google/veo-3":
            video_model_params.setdefault("fps", 24)
            video_model_params.setdefault("quality", 10)
            
        elif selected_video_model_id == "minimax/video-01-director":
            video_model_params.setdefault("fps", 24)
            video_model_params.setdefault("num_inference_steps", 50)
        
        elif selected_video_model_id == "google/veo-2":
            video_model_params.setdefault("fps", 24)
            video_model_params.setdefault("quality", 7)
        
        elif selected_video_model_id == "wan-video/wan-2.1-1.3b":
            # No specific advanced parameters to set beyond prompt
            pass

        try:
            video_uri = run_replicate(
                selected_video_model_id,
                {
                    "prompt": sanitized_video_prompt,
                    **video_model_params
                },
            )
            video_path = download_to_file(video_uri, suffix=".mp4")
            temp_video_paths.append(video_path)

            clip = VideoFileClip(video_path).subclip(0, 5)
            segment_clips.append(clip)

            st.video(video_path)
            st.download_button(f"Download Segment {i+1}", video_path, f"segment_{i+1}.mp4")
        except Exception as e:
            st.error(f"Failed to generate or download segment {i+1} video: {e}")
            st.stop()

    # Step 4: Concatenate video segments first
    st.info("Step 4: Combining video segments")
    try:
        final_video = concatenate_videoclip(segment_clips, method="compose") # Changed to concatenate_videoclip
        final_video = final_video.set_duration(total_video_duration)
        final_duration = final_video.duration
        st.success(f"Video segments combined - Total duration: {final_duration} seconds")
    except Exception as e:
        st.error(f"Failed to combine video segments: {e}")
        st.stop()

    # Step 5: Generate background music
    st.info("Step 5: Creating background music using {selected_music_model_name}")
    music_path = None
    try:
        sanitized_music_prompt = sanitize_for_api(music_style_prompt.format(video_topic=video_topic))
        
        music_model_params = {}
        for param_name, details in music_model_config["parameters"].items():
            if param_name in advanced_params:
                music_model_params[param_name] = advanced_params[param_name]

        # Ensure 'prompt' is always passed, other model-specific defaults
        if selected_music_model_id == "google/lyria-2":
            music_model_params.setdefault("prompt", sanitized_music_prompt)
        elif selected_music_model_id == "meta/musicgen":
            music_model_params.setdefault("prompt", sanitized_music_prompt)
            music_model_params.setdefault("duration", 10.0) # Default duration for musicgen
            music_model_params.setdefault("model_version", "melody")
            # If user has set duration, override it.
            if "duration" in advanced_params:
                music_model_params["duration"] = advanced_params["duration"]
            if "model_version" in advanced_params:
                music_model_params["model_version"] = advanced_params["model_version"]
        
        elif selected_music_model_id == "lucataco/ace-step":
            music_model_params.setdefault("prompt", sanitized_music_prompt)


        music_uri = run_replicate(
            selected_music_model_id,
            {
                "prompt": sanitized_music_prompt, # Prompt is always needed
                **music_model_params
            },
        )
        music_path = download_to_file(music_uri, suffix=".mp3")
        st.audio(music_path)
        st.download_button("Download Background Music", music_path, "background_music.mp3")
    except Exception as e:
        st.error(f"Failed to generate or download music: {e}")
        music_path = None

    # Step 6: Merge all audio with video
    st.info("Step 6: Merging final audio and video")
    try:
        audio_clips = []
        
        if voice_path:
            try:
                voice_clip = AudioFileClip(voice_path)
                st.write(f"DEBUG: Original voiceover clip loaded. Duration: {voice_clip.duration}s, FPS: {voice_clip.fps}, Channels: {voice_clip.nchannels}")
                
                if voice_clip.duration == 0:
                    st.error("Generated voiceover audio clip has zero duration. It might be corrupted or empty.")
                    voice_path = None
                else:
                    voice_volume = 1.2
                    voice_clip = voice_clip.volumex(voice_volume)

                    initial_silence_duration = 2.0
                    sr = int(voice_clip.fps) if voice_clip.fps else 44100
                    nchannels = voice_clip.nchannels if voice_clip.nchannels else 1

                    if initial_silence_duration > 0 and sr > 0:
                        initial_silence = np.zeros((int(initial_silence_duration * sr), nchannels), dtype=np.float32)
                        initial_silence_clip = AudioArrayClip(initial_silence, fps=sr)
                        voice_clip = concatenate_audioclips([initial_silence_clip, voice_clip])
                        st.write(f"DEBUG: Voiceover clip duration after adding {initial_silence_duration}s initial silence: {voice_clip.duration} seconds.")
                    else:
                        st.warning(f"DEBUG: Skipping initial voiceover silence. Silence duration: {initial_silence_duration}, Sample Rate: {sr}")
                    
                    if voice_clip.duration > final_duration:
                        voice_clip = voice_clip.subclip(0, final_duration)
                        st.write(f"DEBUG: Voiceover trimmed to final duration. New total duration: {voice_clip.duration} seconds.")
                    elif voice_clip.duration < final_duration:
                        silence_needed = final_duration - voice_clip.duration
                        if silence_needed > 0 and sr > 0:
                            silence = np.zeros((int(silence_needed * sr), nchannels), dtype=np.float32)
                            silence_clip = AudioArrayClip(silence, fps=sr)
                            voice_clip = concatenate_audioclips([voice_clip, silence_clip])
                            st.write(f"DEBUG: Voiceover padded with {silence_needed} seconds of silence to match final duration. New total duration: {voice_clip.duration} seconds.")
                        else:
                            st.warning(f"DEBUG: Skipping voiceover end-padding. Silence needed: {silence_needed}, Sample Rate: {sr}")
                            
                    audio_clips.append(voice_clip)
                    st.write("DEBUG: Playing processed voiceover clip (before final merge):")
                    temp_voice_clip_export_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                    try:
                        voice_clip.write_audiofile(temp_voice_clip_export_path, fps=sr)
                        st.audio(temp_voice_clip_export_path)
                    except Exception as export_e:
                        st.error(f"DEBUG: Could not play processed voiceover for debug: {export_e}")
                    finally:
                        if os.path.exists(temp_voice_clip_export_path):
                            os.remove(temp_voice_clip_export_path)

            except Exception as e:
                st.error(f"Error loading or processing voiceover audio clip: {e}")
                voice_path = None

        if music_path:
            try:
                music_clip = AudioFileClip(music_path)
                st.write(f"DEBUG: Music clip loaded. Duration: {music_clip.duration}s, FPS: {music_clip.fps}, Channels: {music_clip.nchannels}")
                
                if music_clip.duration == 0:
                    st.error("Generated background music audio clip has zero duration. It might be corrupted or empty.")
                    music_path = None
                else:
                    music_volume = 0.3
                    music_clip = music_clip.volumex(music_volume).audio_fadein(0.5).audio_fadeout(2.5)
                    
                    if music_clip.duration < final_duration:
                        loops_needed = int(final_duration / music_clip.duration) + 1
                        music_clips_looped = [music_clip] * loops_needed
                        music_clip = concatenate_audioclips(music_clips_looped)
                        music_clip = music_clip.subclip(0, final_duration)
                        st.write(f"DEBUG: Music looped and trimmed. New duration: {music_clip.duration} seconds.")
                    elif music_clip.duration > final_duration:
                        music_clip = music_clip.subclip(0, final_duration)
                        st.write(f"DEBUG: Music trimmed. New duration: {music_clip.duration} seconds.")
                        
                    audio_clips.append(music_clip)
                    st.write("DEBUG: Playing processed music clip (before final merge):")
                    temp_music_clip_export_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
                    try:
                        music_clip.write_audiofile(temp_music_clip_export_path, fps=music_clip.fps)
                        st.audio(temp_music_clip_export_path)
                    except Exception as export_e:
                        st.error(f"DEBUG: Could not play processed music for debug: {export_e}")
                    finally:
                        if os.path.exists(temp_music_clip_export_path):
                            os.remove(temp_music_clip_export_path)

            except Exception as e:
                st.error(f"Error loading or processing background music clip: {e}")
                music_path = None

        valid_audio_clips = [clip for clip in audio_clips if clip is not None and clip.duration is not None and clip.duration > 0]
        if valid_audio_clips:
            final_audio = CompositeAudioClip(valid_audio_clips)
            st.write(f"DEBUG: Final composite audio clip duration: {final_audio.duration} seconds.")
            final_video = final_video.set_audio(final_audio)
        else:
            final_video = final_video.set_audio(None)
            st.warning("DEBUG: No valid audio clips found after processing. Final video will have no audio.")

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            fps=24
        )

        st.success("🎬 Final video with narration and music is ready")
        st.video(output_path)
        st.download_button("Download Final Video", output_path, "final_video.mp4")

        # --- Create a zip file with all assets and provide a download button ---
        import zipfile
        asset_paths = []
        asset_names = []

        if os.path.exists(script_file_path):
            asset_paths.append(script_file_path)
            asset_names.append("script.txt")
        for idx, seg_path in enumerate(temp_video_paths):
            if os.path.exists(seg_path):
                asset_paths.append(seg_path)
                asset_names.append(f"segment_{idx+1}.mp4")
        if voice_path and os.path.exists(voice_path):
            asset_paths.append(voice_path)
            asset_names.append("voiceover.mp3")
        if music_path and os.path.exists(music_path):
            asset_paths.append(music_path)
            asset_names.append("background_music.mp3")
        if os.path.exists(output_path):
            asset_paths.append(output_path)
            asset_names.append("final_video.mp4")

        zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file_path, arcname in zip(asset_paths, asset_names):
                zipf.write(file_path, arcname=arcname)
        st.download_button("Download All Assets (ZIP)", zip_path, "video_assets.zip")

    except Exception as e:
        st.warning("Final video merge failed, but you can still download individual assets.")
        st.error(f"Error writing final video: {e}")

    # Cleanup: Remove all temporary files
    files_to_clean = temp_video_paths + [script_file_path]
    if voice_path:
        files_to_clean.append(voice_path)
    if music_path:
        files_to_clean.append(music_path)

    for path in files_to_clean:
        try:
            os.remove(path)
        except OSError:
            pass
