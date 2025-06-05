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
import numpy as np # Import numpy for AudioArrayClip silence generation

# Set Streamlit page configuration for a wider layout and custom title
st.set_page_config(layout="wide", page_title="AI Multi-Agent Video Creator")

# Main title of the application
st.title("AI Multi-Agent Video Creator")

# Input fields for Replicate API Key and video topic
replicate_api_key = st.text_input("Enter your Replicate API Key", type="password")
video_topic = st.text_input("Enter a video topic (e.g., 'Why the Earth rotates' for Educational, 'New running shoes' for Advertisement, 'A dystopian future' for Movie Trailer)")

# Dictionary mapping display names to Replicate voice IDs
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

# Section for Video Settings
st.subheader("Video Settings")
# Use columns for better layout of video settings
col1, col2, col3 = st.columns(3)

with col1:
    # New selectbox for video category
    video_category = st.selectbox(
        "Video Category:",
        ["Educational", "Advertisement", "Movie Trailer"],
        help="Choose the category for your video, influencing script and visual style."
    )
    
    # Selectbox for video style
    video_style = st.selectbox(
        "Video Style:",
        ["Documentary", "Cinematic", "Educational", "Modern", "Nature", "Scientific"],
        help="Choose the visual style for your video"
    )

    # Selectbox for aspect ratio
    aspect_ratio = st.selectbox(
        "Video Dimensions:",
        ["16:9", "9:16", "1:1", "4:3"],
        help="Choose aspect ratio for your video"
    )

with col2:
    # Selectbox for video quality (number of frames)
    num_frames = st.selectbox(
        "Video Quality:",
        [("Standard (120 frames)", 120), ("High (200 frames)", 200)],
        format_func=lambda x: x[0] # Display the text part of the tuple
    )[1] # Get the frame count

    # Checkbox to enable looping video segments
    enable_loop = st.checkbox(
        "Loop video segments",
        value=False,
        help="Make video segments loop smoothly"
    )

with col3:
    # Selectbox for voice narration
    selected_voice = st.selectbox(
        "Voice:",
        options=list(voice_options.keys()),
        index=0,
        help="Select the voice that will narrate your video"
    )

    # Selectbox for voice emotion
    selected_emotion = st.selectbox(
        "Voice emotion:",
        options=emotion_options,
        index=0,
        help="Select the emotional tone for the voiceover"
    )

# Section for Audio Settings
st.subheader("Audio Settings")
# Use columns for better layout of audio settings
col_audio1, col_audio2 = st.columns(2)
with col_audio1:
    # Checkbox to include voiceover
    include_voiceover = st.checkbox("Include VoiceOver", value=True, help="Check to include a generated voiceover narration in your video.")
with col_audio2:
    # Selectbox for total video length
    video_length_option = st.selectbox(
        "Video Length:",
        ["10 seconds", "15 seconds", "20 seconds"],
        index=2, # Default to 20 seconds
        help="Select the desired total length of your video."
    )

# Determine video parameters (total duration and number of segments) based on selected length
total_video_duration = 0
num_segments = 0
script_prompt_template = ""
video_visual_style_prompt = ""
music_style_prompt = ""

# Base prompt for script generation, will be modified by category
base_script_prompt = ""
if video_length_option == "10 seconds":
    num_segments = 2
    total_video_duration = 10
    base_script_prompt = f"The video will be {total_video_duration} seconds long; divide your script into {num_segments} segments of approximately 5 seconds each. Each segment should be 5-8 words. Label each section clearly as '1:', and '2:'. "
elif video_length_option == "15 seconds":
    num_segments = 3
    total_video_duration = 15
    base_script_prompt = f"The video will be {total_video_duration} seconds long; divide your script into {num_segments} segments of approximately 5 seconds each. Each segment should be 6-10 words. Label each section clearly as '1:', '2:', and '3:'. "
else: # Default to 20 seconds
    num_segments = 4
    total_video_duration = 20
    base_script_prompt = f"The video will be {total_video_duration} seconds long; divide your script into {num_segments} segments of approximately 5 seconds each. Each segment should be 6-10 words. Label each section clearly as '1:', '2:', '3:', and '4:'. "

# Adjust prompts based on video category
if video_category == "Educational":
    script_prompt_template = (
        f"You are an expert video scriptwriter. Write a clear, engaging, thematically consistent voiceover script for a {total_video_duration}-second educational video titled '{{video_topic}}'. "
        f"{base_script_prompt}"
        f"Make sure the {num_segments} segments tell a cohesive, progressive story that builds toward a compelling conclusion. "
        f"Use vivid, concrete language that translates well to visuals. Include specific details, numbers, or comparisons when relevant. "
        f"Write in an engaging, conversational tone that keeps viewers hooked. Avoid generic statements."
    )
    video_visual_style_prompt = f"educational video about '{{video_topic}}'. Style: {video_style.lower()}, clean, professional, well-lit. Camera movement: smooth, purposeful. No text overlays."
    music_style_prompt = f"Background music for a cohesive, {total_video_duration}-second educational video about {{video_topic}}. Light, non-distracting, slightly cinematic tone."

elif video_category == "Advertisement":
    script_prompt_template = (
        f"You are an expert video scriptwriter. Write a compelling, persuasive script for a {total_video_duration}-second **advertisement** about '{{video_topic}}'. "
        f"{base_script_prompt}"
        f"Focus on benefits, problem-solution, and a clear call to action. Each segment should highlight a key feature, benefit, or evoke a positive emotion. "
        f"The final segment should include a strong call to action (e.g., 'Learn more at...', 'Buy now!', 'Visit our website!'). "
        f"Use a professional, enticing, and slightly urgent tone. Avoid generic statements."
    )
    video_visual_style_prompt = f"dynamic, visually appealing shots for a product/service advertisement about '{{video_topic}}'. Highlight features. Style: modern, vibrant, clean, commercial-ready. Camera movement: engaging, product-focused. No text overlays."
    music_style_prompt = f"Upbeat, modern, and catchy background music for a commercial advertisement about {{video_topic}}. Energetic and positive tone."

elif video_category == "Movie Trailer":
    script_prompt_template = (
        f"You are an expert video scriptwriter. Write a dramatic, suspenseful script for a {total_video_duration}-second **movie trailer** for a film titled '{{video_topic}}'. "
        f"{base_script_prompt}"
        f"Each segment should introduce elements of the plot, characters, or rising conflict, building suspense. "
        f"The final segment should be a compelling, open-ended hook that leaves the audience wanting more. "
        f"Use evocative language, questions, and a fast-paced, intense tone. Build anticipation."
    )
    video_visual_style_prompt = f"epic, dramatic, cinematic shots for a movie trailer about '{{video_topic}}'. Emphasize tension, conflict, character expressions. Style: dark, moody, high-contrast, blockbuster film. Camera movement: intense, sweeping, purposeful. No text overlays."
    music_style_prompt = f"Dramatic, suspenseful, and epic background music for a movie trailer about {{video_topic}}. Build tension and excitement with orchestral elements."


# Section for Camera Movement options
st.subheader("Camera Movement (Optional)")
# List of available camera movement concepts
camera_concepts = [
    "static", "zoom_in", "zoom_out", "pan_left", "pan_right",
    "tilt_up", "tilt_down", "orbit_left", "orbit_right",
    "push_in", "pull_out", "crane_up", "crane_down",
    "aerial", "aerial_drone", "handheld", "dolly_zoom"
]

# Multiselect for choosing camera movements
selected_concepts = st.multiselect(
    "Choose camera movements (will be applied randomly to segments):",
    options=camera_concepts,
    default=["static", "zoom_in", "pan_right"],
    help="Select camera movements to make your video more dynamic"
)

# Helper function to sanitize string for API calls
def sanitize_for_api(text_string):
    """Encodes a string to ASCII, ignoring errors, then decodes back to string.
    This removes any non-ASCII characters that might cause UnicodeEncodeError."""
    if isinstance(text_string, str):
        return text_string.encode('ascii', 'ignore').decode('ascii')
    return str(text_string).encode('ascii', 'ignore').decode('ascii')


# Main generation button, dynamically displays the selected video length
if replicate_api_key and video_topic and st.button(f"Generate {video_length_option} Video"):
    # Initialize Replicate client with the provided API key
    replicate_client = replicate.Client(api_token=replicate_api_key)

    # Helper function to run Replicate models
    def run_replicate(model_path, input_data):
        return replicate_client.run(model_path, input=input_data)

    # Helper function to download content from a URL to a temporary file
    def download_to_file(url: str, suffix: str):
        resp = requests.get(url, stream=True)
        resp.raise_for_status() # Raise an exception for bad status codes
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with open(tmp.name, "wb") as f:
            for chunk in resp.iter_content(1024 * 32): # Download in chunks
                f.write(chunk)
        return tmp.name

    # Step 1: Write the cohesive script for the full video
    st.info(f"Step 1: Writing cohesive script for {total_video_duration}-second {video_category} video")
    
    # Sanitize the prompt before sending to Replicate
    sanitized_script_prompt = sanitize_for_api(script_prompt_template.format(video_topic=video_topic))
    full_script = run_replicate(
        "anthropic/claude-4-sonnet",
        {
            "prompt": sanitized_script_prompt
        },
    )

    # Process the script output from Replicate
    script_text = "".join(full_script) if isinstance(full_script, list) else full_script
    # Extract script segments using regex (e.g., "1: First segment", "2: Second segment")
    script_segments = re.findall(r"\d+:\s*(.+)", script_text)

    # Validate that the correct number of segments were extracted
    if len(script_segments) < num_segments:
        st.error(f"Failed to extract {num_segments} clear script segments. Try adjusting your topic or refining the prompt.")
        st.stop()
    # Ensure we only take the exact number of required segments
    script_segments = script_segments[:num_segments]

    st.success("Script written successfully")
    # Save the script to a temporary file and provide a download button
    script_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(script_file_path, "w") as f:
        f.write("\n\n".join(script_segments))
    st.download_button("Download Script", script_file_path, "script.txt")

    # NEW Step 2: Generate voiceover narration directly after script
    voice_path = None
    if include_voiceover:
        st.info(f"Step 2: Generating voiceover narration with {selected_voice} voice")
        full_narration = " ".join(script_segments)
        # Remove punctuation and special characters from the narration for cleaner VoiceOver
        # Keep common punctuation marks like periods, commas, question marks, exclamation marks
        cleaned_narration = re.sub(r'[^\w\s.,!?]', '', full_narration)
        
        # Add a check for empty narration after cleaning
        if not cleaned_narration.strip():
            st.warning("Voiceover script became empty after cleaning. Skipping voiceover generation.")
            voice_path = None
        else:
            try:
                # Sanitize the narration text before sending to Replicate
                sanitized_narration_text = sanitize_for_api(cleaned_narration)
                # Run the speech generation model - NOW USING 'minimax/speech-02-turbo'
                voiceover_uri = run_replicate(
                    "minimax/speech-02-turbo",
                    {
                        "text": sanitized_narration_text,
                        "voice_id": voice_options[selected_voice],
                        "emotion": selected_emotion,
                        "speed": 1.1,
                        "pitch": 0,
                        "volume": 1,
                        "bitrate": 128000,
                        "channel": "mono",
                        "sample_rate": 32000,
                        "language_boost": "English",
                        "english_normalization": True
                    },
                )
                # Download the generated voiceover
                voice_path = download_to_file(voiceover_uri, suffix=".mp3")

                # Validate if the downloaded voiceover file is empty or missing
                if not os.path.exists(voice_path) or os.path.getsize(voice_path) == 0:
                    st.error("Generated voiceover file is empty or missing. It might not have generated correctly on Replicate's side.")
                    voice_path = None
                else:
                    # Display the voiceover and provide a download button
                    st.audio(voice_path)
                    st.download_button("Download Voiceover", voice_path, "voiceover.mp3")

            except Exception as e:
                st.error(f"Failed to generate or download voiceover: {e}")
                voice_path = None # Set to None if generation fails, so it's not used later


    temp_video_paths = []
    segment_clips = []

    # Original Step 2 becomes NEW Step 3: Generate segment visuals for each script segment
    for i, segment in enumerate(script_segments):
        st.info(f"Step 3.{i+1}: Generating visuals for segment {i+1}")
        # Determine shot type based on segment index for cinematic variety
        if i == 0:
            shot_type = "establishing wide shot"
        elif i == 1 and num_segments > 2: # Only apply if there are more than 2 segments
            shot_type = "medium shot with focus on key elements"
        elif i == 2 and num_segments > 3: # Only apply if there are more than 3 segments
            shot_type = "close-up shot showing important details"
        else: # For the last segment or if fewer segments, make it a concluding shot
            shot_type = "dynamic concluding shot"

        # Construct the video generation prompt using the category-specific visual style
        video_prompt = f"Cinematic {shot_type} for {video_visual_style_prompt.format(video_topic=video_topic)}. Visual content: {segment}."
        
        # Sanitize the video prompt before sending to Replicate
        sanitized_video_prompt = sanitize_for_api(video_prompt)
        try:
            # Run the video generation model
            video_uri = run_replicate(
                "luma/ray-flash-2-540p",
                {
                    "prompt": sanitized_video_prompt,
                    "num_frames": num_frames,
                    "fps": 24,
                    "guidance": 3.0,  # Higher guidance for better prompt adherence
                    "num_inference_steps": 30  # More steps for better quality
                },
            )
            # Download the generated video
            video_path = download_to_file(video_uri, suffix=".mp4")
            temp_video_paths.append(video_path)

            # Create a VideoFileClip from the downloaded video, ensuring 5s per segment
            clip = VideoFileClip(video_path).subclip(0, 5)
            segment_clips.append(clip)

            # Display the generated video and provide a download button
            st.video(video_path)
            st.download_button(f"Download Segment {i+1}", video_path, f"segment_{i+1}.mp4")
        except Exception as e:
            st.error(f"Failed to generate or download segment {i+1} video: {e}")
            st.stop() # Stop execution if a segment video fails to generate

    # Original Step 3 becomes NEW Step 4: Concatenate video segments first
    st.info("Step 4: Combining video segments")
    try:
        # Concatenate all generated video clips
        final_video = concatenate_videoclips(segment_clips, method="compose")
        # Ensure the final video duration matches the selected total length
        final_video = final_video.set_duration(total_video_duration)
        final_duration = final_video.duration
        st.success(f"Video segments combined - Total duration: {final_duration} seconds")
    except Exception as e:
        st.error(f"Failed to combine video segments: {e}")
        st.stop()

    # Original Step 4 becomes NEW Step 5: Generate background music
    st.info("Step 5: Creating background music")
    music_path = None
    try:
        # Sanitize the music prompt before sending to Replicate
        sanitized_music_prompt = sanitize_for_api(music_style_prompt.format(video_topic=video_topic))
        # Run the music generation model
        music_uri = run_replicate(
            "google/lyria-2",
            {"prompt": sanitized_music_prompt},
        )
        # Download the generated music
        music_path = download_to_file(music_uri, suffix=".mp3")
        # Display the music and provide a download button
        st.audio(music_path)
        st.download_button("Download Background Music", music_path, "background_music.mp3")
    except Exception as e:
        st.error(f"Failed to generate or download music: {e}")
        music_path = None # Set to None if generation fails

    # Original Step 6 becomes NEW Step 6: Merge all audio with video
    st.info("Step 6: Merging final audio and video")
    try:
        audio_clips = []
        
        # Handle voiceover audio - fit exactly to video duration
        if voice_path:
            try:
                voice_clip = AudioFileClip(voice_path)
                st.write(f"DEBUG: Voiceover clip duration after loading: {voice_clip.duration} seconds (Target: {final_duration})")
                voice_volume = 1.0
                voice_clip = voice_clip.volumex(voice_volume)
                
                # Fit voiceover to exact video duration
                if voice_clip.duration > final_duration:
                    # Trim if too long (from the end to preserve beginning)
                    voice_clip = voice_clip.subclip(0, final_duration)
                    st.write(f"DEBUG: Voiceover trimmed. New duration: {voice_clip.duration} seconds.")
                elif voice_clip.duration < final_duration:
                    # Pad with silence at the end if too short
                    from moviepy.audio.AudioClip import AudioArrayClip # Moved here for clarity, though already imported globally
                    sr = int(voice_clip.fps)
                    silence_duration = final_duration - voice_clip.duration
                    # Create silence: (samples, 1) for mono
                    # Ensure the silence array is created with the correct shape for moviepy to recognize it as audio
                    # If audio_clip.duration is 0, this can cause issues. Add a check.
                    if silence_duration > 0 and sr > 0:
                        silence = np.zeros((int(silence_duration * sr), 1), dtype=np.float32)
                        silence_clip = AudioArrayClip(silence, fps=sr)
                        voice_clip = concatenate_audioclips([voice_clip, silence_clip])
                        st.write(f"DEBUG: Voiceover padded with {silence_duration} seconds of silence. New duration: {voice_clip.duration} seconds.")
                    else:
                        st.warning(f"DEBUG: Skipping voiceover padding. Silence duration: {silence_duration}, Sample Rate: {sr}")
                        
                audio_clips.append(voice_clip)
            except Exception as e:
                st.error(f"Error loading or processing voiceover audio clip: {e}")
                voice_path = None # Ensure it's not used further if there's an issue

        # Handle background music - fit exactly to video duration
        if music_path:
            try:
                music_clip = AudioFileClip(music_path)
                st.write(f"DEBUG: Music clip duration after loading: {music_clip.duration} seconds (Target: {final_duration})")
                music_volume = 0.2  # Lower music volume for better voice clarity
                # Apply fade effects
                music_clip = music_clip.volumex(music_volume).audio_fadein(0.5).audio_fadeout(2.5)
                
                # Fit music to exact video duration
                if music_clip.duration < final_duration:
                    # Loop music if it's shorter than video
                    loops_needed = int(final_duration / music_clip.duration) + 1
                    music_clips_looped = [music_clip] * loops_needed
                    music_clip = concatenate_audioclips(music_clips_looped)
                    # Trim to exact duration after looping
                    music_clip = music_clip.subclip(0, final_duration)
                    st.write(f"DEBUG: Music looped and trimmed. New duration: {music_clip.duration} seconds.")
                elif music_clip.duration > final_duration:
                    # Trim from beginning to preserve the natural ending
                    music_clip = music_clip.subclip(0, final_duration)
                    st.write(f"DEBUG: Music trimmed. New duration: {music_clip.duration} seconds.")
                    
                audio_clips.append(music_clip)
            except Exception as e:
                st.error(f"Error loading or processing background music clip: {e}")
                music_path = None # Ensure it's not used further if there's an issue

        # Composite all audio clips if any exist, otherwise set no audio
        if audio_clips:
            final_audio = CompositeAudioClip(audio_clips)
            st.write(f"DEBUG: Final composite audio clip duration: {final_audio.duration} seconds.")
            final_video = final_video.set_audio(final_audio)
        else:
            final_video = final_video.set_audio(None) # No audio if nothing was generated
            st.warning("DEBUG: No audio clips generated for final video.")


        # Define output path for the final video
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        # Write the final video file
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            fps=24
        )

        st.success("🎬 Final video with narration and music is ready")
        # Display the final video and provide a download button
        st.video(output_path)
        st.download_button("Download Final Video", output_path, "final_video.mp4")

        # --- Create a zip file with all assets and provide a download button ---
        import zipfile
        asset_paths = []
        asset_names = []

        # Add script
        if os.path.exists(script_file_path):
            asset_paths.append(script_file_path)
            asset_names.append("script.txt")
        # Add video segments
        for idx, seg_path in enumerate(temp_video_paths):
            if os.path.exists(seg_path):
                asset_paths.append(seg_path)
                asset_names.append(f"segment_{idx+1}.mp4")
        # Add voiceover
        if voice_path and os.path.exists(voice_path):
            asset_paths.append(voice_path)
            asset_names.append("voiceover.mp3")
        # Add music
        if music_path and os.path.exists(music_path):
            asset_paths.append(music_path)
            asset_names.append("background_music.mp3")
        # Add final video
        if os.path.exists(output_path):
            asset_paths.append(output_path)
            asset_names.append("final_video.mp4")

        # Create zip file
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
