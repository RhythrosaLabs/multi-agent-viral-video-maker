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

# Set Streamlit page configuration for a wider layout and custom title
st.set_page_config(layout="wide", page_title="AI Multi-Agent Video Creator")

# Main title of the application
st.title("AI Multi-Agent Video Creator")

# Input fields for Replicate API Key and video topic
replicate_api_key = st.text_input("Enter your Replicate API Key", type="password")
video_topic = st.text_input("Enter a video topic (e.g., 'Why the Earth rotates')")

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

if video_length_option == "10 seconds":
    num_segments = 2
    total_video_duration = 10
    # Script prompt for a super short, 10-second video
    script_prompt_template = (
        f"You are an expert video scriptwriter. Write a clear, engaging, thematically consistent voiceover script for a {total_video_duration}-second educational video titled '{{video_topic}}'. "
        f"The video will be {total_video_duration} seconds long; divide your script into {num_segments} segments of approximately 5 seconds each. "
        f"Each segment should be 5-8 words. "
        f"Make sure the {num_segments} segments tell a cohesive, progressive story that builds toward a compelling conclusion. "
        f"Use vivid, concrete language that translates well to visuals. Include specific details, numbers, or comparisons when relevant. "
        f"Label each section clearly as '1:', and '2:'. "
        f"Write in an engaging, conversational tone that keeps viewers hooked. Avoid generic statements."
    )
elif video_length_option == "15 seconds":
    num_segments = 3
    total_video_duration = 15
    # Script prompt for a short, 15-second video
    script_prompt_template = (
        f"You are an expert video scriptwriter. Write a clear, engaging, thematically consistent voiceover script for a {total_video_duration}-second educational video titled '{{video_topic}}'. "
        f"The video will be {total_video_duration} seconds long; divide your script into {num_segments} segments of approximately 5 seconds each. "
        f"Each segment should be 6-10 words. "
        f"Make sure the {num_segments} segments tell a cohesive, progressive story that builds toward a compelling conclusion. "
        f"Use vivid, concrete language that translates well to visuals. Include specific details, numbers, or comparisons when relevant. "
        f"Label each section clearly as '1:', '2:', and '3:'. "
        f"Write in an engaging, conversational tone that keeps viewers hooked. Avoid generic statements."
    )
else: # Default to 20 seconds
    num_segments = 4
    total_video_duration = 20
    # Script prompt for a 20-second video
    script_prompt_template = (
        f"You are an expert video scriptwriter. Write a clear, engaging, thematically consistent voiceover script for a {total_video_duration}-second educational video titled '{{video_topic}}'. "
        f"The video will be {total_video_duration} seconds long; divide your script into {num_segments} segments of approximately 5 seconds each. "
        f"Each segment should be 6-10 words. "
        f"Make sure the {num_segments} segments tell a cohesive, progressive story that builds toward a compelling conclusion. "
        f"Use vivid, concrete language that translates well to visuals. Include specific details, numbers, or comparisons when relevant. "
        f"Label each section clearly as '1:', '2:', '3:', and '4:'. "
        f"Write in an engaging, conversational tone that keeps viewers hooked. Avoid generic statements."
    )

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

# Main generation button, dynamically displays the selected video length
if replicate_api_key and video_topic and st.button(f"Generate {video_length_option} Video"):
    # Initialize Replicate client with the provided API key
    replicate_client = replicate.Client(api_token=replicate_api_key)

    # Helper function to run Replicate models
    def run_replicate(model_path, input_data):
        return replicate_client.run(model_path, input=input_data)

    # Step 1: Write the cohesive script for the full video
    st.info(f"Step 1: Writing cohesive script for {total_video_duration}-second video")
    full_script = run_replicate(
        "anthropic/claude-4-sonnet",
        {
            "prompt": script_prompt_template.format(video_topic=video_topic)
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
    st.download_button("📜 Download Script", script_file_path, "script.txt")

    temp_video_paths = []
    segment_clips = []

    # Helper function to download content from a URL to a temporary file
    def download_to_file(url: str, suffix: str):
        resp = requests.get(url, stream=True)
        resp.raise_for_status() # Raise an exception for bad status codes
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with open(tmp.name, "wb") as f:
            for chunk in resp.iter_content(1024 * 32): # Download in chunks
                f.write(chunk)
        return tmp.name

    # Step 2: Generate segment visuals for each script segment
    for i, segment in enumerate(script_segments):
        st.info(f"Step 2.{i+1}: Generating visuals for segment {i+1}")
        # Determine shot type based on segment index for cinematic variety
        if i == 0:
            shot_type = "establishing wide shot"
        elif i == 1 and num_segments > 2: # Only apply if there are more than 2 segments
            shot_type = "medium shot with focus on key elements"
        elif i == 2 and num_segments > 3: # Only apply if there are more than 3 segments
            shot_type = "close-up shot showing important details"
        else: # For the last segment or if fewer segments, make it a concluding shot
            shot_type = "dynamic concluding shot"

        # Construct the video generation prompt
        video_prompt = f"Cinematic {shot_type} for educational video about '{video_topic}'. Visual content: {segment}. Style: {video_style.lower()}, clean, professional, well-lit. Camera movement: smooth, purposeful. No text overlays."
        try:
            # Run the video generation model
            video_uri = run_replicate(
                "luma/ray-flash-2-540p",
                {
                    "prompt": video_prompt,
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
            st.download_button(f"🎥 Download Segment {i+1}", video_path, f"segment_{i+1}.mp4")
        except Exception as e:
            st.error(f"Failed to generate or download segment {i+1} video: {e}")
            st.stop() # Stop execution if a segment video fails to generate

    voice_path = None # Initialize voice_path to None
    if include_voiceover:
        # Step 4: Generate voiceover narration if the checkbox is selected
        st.info(f"Step 4: Generating voiceover narration with {selected_voice} voice")
        full_narration = " ".join(script_segments)
        # Remove punctuation and special characters from the narration for cleaner VoiceOver
        cleaned_naration = re.sub(r'[^\w\s]', '', full_narration)
        try:
            # Run the speech generation model
            voiceover_uri = run_replicate(
                "minimax/speech-02-hd",
                {
                    "text": cleaned_naration,
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
            # Display the voiceover and provide a download button
            st.audio(voice_path)
            st.download_button("🎙 Download Voiceover", voice_path, "voiceover.mp3")
        except Exception as e:
            st.error(f"Failed to generate or download voiceover: {e}")
            voice_path = None # Set to None if generation fails, so it's not used later

    music_path = None # Initialize music_path to None
    # Step 5: Generate background music
    st.info("Step 5: Creating background music")
    try:
        # Run the music generation model
        music_uri = run_replicate(
            "google/lyria-2",
            {"prompt": f"Background music for a cohesive, {total_video_duration}-second educational video about {video_topic}. Light, non-distracting, slightly cinematic tone."},
        )
        # Download the generated music
        music_path = download_to_file(music_uri, suffix=".mp3")
        # Display the music and provide a download button
        st.audio(music_path)
        st.download_button("🎵 Download Background Music", music_path, "background_music.mp3")
    except Exception as e:
        st.error(f"Failed to generate or download music: {e}")
        music_path = None # Set to None if generation fails

    # Step 6: Merge audio and video
    st.info("Step 6: Merging final audio and video")
    try:
        # Concatenate all generated video clips
        final_video = concatenate_videoclips(segment_clips, method="compose")
        # Ensure the final video duration matches the selected total length
        final_video = final_video.set_duration(total_video_duration)
        final_duration = final_video.duration

        audio_clips = []
        if voice_path:
            # Load voice clip, set volume, and adjust duration to match video
            voice_clip = AudioFileClip(voice_path)
            voice_volume = 1.0
            voice_clip = voice_clip.volumex(voice_volume)
            if voice_clip.duration > final_duration:
                voice_clip = voice_clip.subclip(0, final_duration)
            elif voice_clip.duration < final_duration:
                # Center the voice in the timeline if it's shorter than the video
                voice_start = (final_duration - voice_clip.duration) / 2
                voice_clip = voice_clip.set_start(voice_start)
            audio_clips.append(voice_clip)

        if music_path:
            # Load music clip, set volume, add fade in/out, and loop/trim to match video duration
            music_clip = AudioFileClip(music_path)
            music_volume = 0.2  # Lower music volume for better voice clarity
            music_clip = music_clip.volumex(music_volume).audio_fadein(1).audio_fadeout(1)
            if music_clip.duration < final_duration:
                # Loop music if its duration is less than the video's
                loops_needed = int(final_duration / music_clip.duration) + 1
                music_clips_looped = [music_clip] * loops_needed
                music_clip = concatenate_audioclips(music_clips_looped).subclip(0, final_duration)
            elif music_clip.duration > final_duration:
                # Trim music if its duration is greater than the video's
                music_clip = music_clip.subclip(0, final_duration)
            audio_clips.append(music_clip)

        # Composite all audio clips if any exist, otherwise set no audio
        if audio_clips:
            final_audio = CompositeAudioClip(audio_clips)
            final_video = final_video.set_audio(final_audio)
        else:
            final_video = final_video.set_audio(None) # No audio if nothing was generated

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
        st.download_button("📽 Download Final Video", output_path, "final_video.mp4")

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
