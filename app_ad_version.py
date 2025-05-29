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
)

st.title("AI Multi-Agent Ad Creator")

replicate_api_key = st.text_input("Enter your Replicate API Key", type="password")

# Ad-specific inputs
col1, col2 = st.columns(2)
with col1:
    product_name = st.text_input("Product/Service Name", placeholder="e.g., 'EcoClean Detergent'")
    target_audience = st.selectbox("Target Audience", [
        "Young Adults (18-35)", 
        "Families with Children", 
        "Professionals", 
        "Seniors (55+)", 
        "Tech Enthusiasts",
        "Health & Fitness"
    ])

with col2:
    ad_tone = st.selectbox("Ad Tone", [
        "Exciting & Energetic",
        "Warm & Friendly", 
        "Professional & Trustworthy",
        "Fun & Playful",
        "Luxury & Premium",
        "Urgent & Action-Driven"
    ])
    call_to_action = st.text_input("Call to Action", placeholder="e.g., 'Visit our website today!'")

key_benefits = st.text_area("Key Benefits/Features (1-3 main points)", 
                           placeholder="e.g., '99% effective cleaning, eco-friendly, saves time'")

if replicate_api_key and product_name and key_benefits and st.button("Generate 20s Ad"):
    replicate_client = replicate.Client(api_token=replicate_api_key)

    def run_replicate(model_path, input_data):
        return replicate_client.run(model_path, input=input_data)

    st.info("Step 1: Writing compelling ad script")
    
    # Enhanced ad script prompt
    ad_script_prompt = f"""You are an expert advertising copywriter. Write a compelling, persuasive 20-second video ad script for '{product_name}'.

Target Audience: {target_audience}
Tone: {ad_tone}
Key Benefits: {key_benefits}
Call to Action: {call_to_action}

Create a 4-segment script (5 seconds each) that follows this structure:
1: Hook/Problem - Grab attention with a relatable problem or exciting opening
2: Solution - Introduce the product as the perfect solution
3: Benefits - Highlight the key benefits that matter to the target audience
4: Call to Action - Strong, compelling call to action with urgency

Keep each segment to 6-8 words maximum for clear delivery. Make it persuasive and memorable.
Label each section as '1:', '2:', '3:', and '4:'."""

    full_script = run_replicate(
        "anthropic/claude-4-sonnet",
        {"prompt": ad_script_prompt}
    )

    script_text = "".join(full_script) if isinstance(full_script, list) else full_script
    script_segments = re.findall(r"\d+:\s*(.+)", script_text)

    if len(script_segments) < 4:
        st.error("Failed to extract 4 clear script segments. Try adjusting your inputs.")
        st.stop()

    st.success("Ad script written successfully")
    st.write("**Generated Script:**")
    for i, segment in enumerate(script_segments):
        st.write(f"**Segment {i+1}:** {segment}")
    
    script_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(script_file_path, "w") as f:
        f.write(f"Ad Script for: {product_name}\n")
        f.write(f"Target: {target_audience}\n")
        f.write(f"Tone: {ad_tone}\n\n")
        f.write("\n\n".join([f"Segment {i+1}: {seg}" for i, seg in enumerate(script_segments)]))
    st.download_button("📜 Download Ad Script", script_file_path, "ad_script.txt")

    temp_video_paths = []

    def download_to_file(url: str, suffix: str):
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        with open(tmp.name, "wb") as f:
            for chunk in resp.iter_content(1024 * 32):
                f.write(chunk)
        return tmp.name

    segment_clips = []

    # Step 2: Generate ad visuals with commercial style
    visual_styles = {
        "Exciting & Energetic": "dynamic, high-energy, vibrant colors, fast-paced",
        "Warm & Friendly": "warm lighting, friendly faces, cozy atmosphere",
        "Professional & Trustworthy": "clean, professional, modern office setting",
        "Fun & Playful": "bright, colorful, animated, joyful expressions",
        "Luxury & Premium": "elegant, sophisticated, high-end materials, golden lighting",
        "Urgent & Action-Driven": "dramatic, bold, intense, action-packed"
    }
    
    style_description = visual_styles.get(ad_tone, "professional, appealing")

    for i, segment in enumerate(script_segments):
        st.info(f"Step 2.{i+1}: Generating commercial visuals for segment {i+1}")
        
        # Ad-specific visual prompts
        if i == 0:  # Hook/Problem
            video_prompt = f"Commercial ad opening scene: {style_description}. Scene showing the problem or hook for {product_name}. {segment}"
        elif i == 1:  # Solution  
            video_prompt = f"Commercial ad scene: {style_description}. Product showcase for {product_name}, revealing the solution. {segment}"
        elif i == 2:  # Benefits
            video_prompt = f"Commercial ad scene: {style_description}. Demonstrating benefits of {product_name} in action. {segment}"
        else:  # Call to Action
            video_prompt = f"Commercial ad finale: {style_description}. Strong call-to-action scene for {product_name}. {segment}"
            
        try:
            video_uri = run_replicate(
                "luma/ray-flash-2-540p",
                {"prompt": video_prompt, "num_frames": 120, "fps": 24},
            )
            video_path = download_to_file(video_uri, suffix=".mp4")
            temp_video_paths.append(video_path)

            # Ensure exactly 5s per segment with proper handling
            clip = VideoFileClip(video_path)
            if clip.duration >= 5:
                clip = clip.subclip(0, 5)
            else:
                # If clip is shorter than 5s, loop it to reach 5s
                loops_needed = int(5 / clip.duration) + 1
                clip = concatenate_videoclips([clip] * loops_needed).subclip(0, 5)
            
            segment_clips.append(clip)

            st.video(video_path)
            st.download_button(f"🎥 Download Segment {i+1}", video_path, f"ad_segment_{i+1}.mp4")
        except Exception as e:
            st.error(f"Failed to generate segment {i+1} visuals: {e}")
            st.stop()

    # Step 4: Generate professional voiceover
    st.info("Step 4: Generating professional ad voiceover")
    full_narration = " ".join(script_segments)
    
    # Add voiceover direction based on tone
    voice_direction = {
        "Exciting & Energetic": "enthusiastic, high-energy",
        "Warm & Friendly": "warm, conversational", 
        "Professional & Trustworthy": "authoritative, confident",
        "Fun & Playful": "upbeat, cheerful",
        "Luxury & Premium": "sophisticated, smooth",
        "Urgent & Action-Driven": "urgent, compelling"
    }.get(ad_tone, "professional")
    
    try:
        voiceover_uri = run_replicate(
            "minimax/speech-02-hd",
            {
                "text": f"[{voice_direction} tone] {full_narration}",
                "voice": "default"
            },
        )
        voice_path = download_to_file(voiceover_uri, suffix=".mp3")
        st.audio(voice_path)
        st.download_button("🎙 Download Ad Voiceover", voice_path, "ad_voiceover.mp3")
    except Exception as e:
        st.error(f"Failed to generate voiceover: {e}")
        st.stop()

    # Step 5: Generate commercial background music
    st.info("Step 5: Creating commercial background music")
    
    music_styles = {
        "Exciting & Energetic": "upbeat electronic, driving beat, energetic",
        "Warm & Friendly": "acoustic, warm, feel-good melody",
        "Professional & Trustworthy": "corporate, inspiring, confidence-building",
        "Fun & Playful": "upbeat, playful, catchy melody",
        "Luxury & Premium": "elegant orchestral, sophisticated, premium",
        "Urgent & Action-Driven": "dramatic, intense, building tension"
    }
    
    music_style = music_styles.get(ad_tone, "commercial, professional")
    
    try:
        music_uri = run_replicate(
            "google/lyria-2",
            {
                "prompt": f"Commercial ad background music: {music_style}. 20-second instrumental track for {product_name} advertisement. Professional quality, suitable for TV commercial."
            },
        )
        music_path = download_to_file(music_uri, suffix=".mp3")
        st.audio(music_path)
        st.download_button("🎵 Download Ad Music", music_path, "ad_background_music.mp3")
    except Exception as e:
        st.error(f"Failed to generate background music: {e}")
        st.stop()

    # Step 6: Create final commercial with improved audio/video sync
    st.info("Step 6: Assembling final commercial")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 6a: Concatenate video clips
        status_text.text("Concatenating video segments...")
        progress_bar.progress(10)
        
        final_video = concatenate_videoclips(segment_clips, method="compose")
        target_duration = 20.0
        
        status_text.text("Adjusting video duration...")
        progress_bar.progress(20)
        
        if final_video.duration > target_duration:
            final_video = final_video.subclip(0, target_duration)
        elif final_video.duration < target_duration:
            final_video = final_video.set_duration(target_duration)

        # Step 6b: Load audio clips
        status_text.text("Loading audio files...")
        progress_bar.progress(30)
        
        voice_clip = AudioFileClip(voice_path)
        music_clip = AudioFileClip(music_path)
        
        voice_duration = voice_clip.duration
        music_duration = music_clip.duration
        video_duration = final_video.duration
        
        st.write(f"Debug info - Video: {video_duration:.2f}s, Voice: {voice_duration:.2f}s, Music: {music_duration:.2f}s")
        
        # Step 6c: Sync audio durations with proper padding
        status_text.text("Synchronizing audio durations...")
        progress_bar.progress(40)
        
        from moviepy.audio.AudioClip import AudioArrayClip
        import numpy as np
        
        # Handle voice clip duration
        if voice_duration > video_duration:
            voice_clip = voice_clip.subclip(0, video_duration)
        elif voice_duration < video_duration:
            # Create silence to pad the voice clip
            silence_duration = video_duration - voice_duration
            silence_array = np.zeros((int(silence_duration * 22050), 2))  # 22050 Hz stereo silence
            silence_clip = AudioArrayClip(silence_array, fps=22050)
            
            # Concatenate voice with silence
            from moviepy.audio.io.AudioFileClip import concatenate_audioclips
            voice_clip = concatenate_audioclips([voice_clip, silence_clip])
            
        # Handle music clip duration  
        if music_duration > video_duration:
            music_clip = music_clip.subclip(0, video_duration)
        elif music_duration < video_duration:
            # Loop music to fill duration
            loops_needed = int(video_duration / music_duration) + 1
            music_clip = concatenate_audioclips([music_clip] * loops_needed)
            music_clip = music_clip.subclip(0, video_duration)
        
        # Step 6d: Create composite audio with explicit duration control
        status_text.text("Mixing audio tracks...")
        progress_bar.progress(50)
        
        # Ensure exact durations before mixing
        voice_clip = voice_clip.subclip(0, video_duration)
        music_clip = music_clip.subclip(0, video_duration).volumex(0.25)
        
        final_audio = CompositeAudioClip([voice_clip, music_clip])
        # Force exact duration with subclip instead of set_duration
        final_audio = final_audio.subclip(0, video_duration)
        
        # Step 6e: Combine video and audio
        status_text.text("Combining video and audio...")
        progress_bar.progress(60)
        
        final_video = final_video.set_audio(final_audio)
        
        # Step 6f: Try multiple encoding approaches
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        
        # First try: Standard encoding
        status_text.text("Encoding final video (attempt 1/3)...")
        progress_bar.progress(70)
        
        try:
            final_video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                fps=24,
                bitrate="2000k",
                verbose=False,
                logger=None,
                preset='ultrafast'  # Faster encoding
            )
            encoding_success = True
        except Exception as encoding_error:
            st.warning(f"Standard encoding failed: {str(encoding_error)[:100]}...")
            encoding_success = False
        
        # Second try: Simpler encoding if first failed
        if not encoding_success:
            status_text.text("Encoding final video (attempt 2/3)...")
            progress_bar.progress(80)
            
            try:
                output_path2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                final_video.write_videofile(
                    output_path2,
                    codec="libx264",
                    audio_codec="aac",
                    fps=24,
                    verbose=False,
                    logger=None,
                    preset='ultrafast',
                    threads=1  # Single thread to avoid conflicts
                )
                output_path = output_path2
                encoding_success = True
            except Exception as encoding_error2:
                st.warning(f"Simplified encoding failed: {str(encoding_error2)[:100]}...")
        
        # Third try: Create audio file separately first (most robust)
        if not encoding_success:
            status_text.text("Encoding final video (attempt 3/3 - audio-first method)...")
            progress_bar.progress(90)
            
            try:
                # Step 1: Create the final audio file separately
                temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                final_audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                
                # Step 2: Create video without audio
                video_only = final_video.without_audio()
                temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                video_only.write_videofile(
                    temp_video_path,
                    codec="libx264",
                    fps=24,
                    verbose=False,
                    logger=None,
                    preset='ultrafast'
                )
                
                # Step 3: Load the audio file and combine
                final_audio_loaded = AudioFileClip(temp_audio_path)
                temp_video_loaded = VideoFileClip(temp_video_path)
                
                # Ensure durations match exactly
                final_audio_loaded = final_audio_loaded.subclip(0, temp_video_loaded.duration)
                
                final_combined = temp_video_loaded.set_audio(final_audio_loaded)
                
                output_path3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                final_combined.write_videofile(
                    output_path3,
                    codec="libx264",
                    audio_codec="aac",
                    verbose=False,
                    logger=None,
                    preset='ultrafast'
                )
                
                output_path = output_path3
                encoding_success = True
                
                # Cleanup temp files
                video_only.close()
                temp_video_loaded.close()
                final_audio_loaded.close()
                final_combined.close()
                os.remove(temp_audio_path)
                os.remove(temp_video_path)
                
            except Exception as encoding_error3:
                st.error(f"Audio-first encoding failed: {str(encoding_error3)[:200]}...")
                encoding_success = False

        progress_bar.progress(100)
        
        if encoding_success:
            status_text.text("✅ Commercial assembly complete!")
            st.success("🎬 Your 20-second commercial is ready!")
            st.video(output_path)
            
            # Summary of created ad
            st.write("**Ad Summary:**")
            st.write(f"**Product:** {product_name}")
            st.write(f"**Target Audience:** {target_audience}")
            st.write(f"**Tone:** {ad_tone}")
            st.write(f"**Key Message:** {key_benefits}")
            
            st.download_button("📽 Download Final Commercial", output_path, f"{product_name.replace(' ', '_')}_ad.mp4")
        else:
            st.error("❌ Final video encoding failed after multiple attempts.")
            st.info("💡 You can still download the individual components and combine them manually using video editing software.")

        # Clean up clips to free memory
        try:
            final_video.close()
            voice_clip.close()
            music_clip.close()
            for clip in segment_clips:
                clip.close()
        except:
            pass  # Ignore cleanup errors

    except Exception as e:
        status_text.text("❌ Assembly failed")
        progress_bar.progress(0)
        st.warning("Final commercial assembly failed, but you can still download individual components.")
        st.error(f"Error details: {str(e)[:200]}...")
        
        # Provide manual assembly instructions
        st.info("""
        **Manual Assembly Option:**
        1. Download all individual components above
        2. Use video editing software like DaVinci Resolve (free) or Adobe Premiere
        3. Import all 4 video segments and arrange them in sequence
        4. Add the voiceover audio track
        5. Add background music at 25% volume
        6. Export as MP4
        """)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Cleanup temporary files
    for path in (*temp_video_paths, voice_path, music_path, script_file_path):
        try:
            os.remove(path)
        except OSError:
            pass

# Add helpful tips section
with st.expander("💡 Tips for Better Ads"):
    st.write("""
    **Script Tips:**
    - Keep benefits customer-focused (what's in it for them?)
    - Use action words and emotional triggers
    - Make your call-to-action specific and urgent
    
    **Visual Tips:**
    - Show the product in use, not just static shots
    - Include people who represent your target audience
    - Use consistent branding colors and style
    
    **Audio Tips:**
    - Match voice tone to your brand personality
    - Keep music volume low enough that narration is clear
    - End with a memorable audio signature if possible
    
    **Troubleshooting:**
    - If video assembly fails, try downloading individual components
    - Check that all generated content meets expected durations
    - Ensure stable internet connection for AI model calls
    """)

# Add troubleshooting section
with st.expander("🔧 Common Issues & Solutions"):
    st.write("""
    **Duration Mismatch Errors:**
    - The app now automatically handles audio/video sync issues
    - All clips are normalized to exact durations before assembly
    
    **Memory Issues:**
    - Close browser tabs if experiencing slowdowns
    - The app now properly closes video clips after use
    
    **API Errors:**
    - Verify your Replicate API key is correct
    - Check your Replicate account has sufficient credits
    - Some models may have temporary availability issues
    """)
