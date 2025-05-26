# whisper_caption_extractor.py

import os
import whisper
import ffmpeg

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extracts audio from the video using ffmpeg"""
    ffmpeg.input(video_path).output(audio_path, ac=1, ar='16k').run(overwrite_output=True)
    return audio_path

def transcribe_with_whisper(audio_path):
    """Runs Whisper on the extracted audio"""
    model = whisper.load_model("base")  # You can change to "small", "medium", "large" if needed
    result = model.transcribe(audio_path)
    return result["segments"]  # list of dicts with 'start', 'end', 'text'

def save_captions(segments, output_file="captions.txt"):
    """Saves the transcribed segments to a file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for seg in segments:
            start = round(seg['start'], 2)
            end = round(seg['end'], 2)
            text = seg['text'].strip()
            f.write(f"[{start}s - {end}s] {text}\n")
    print(f"Saved captions to {output_file}")

def process_video_for_captions(video_path, output_file="captions.txt"):
    audio_path = extract_audio(video_path)
    segments = transcribe_with_whisper(audio_path)
    save_captions(segments, output_file)
    os.remove(audio_path)  # clean up temp audio

if __name__ == "__main__":
    video_path = "Video/Dessert.mp4"  # Replace with your video path
    process_video_for_captions(video_path)
