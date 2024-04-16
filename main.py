import subprocess


def extract_audio_from_video(video_path, audio_path):
    command = ["/opt/homebrew/bin/ffmpeg", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)


extract_audio_from_video("files/zzoneddeck.mp4", "files/audio.mp3")
