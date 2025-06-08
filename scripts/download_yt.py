import os
import yt_dlp

def download_best_video(url, output_dir="raw_videos"):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',  # Ensures video+audio merged
        'outtmpl': os.path.join(output_dir, '%(title).80s.%(ext)s'),
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ").strip()
    download_best_video(video_url)