from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


def youtube_transcript(video_url):
    parsed = urlparse(video_url)

    if parsed.hostname == "youtu.be":
        video_id = parsed.path.lstrip("/")

    elif parsed.hostname in (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com"
    ):
        video_id = parse_qs(parsed.query)["v"][0]

    else:
        raise ValueError("Invalid YouTube URL")

    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)

    return " ".join(
        entry.text
        for entry in transcript
    )

