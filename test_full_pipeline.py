from pathlib import Path
from gtts import gTTS
import numpy as np

# Импортируем универсальный MoviePy
try:
    from moviepy.editor import ColorClip, AudioFileClip
except (ImportError, ModuleNotFoundError):
    from moviepy import ColorClip, AudioFileClip

from app.services.pipeline import run_dubbing_pipeline


def create_sample_russian_video(video_path: Path):
    """Создает полноценный видеоролик с русской озвучкой для проверки дубляжа."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    temp_voice_mp3 = video_path.parent / "temp_voice.mp3"

    russian_phrase = "Привет! Сегодня мы тестируем автоматический перевод видео на английский язык."
    tts = gTTS(text=russian_phrase, lang="ru")
    tts.save(str(temp_voice_mp3))

    voice_audio = AudioFileClip(str(temp_voice_mp3))
    duration = voice_audio.duration

    video_clip = ColorClip(size=(640, 360), color=[40, 44, 52], duration=duration)
    
    if hasattr(video_clip, "with_audio"):
        video_clip = video_clip.with_audio(voice_audio)
    else:
        video_clip = video_clip.set_audio(voice_audio)

    video_clip.write_videofile(
        str(video_path), 
        fps=24, 
        codec="libx264", 
        audio_codec="aac", 
        logger=None
    )
    video_clip.close()
    voice_audio.close()


def run_test():
    print("=== ПОДГОТОВКА: Создание исходного русского видео ===")
    source_video = Path("storage/raw_videos/russian_source_video.mp4")
    create_sample_russian_video(source_video)
    print(f"Исходное видео готово: {source_video}")

    print("\n=== ЗАПУСК СКВОЗНОГО ПАЙПЛАЙНА ДУБЛЯЖА НА АНГЛИЙСКИЙ ===")
    result = run_dubbing_pipeline(
        input_video_path=source_video,
        target_language="en"
    )

    print("\n" + "#" * 50)
    print("ИТОГОВЫЙ ОТЧЕТ ПАЙПЛАЙНА:")
    print("#" * 50)
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("#" * 50)


if __name__ == "__main__":
    run_test()