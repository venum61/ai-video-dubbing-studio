from pathlib import Path
import numpy as np

try:
    from moviepy.editor import ColorClip, AudioClip
except (ImportError, ModuleNotFoundError):
    from moviepy import ColorClip, AudioClip

from app.services.audio_extractor import extract_audio_from_video


def make_sound(t):
    return np.sin(2 * np.pi * 440 * t)


def run_test():
    print("=== ТЕСТ 1: Генерация синтетического видео ===")
    test_video_path = Path("storage/raw_videos/sample_test.mp4")
    test_video_path.parent.mkdir(parents=True, exist_ok=True)

    clip = ColorClip(size=(320, 240), color=[0, 128, 255], duration=2)
    
    audio = AudioClip(make_sound, duration=2, fps=22050)
    
    if hasattr(clip, "with_audio"):
        clip = clip.with_audio(audio)
    else:
        clip.audio = audio

    clip.write_videofile(
        str(test_video_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )
    clip.close()
    audio.close()
    print(f"Тестовое видео создано: {test_video_path}")

    print("\n=== ТЕСТ 2: Извлечение аудио через наш сервис ===")
    output_audio_path = Path("storage/extracted_audio/sample_test.wav")

    result_path = extract_audio_from_video(test_video_path, output_audio_path)

    if result_path.exists() and result_path.stat().st_size > 0:
        file_size_kb = result_path.stat().st_size / 1024
        print(f"УСПЕХ! Аудиофайл создан: {result_path}")
        print(f"Размер файла: {file_size_kb:.2f} КБ")
    else:
        print("ОШИБКА: Аудиофайл не был создан или пуст!")


if __name__ == "__main__":
    run_test()