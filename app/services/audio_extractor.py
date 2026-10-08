from pathlib import Path
from typing import Union

try:
    from moviepy.editor import VideoFileClip
except (ImportError, ModuleNotFoundError):
    from moviepy import VideoFileClip
    
import logging

logger = logging.getLogger(__name__)

def extract_audio_from_video(
        video_path: Union[str, Path],
        output_audio_path: Union[str, Path])-> Path:
    """
    Извлекает аудиодорожку из видеофайла и сохраняет ее в формате WAV (16kHz, mono).
    
    :param video_path: Путь к исходному видеофайлу.
    :param output_audio_path: Путь для сохранения извлеченного аудио.
    :return: Path к сохраненному аудиофайлу.
    """
    video_path = Path(video_path)
    output_audio_path = Path(output_audio_path)

    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError (f"Видеофайл не найден!")

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    video = None
    try:
        logger.info(f"Открытие видеофайла: {video_path}")
        video = VideoFileClip(str(video_path))

        if video.audio is None:
            raise ValueError(f"В видеофайле отсутствует аудиодорожка: {video_path}")

        logger.info(f"Экспорт аудио в: {output_audio_path}")
        video.audio.write_audiofile(
            str(output_audio_path),
            fps=16000,
            nbytes=2,
            codec="pcm_s16le",
            logger=None  
        )

        return output_audio_path

    finally:
        if video is not None:
            if video.audio is not None:
                video.audio.close()
            video.close()
            logger.info("Ресурсы видео и аудио успешно освобождены.")