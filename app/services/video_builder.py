from pathlib import Path
from typing import Union, List, Dict, Any
import numpy as np
import subprocess
import logging

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, vfx
except (ImportError, ModuleNotFoundError):
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip, vfx

logger = logging.getLogger(__name__)


def apply_smart_ducking(audio_clip, speech_segments: List[Dict[str, Any]], background_volume: float = 1.0, duck_volume: float = 0.0):
    """Векторизованный дакинг: глушит русский голос во время реплик."""
    def volume_filter(gf, t):
        chunk = gf(t)
        t_arr = np.asarray(t)
        multipliers = np.full(t_arr.shape, background_volume, dtype=np.float32)
        
        for seg in speech_segments:
            start_t = seg.get("start", 0)
            end_t = seg.get("end", 0)
            mask = (t_arr >= start_t) & (t_arr <= end_t)
            multipliers[mask] = duck_volume
        
        if chunk.ndim == 2:
            return chunk * multipliers[:, None]
        return chunk * multipliers

    return audio_clip.fl(volume_filter)


def build_dubbed_video_with_segments(
    video_path: Union[str, Path],
    speech_segments: List[Dict[str, Any]],
    output_video_path: Union[str, Path],
    background_volume: float = 1.0,
    duck_volume: float = 0.0
) -> Path:
    """
    Супер-быстрый монтаж: рендерит только звук, а видео склеивает через FFmpeg Stream Copy за 0.3 сек.
    """
    video_path = Path(video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    temp_audio_mix = output_video_path.parent / f"temp_mix_{output_video_path.stem}.wav"

    video_clip = None
    audio_clips_to_close = []

    try:
        video_clip = VideoFileClip(str(video_path))
        
        # 1. Сводим фоновые звуки со Smart Ducking
        if video_clip.audio is not None:
            background_audio = apply_smart_ducking(
                video_clip.audio, 
                speech_segments, 
                background_volume=background_volume, 
                duck_volume=duck_volume
            )
        else:
            background_audio = None

        speech_audio_clips = []
        if background_audio is not None:
            speech_audio_clips.append(background_audio)

        # 2. Расставляем фразы перевода
        for seg in speech_segments:
            audio_file = seg.get("audio_file")
            exact_start = seg.get("start", 0.0)
            orig_duration = seg.get("duration", 0.0)

            if audio_file and Path(audio_file).exists():
                clip = AudioFileClip(str(audio_file))
                
                # Подгонка скорости речи (до 1.4x)
                if orig_duration > 0.5 and clip.duration > orig_duration:
                    speed_factor = min(1.4, clip.duration / orig_duration)
                    if hasattr(clip, "fx"):
                        try:
                            clip = clip.fx(vfx.speedx, speed_factor)
                        except Exception:
                            pass

                clip = clip.set_start(exact_start)
                speech_audio_clips.append(clip)
                audio_clips_to_close.append(clip)

        # 3. Сохраняем ТОЛЬКО аудиодорожку (это происходит мгновенно!)
        if speech_audio_clips:
            final_audio = CompositeAudioClip(speech_audio_clips)
            final_audio = final_audio.set_duration(video_clip.duration)
        else:
            final_audio = background_audio

        logger.info("Быстрый рендеринг аудиодорожки...")
        final_audio.write_audiofile(
            str(temp_audio_mix),
            fps=44100,
            nbytes=2,
            codec="pcm_s16le",
            logger=None
        )

        # 4. FFMPEG STREAM COPY: соединяем оригинальное видео и новый звук за 0.3 сек!
        logger.info(f"Мгновенный Stream Copy монтаж через FFmpeg: {output_video_path.name}")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),          # Исходное видео (видеопоток 0)
            "-i", str(temp_audio_mix),      # Новая аудиодорожка (аудиопоток 1)
            "-c:v", "copy",                 # ПРЯМОЕ КОПИРОВАНИЕ ВИДЕО (без пережатия!)
            "-c:a", "aac",                  # Сжатие звука в AAC
            "-map", "0:v:0",                # Берем видео из 1-го файла
            "-map", "1:a:0",                # Берем звук из 2-го файла
            "-shortest",
            str(output_video_path)
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_video_path

    finally:
        for ac in audio_clips_to_close:
            try: ac.close()
            except Exception: pass
        if video_clip is not None:
            try: video_clip.close()
            except Exception: pass
            
        # Удаляем промежуточный файл микса
        if temp_audio_mix.exists():
            try: temp_audio_mix.unlink()
            except Exception: pass