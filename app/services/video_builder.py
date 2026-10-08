from pathlib import Path
from typing import Union, List, Dict, Any
import numpy as np
import logging

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, vfx
except (ImportError, ModuleNotFoundError):
    from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip, vfx

logger = logging.getLogger(__name__)


def apply_smart_ducking(audio_clip, speech_segments: List[Dict[str, Any]], background_volume: float = 1.0, duck_volume: float = 0.0):
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
    video_path = Path(video_path)
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    video_clip = None
    audio_clips_to_close = []

    try:
        video_clip = VideoFileClip(str(video_path))
   
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

        max_audio_end = video_clip.duration

        for seg in speech_segments:
            audio_file = seg.get("audio_file")
            exact_start = seg.get("start", 0.0)
            orig_duration = seg.get("duration", 0.0)

            if audio_file and Path(audio_file).exists():
                clip = AudioFileClip(str(audio_file))
                
                if orig_duration > 0.5 and clip.duration > orig_duration:
                    speed_factor = clip.duration / orig_duration
                    clamped_factor = min(1.45, speed_factor)
                    if hasattr(clip, "fx"):
                        try:
                            clip = clip.fx(vfx.speedx, clamped_factor)
                        except Exception:
                            pass

                clip = clip.set_start(exact_start)
                speech_audio_clips.append(clip)
                audio_clips_to_close.append(clip)

                if (exact_start + clip.duration) > max_audio_end:
                    max_audio_end = exact_start + clip.duration

        if speech_audio_clips:
            final_audio = CompositeAudioClip(speech_audio_clips)
        else:
            final_audio = background_audio

        # Если фраза длилась чуть дольше видео, продлеваем видеоряд на последнем кадре
        if max_audio_end > video_clip.duration:
            video_duration = max_audio_end + 0.3
            if hasattr(video_clip, "set_duration"):
                final_video = video_clip.set_duration(video_duration).set_audio(final_audio)
            else:
                final_video = video_clip.with_duration(video_duration).with_audio(final_audio)
        else:
            final_video = video_clip.set_audio(final_audio)

        logger.info(f"Сборка видео (Без обрезания реплик): {output_video_path}")
        final_video.write_videofile(
            str(output_video_path),
            codec="libx264",
            audio_codec="aac",
            fps=video_clip.fps if video_clip.fps else 24,
            logger=None
        )

        return output_video_path

    finally:
        for ac in audio_clips_to_close:
            try: ac.close()
            except Exception: pass
        if video_clip is not None:
            try: video_clip.close()
            except Exception: pass