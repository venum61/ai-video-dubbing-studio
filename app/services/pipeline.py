from pathlib import Path
from typing import Union, Dict, Any
import logging
import time

from app.services.audio_extractor import extract_audio_from_video
from app.services.vad_service import analyze_audio_speech
from app.services.voice_analyzer import analyze_speaker_voice
from app.services.transcriber import transcribe_audio
from app.services.translator import translate_segments
from app.services.synthesizer import synthesize_speech
from app.services.video_builder import build_dubbed_video_with_segments

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_dubbing_pipeline(
    input_video_path: Union[str, Path],
    source_language: str = "auto",
    target_language: str = "ru",
    output_video_path: Union[str, Path] = None
) -> Dict[str, Any]:
    start_time = time.time()
    input_video_path = Path(input_video_path)

    if output_video_path is None:
        output_dir = Path("storage/dubbed_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = output_dir / f"dubbed_{target_language}_{input_video_path.name}"
    else:
        output_video_path = Path(output_video_path)

    temp_dir = Path("storage/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    extracted_audio_path = temp_dir / f"extracted_{input_video_path.stem}.wav"

    logger.info("=" * 60)
    logger.info(f"СТАРТ НЕЙРО-ДУБЛЯЖА (Voice Cloning): {input_video_path.name}")
    logger.info("=" * 60)

    logger.info(">> [1/6] Извлечение аудиодорожки...")
    extract_audio_from_video(input_video_path, extracted_audio_path)

    logger.info(">> [2/6] PyTorch VAD фильтрация...")
    vad_result = analyze_audio_speech(extracted_audio_path)

    logger.info(f">> [3/6] Распознавание речи ({source_language.upper()} -> {target_language.upper()})...")
    trans_data = transcribe_audio(extracted_audio_path, model_size="base", source_language=source_language)
    segments = trans_data.get("segments", [])
    original_text = trans_data.get("text", "")
    detected_lang = trans_data.get("language", "en")

    if not segments and original_text.strip():
        segments = [{"id": 0, "start": 0.0, "end": 4.0, "duration": 4.0, "text": original_text.strip()}]

    if not original_text.strip() and not segments:
        raise ValueError("В видео не обнаружено четких речевых фраз для перевода.")

    logger.info(">> [4/6] Анализ тембра и пола оригинального спикера...")
    speaker_profile = analyze_speaker_voice(extracted_audio_path, speech_segments=segments)
    speaker_gender = speaker_profile["gender"]
    logger.info(f"   Профиль спикера: {speaker_profile['voice_description']} ({speaker_profile['pitch_hz']} Hz)")

    logger.info(f">> [5/6] Перевод реплик на '{target_language}'...")
    translated_segments = translate_segments(
        segments=segments,
        source_lang=detected_lang,
        target_lang=target_language
    )

    logger.info(f">> [6/6] Кинематографичный нейро-синтез под пол ({speaker_gender})...")
    full_translated_text = []
    for idx, seg in enumerate(translated_segments):
        text_to_speak = seg.get("translated_text", "").strip()
        if text_to_speak:
            seg_audio_file = temp_dir / f"seg_actor_{idx}_{input_video_path.stem}.mp3"
            synthesize_speech(
                text=text_to_speak, 
                output_audio_path=seg_audio_file, 
                language=target_language,
                gender=speaker_gender
            )
            seg["audio_file"] = str(seg_audio_file)
            full_translated_text.append(text_to_speak)
        else:
            seg["audio_file"] = None

    build_dubbed_video_with_segments(
        video_path=input_video_path,
        speech_segments=translated_segments,
        output_video_path=output_video_path,
        background_volume=1.0,
        duck_volume=0.0
    )

    elapsed_time = round(time.time() - start_time, 2)
    logger.info(f"ДУБЛЯЖ ЗАВЕРШЕН ЗА {elapsed_time} сек!")

    return {
        "status": "success",
        "elapsed_time_sec": elapsed_time,
        "speaker_profile": speaker_profile,
        "vad_metrics": {"has_speech": vad_result["has_speech"], "speech_ratio": vad_result["speech_ratio"]},
        "original_language": detected_lang,
        "target_language": target_language,
        "original_text": original_text,
        "translated_text": " ".join(full_translated_text),
        "output_video_path": str(output_video_path)
    }