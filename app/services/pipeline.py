from pathlib import Path
from typing import Union, Dict, Any
import logging
import time

from app.services.audio_extractor import extract_audio_from_video
from app.services.vad_service import analyze_audio_speech
from app.services.voice_analyzer import analyze_speaker_voice
from app.services.transcriber import transcribe_audio
from app.services.translator import translate_segments
from app.services.synthesizer import synthesize_segments_batch
from app.services.video_builder import build_dubbed_video_with_segments

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_dubbing_pipeline(
    input_video_path: Union[str, Path],
    source_language: str = "auto",
    target_language: str = "en",
    output_video_path: Union[str, Path] = None,
    progress_callback = None
) -> Dict[str, Any]:
    """
    Оптимизированный конвейер дубляжа с живым трекингом прогресса:
    Stream Copy + Батчевый VAD + Параллельный синтез речи.
    """
    start_time = time.time()
    input_video_path = Path(input_video_path)

    def update_progress(percent: int, msg: str):
        if progress_callback:
            try:
                progress_callback(percent, msg)
            except Exception:
                pass

    # 1. Формирование путей назначения
    if output_video_path is None:
        output_dir = Path("storage/dubbed_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = output_dir / f"dubbed_{target_language}_{input_video_path.name}"
    else:
        output_video_path = Path(output_video_path)

    # 2. Временная папка и пути файлов (ОБЪЯВЛЕНИЕ ПЕРЕМЕННЫХ)
    temp_dir = Path("storage/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    extracted_audio_path = temp_dir / f"extracted_{input_video_path.stem}.wav"

    logger.info("=" * 60)
    logger.info(f"СТАРТ НЕЙРО-ДУБЛЯЖА: {input_video_path.name}")
    logger.info("=" * 60)

    # ШАГ 1: Извлечение аудиодорожки
    update_progress(15, "Извлечение аудиодорожки из видео...")
    logger.info(">> [1/6] Извлечение аудиодорожки...")
    extract_audio_from_video(input_video_path, extracted_audio_path)

    # ШАГ 2: Батчевая VAD-фильтрация (PyTorch 1D-CNN)
    update_progress(30, "Сканирование речи через PyTorch VAD...")
    logger.info(">> [2/6] Батчевая PyTorch VAD фильтрация...")
    vad_result = analyze_audio_speech(extracted_audio_path)

    # ШАГ 3: Транскрибация Whisper с пословными таймкодами
    update_progress(50, f"Распознавание речи Whisper ({source_language.upper()})...")
    logger.info(f">> [3/6] Распознавание речи ({source_language.upper()} -> {target_language.upper()})...")
    trans_data = transcribe_audio(
        extracted_audio_path, 
        model_size="base", 
        source_language=source_language
    )
    segments = trans_data.get("segments", [])
    original_text = trans_data.get("text", "")
    detected_lang = trans_data.get("language", "ru")

    if not segments and original_text.strip():
        segments = [{"id": 0, "start": 0.0, "end": 4.0, "duration": 4.0, "text": original_text.strip()}]

    if not original_text.strip() and not segments:
        raise ValueError("В видео не обнаружено четких речевых фраз для перевода.")

    # ШАГ 4: Анализ тембра и пола спикера (F0 Pitch)
    update_progress(70, "Анализ частоты F0 и подбор нейро-актера...")
    logger.info(">> [4/6] Анализ тембра и пола спикера...")
    speaker_profile = analyze_speaker_voice(extracted_audio_path, speech_segments=segments)
    speaker_gender = speaker_profile["gender"]
    logger.info(f"   Профиль спикера: {speaker_profile['voice_description']} ({speaker_profile['pitch_hz']} Hz)")

    # ШАГ 5: Перевод реплик
    update_progress(80, f"Перевод реплик на язык '{target_language.upper()}'...")
    logger.info(f">> [5/6] Перевод реплик на '{target_language}'...")
    translated_segments = translate_segments(
        segments=segments,
        source_lang=detected_lang,
        target_lang=target_language
    )

    # ШАГ 6: Параллельный нейро-синтез через asyncio.gather
    update_progress(90, f"Параллельный нейро-синтез голоса ({speaker_gender})...")
    logger.info(f">> [6/6] Параллельный нейро-синтез реплик ({speaker_gender})...")
    translated_segments = synthesize_segments_batch(
        segments=translated_segments,
        temp_dir=temp_dir,
        language=target_language,
        gender=speaker_gender
    )
    full_translated_text = [seg["translated_text"] for seg in translated_segments if seg.get("translated_text")]

    # ШАГ 7: Мгновенный монтаж FFmpeg Stream Copy
    update_progress(96, "Сведение звука и монтаж FFmpeg Stream Copy...")
    build_dubbed_video_with_segments(
        video_path=input_video_path,
        speech_segments=translated_segments,
        output_video_path=output_video_path,
        background_volume=1.0,
        duck_volume=0.0
    )

    # Очистка временных файлов сессии
    try:
        if extracted_audio_path.exists():
            extracted_audio_path.unlink()
        for seg in translated_segments:
            seg_file = seg.get("audio_file")
            if seg_file and Path(seg_file).exists():
                Path(seg_file).unlink()
        logger.info("Временные файлы сессии успешно очищены.")
    except Exception as cleanup_err:
        logger.warning(f"Ошибка при очистке временных файлов: {cleanup_err}")

    update_progress(100, "Дубляж успешно завершен!")
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