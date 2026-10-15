from pathlib import Path
from typing import Union, List, Dict, Any
import asyncio
import edge_tts
from gtts import gTTS
import os
import logging

logger = logging.getLogger(__name__)

# Таблица профессиональных нейро-актеров дубляжа по языкам и полу
VOICE_ACTORS = {
    "en": {"male": "en-US-ChristopherNeural", "female": "en-US-JennyNeural"},
    "es": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
    "de": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
    "fr": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
    "ru": {"male": "ru-RU-DmitryNeural", "female": "ru-RU-SvetlanaNeural"}
}


async def _generate_edge_tts(text: str, voice_name: str, output_path: Path):
    """Генерация одиночного аудио через нейросеть Microsoft Edge-TTS."""
    communicate = edge_tts.Communicate(text=text, voice=voice_name)
    await communicate.save(str(output_path))


def synthesize_speech(
    text: str, 
    output_audio_path: Union[str, Path], 
    language: str = "en",
    gender: str = "male"
) -> Path:
    """
    Отказоустойчивый синтез одиночной фразы: Edge-TTS -> автоматический Fallback на gTTS.
    """
    if not text or not text.strip():
        raise ValueError("Текст для синтеза речи пуст.")

    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    lang_code = language.lower().strip()
    gender_code = gender.lower().strip() if gender in ["male", "female"] else "male"

    lang_voices = VOICE_ACTORS.get(lang_code, VOICE_ACTORS["en"])
    chosen_voice = lang_voices.get(gender_code, lang_voices["male"])

    # 1. Попытка через Microsoft Neural Voice
    try:
        logger.info(f"Генерация нейро-голоса [{chosen_voice}]: '{text}'")
        asyncio.run(_generate_edge_tts(text.strip(), chosen_voice, output_audio_path))
        return output_audio_path
    except Exception as edge_err:
        logger.warning(f"Edge-TTS временно недоступен ({edge_err}). Переключаемся на gTTS...")

    # 2. Резервный шлюз (Fallback): gTTS
    try:
        tts = gTTS(text=text.strip(), lang=lang_code if lang_code in ["ru", "en", "es", "de", "fr"] else "en")
        tts.save(str(output_audio_path))
        logger.info(f"Резервный синтез успешно завершен: {output_audio_path.name}")
        return output_audio_path
    except Exception as fallback_err:
        logger.error(f"Сбой всех систем синтеза: {fallback_err}")
        raise RuntimeError(f"Не удалось озвучить текст: {fallback_err}")


async def _synthesize_all_segments_async(tasks_data: List[Dict[str, Any]]):
    """
    Параллельно запускает генерацию сразу всех фраз через asyncio.gather.
    """
    async def _safe_single_gen(item: Dict[str, Any]):
        out_path = item["out_path"]
        text = item["text"]
        voice = item["voice"]
        lang = item["lang"]

        # Попытка Edge-TTS
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            await communicate.save(str(out_path))
            return
        except Exception:
            pass

        # Fallback на gTTS
        try:
            tts = gTTS(text=text, lang=lang if lang in ["ru", "en", "es", "de", "fr"] else "en")
            tts.save(str(out_path))
        except Exception as e:
            logger.error(f"Не удалось озвучить фразу '{text}': {e}")

    # Запускаем все задачи генерации одновременно в пуле корутин
    coroutines = [_safe_single_gen(t) for t in tasks_data]
    await asyncio.gather(*coroutines)


def synthesize_segments_batch(
    segments: List[Dict[str, Any]], 
    temp_dir: Path, 
    language: str = "en", 
    gender: str = "male"
) -> List[Dict[str, Any]]:
    """
    Пакетный параллельный синтез: озвучивает ВСЕ сегменты одновременно.
    """
    lang_code = language.lower().strip()
    gender_code = gender.lower().strip() if gender in ["male", "female"] else "male"
    lang_voices = VOICE_ACTORS.get(lang_code, VOICE_ACTORS["en"])
    chosen_voice = lang_voices.get(gender_code, lang_voices["male"])

    tasks_data = []
    for idx, seg in enumerate(segments):
        text = seg.get("translated_text", "").strip()
        if text:
            out_file = temp_dir / f"seg_actor_{idx}.mp3"
            tasks_data.append({
                "text": text,
                "voice": chosen_voice,
                "out_path": out_file,
                "lang": lang_code,
                "seg_ref": seg
            })

    if tasks_data:
        logger.info(f"Параллельный синтез {len(tasks_data)} реплик через asyncio.gather...")
        asyncio.run(_synthesize_all_segments_async(tasks_data))
        for t in tasks_data:
            t["seg_ref"]["audio_file"] = str(t["out_path"])

    return segments