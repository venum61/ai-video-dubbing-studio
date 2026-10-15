from pathlib import Path
import shutil
import subprocess
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
import logging

from app.services.pipeline import run_dubbing_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Video Dubbing Studio",
    description="Синхронизированный дубляж видео на PyTorch VAD, Whisper и Neural TTS с живым прогресс-баром.",
    version="2.2.0"
)

# Создание рабочих папок
Path("storage/raw_videos").mkdir(parents=True, exist_ok=True)
Path("storage/dubbed_videos").mkdir(parents=True, exist_ok=True)
Path("storage/temp").mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Хранилище прогресса текущих задач
TASKS_PROGRESS = {}


def normalize_video_file(input_path: Path) -> Path:
    """Конвертирует потоковый WebM из браузера в стабильный MP4 с расчетом длительности."""
    if input_path.suffix.lower() == ".webm":
        fixed_mp4_path = input_path.with_suffix(".mp4")
        logger.info(f"Нормализация WebM в MP4: {input_path.name} -> {fixed_mp4_path.name}")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            str(fixed_mp4_path)
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0 and fixed_mp4_path.exists():
            return fixed_mp4_path
        else:
            logger.warning(f"Ошибка FFmpeg при нормализации: {result.stderr.decode('utf-8', errors='ignore')}")
            
    return input_path


@app.get("/", tags=["UI"])
def render_ui(request: Request):
    """Главная страница веб-студии."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health", tags=["System"])
def health_check():
    """Проверка доступности сервиса."""
    return {"status": "OK", "service": "AI Video Dubbing Studio", "version": "2.2.0"}


@app.get("/progress/{task_id}", tags=["Dubbing"])
def get_task_progress(task_id: str):
    """Возвращает текущий процент и текстовый статус выполнения задачи."""
    return TASKS_PROGRESS.get(task_id, {"percent": 0, "message": "Подготовка к обработке..."})


@app.post("/dub", tags=["Dubbing"])
def dub_video(
    file: UploadFile = File(..., description="Видеофайл (.mp4, .mov, .avi, .webm)"),
    source_language: str = Form("auto", description="Язык оригинала ('auto', 'en', 'ru', 'es')"),
    target_language: str = Form("ru", description="Целевой язык ('ru', 'en', 'es', 'de', 'fr')"),
    task_id: str = Form(None)
):
    """Синхронный запуск конвейера дубляжа с обновлением прогресса."""
    current_task_id = task_id or str(uuid.uuid4())
    TASKS_PROGRESS[current_task_id] = {"percent": 5, "message": "Загрузка видео на сервер..."}

    def progress_updater(percent: int, message: str):
        TASKS_PROGRESS[current_task_id] = {"percent": percent, "message": message}

    allowed_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Неподдерживаемый формат: {file_extension}. Разрешены: {allowed_extensions}"
        )

    raw_video_path = Path("storage/raw_videos") / file.filename
    try:
        with open(raw_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Ошибка сохранения файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения файла: {e}")

    processed_video_path = normalize_video_file(raw_video_path)

    try:
        pipeline_result = run_dubbing_pipeline(
            input_video_path=processed_video_path,
            source_language=source_language,
            target_language=target_language,
            progress_callback=progress_updater
        )
        
        output_filename = Path(pipeline_result["output_video_path"]).name
        pipeline_result["download_url"] = f"/download/{output_filename}"
        
        return pipeline_result

    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        logger.error(f"Ошибка пайплайна: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка обработки видео: {e}")


@app.get("/download/{filename}", tags=["Dubbing"])
def download_dubbed_video(filename: str):
    """Скачивание готового медиафайла."""
    video_path = Path("storage/dubbed_videos") / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Запрошенный видеофайл не найден.")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=filename
    )