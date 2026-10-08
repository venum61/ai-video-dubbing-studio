from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Тест 1: Проверка доступности сервиса."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "version" in data


def test_invalid_file_extension():
    """Тест 2: Проверка защиты от загрузки не-видео файлов (например, .txt)."""
    fake_file = ("document.txt", b"dummy content", "text/plain")
    response = client.post(
        "/dub",
        files={"file": fake_file},
        data={"source_language": "auto", "target_language": "ru"}
    )
    assert response.status_code == 400
    assert "Неподдерживаемый формат" in response.json()["detail"]


def test_download_not_found():
    """Тест 3: Проверка отдачи 404 для несуществующих файлов."""
    response = client.get("/download/completely_random_file_12345.mp4")
    assert response.status_code == 404
    assert "не найден" in response.json()["detail"]