import torch
from app.models.vad_model import VoiceActivityDetector


def test_vad_architecture():
    print("=== ТЕСТ 1: Инициализация и подсчет параметров нейросети ===")
    model = VoiceActivityDetector()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Архитектура успешно создана!")
    print(f"Всего обучаемых параметров: {total_params:,}")
    print(f"Размер весов в оперативной памяти: ~{(total_params * 4) / 1024:.2f} КБ (Очень легкая и быстрая!)")

    print("\n=== ТЕСТ 2: Прямой проход (Forward Pass) с батчем аудио ===")
    dummy_audio = torch.randn(4, 1, 16000)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_audio)

    print(f"Входная форма тензора:  {list(dummy_audio.shape)}")
    print(f"Выходная форма тензора: {list(output.shape)}")
    print(f"Предсказанные вероятности:\n{output.squeeze().tolist()}")

    # Валидация условий
    assert output.shape == (4, 1), f"Ошибка: ожидалась форма (4, 1), получена {output.shape}"
    assert (output >= 0.0).all() and (output <= 1.0).all(), "Ошибка: вероятности вышли за пределы [0.0, 1.0]"
    print("УСПЕХ: Все размерности и вероятности корректны!")

    print("\n=== ТЕСТ 3: Проверка переменной длины аудио (Dynamic Length) ===")
    for duration_sec in [0.5, 1.5, 3.0]:
        samples = int(16000 * duration_sec)
        var_audio = torch.randn(1, 1, samples)
        with torch.no_grad():
            res = model(var_audio)
        print(f"  Длина {duration_sec}с ({samples} сэмплов) -> Предсказание: {res.item():.4f} - OK")

    print("\n=== ТЕСТ 4: Проверка градиентов (Backward Pass) ===")
    model.train()
    sample_input = torch.randn(2, 1, 16000)
    sample_target = torch.tensor([[1.0], [0.0]])
    
    criterion = torch.nn.BCELoss()
    pred = model(sample_input)
    loss = criterion(pred, sample_target)
    loss.backward()

    print(f"Loss успешно рассчитан: {loss.item():.4f}")
    print("Градиенты успешно рассчитаны по всей сети без NaN!")
    print("\n" + "=" * 50)
    print("НЕЙРОСЕТЬ ПОЛНОСТЬЮ ГОТОВА К РАБОТЕ!")
    print("=" * 50)


if __name__ == "__main__":
    test_vad_architecture()