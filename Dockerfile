# ============================================================
#  Stage 1: устанавливаем зависимости (кэшируется Docker'ом)
# ============================================================
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime AS builder

WORKDIR /app

# opencv-headless не требует системных библиотек для дисплея
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Сначала фиксируем numpy<2 — до установки остальных пакетов
RUN pip install --no-cache-dir "numpy<2.0"

# Остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
#  Stage 2: финальный образ
# ============================================================
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime AS runtime

WORKDIR /app

# Системные библиотеки
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python пакеты из builder
COPY --from=builder /opt/conda/lib/python3.10/site-packages \
                    /opt/conda/lib/python3.10/site-packages
COPY --from=builder /opt/conda/bin /opt/conda/bin

# Код проекта
COPY src/        ./src/
COPY app.py      .
COPY config.yaml .

RUN mkdir -p checkpoints/rgb

EXPOSE 7860
ENV WEIGHTS_PATH=checkpoints/rgb/best_model.pth

CMD ["python", "app.py"]
