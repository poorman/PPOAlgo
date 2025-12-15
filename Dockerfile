FROM python:3.10-slim

# Keep Python from writing .pyc files and enable unbuffered logs.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the project into the image.
COPY . /app

# Install dependencies if a requirements file is present and not empty.
RUN if [ -f requirements.txt ] && [ -s requirements.txt ]; then \
      pip install --no-cache-dir --upgrade pip && \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      pip install --no-cache-dir --upgrade pip; \
    fi

# Default command runs the training script.
CMD ["python", "scripts/train_lstm.py"]

