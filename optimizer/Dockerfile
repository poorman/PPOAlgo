FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    optuna \
    requests \
    numpy \
    websockets \
    psycopg2-binary

# Copy application files
COPY server.py /app/
COPY static /app/static/

# Create results directory
RUN mkdir -p /app/results

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
