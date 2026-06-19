# Cendro inference server. Ollama runs as a separate service (see docker-compose.yml);
# this image is just the thin FastAPI layer that talks to it.
FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching. constraints.txt pins the runtime stack.
COPY pyproject.toml constraints.txt README.md ./
COPY cendro ./cendro
COPY api ./api
COPY model ./model
RUN pip install --no-cache-dir -e . -c constraints.txt

# Reach the ollama service by default; override for a different host.
ENV OLLAMA_HOST=http://ollama:11434 \
    CENDRO_MODEL=qwen2.5-coder:7b

EXPOSE 8000
CMD ["cendro", "serve", "--host", "0.0.0.0", "--port", "8000"]
