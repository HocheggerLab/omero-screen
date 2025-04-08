FROM python:3.12-slim

WORKDIR /app

COPY . .

# Install uv and use it to install dependencies
RUN pip install uv && \
    uv venv && \
    uv sync --all-extras --dev

# Set environment variables
ENV ENV=development \
    LOG_LEVEL=DEBUG \
    HOST=omeroserver \
    USERNAME=root \
    PASSWORD=omero \
    LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s" \
    ENABLE_CONSOLE_LOGGING=True \
    ENABLE_FILE_LOGGING=False \
    LOG_FILE_PATH=logs/app.log \
    LOG_MAX_BYTES=1048576 \
    LOG_BACKUP_COUNT=5

# Ensure no .env files exist that could override our settings
RUN rm -f .env .env.*

CMD ["uv", "run", "pytest", "tests/", "-v", "--ignore=tests/e2e"]
