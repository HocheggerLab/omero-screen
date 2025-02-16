FROM python:3.12-slim

WORKDIR /app

COPY . .

# Install uv and use it to install dependencies
RUN pip install uv && \
    uv venv && \
    uv sync --all-extras --dev

# Set environment variables
ENV LOG_LEVEL=DEBUG \
    LOG_FILE_PATH=/tmp/omero_screen.log \
    HOST=omeroserver \
    USERNAME=root \
    PASSWORD=omero \
    ENV=development

# Ensure no .env files exist that could override our settings
RUN rm -f .env .env.*

CMD ["uv", "run", "pytest", "tests/", "-v", "--ignore=tests/e2e"]
