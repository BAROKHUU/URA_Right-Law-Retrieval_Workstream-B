FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LEGAL_RETRIEVAL_PROJECT_ROOT=/app

WORKDIR /app

# git is useful for run_metadata.git_revision and for some HF model repositories.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /app/requirements.lock
RUN python -m pip install --no-cache-dir -r /app/requirements.lock

COPY . /app
RUN python -m pip install --no-cache-dir --no-deps -e /app

CMD ["python", "scripts/run_experiment.py", "--help"]
