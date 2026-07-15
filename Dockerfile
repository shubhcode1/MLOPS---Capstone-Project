FROM python:3.10-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY flask_app/ ./
COPY models/vectorizer.pkl ./models/vectorizer.pkl

# Download NLTK data
RUN uv run python -m nltk.downloader stopwords wordnet

EXPOSE 5000

# For local
# CMD ["uv", "run", "python", "app.py"]

# For Production
CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]