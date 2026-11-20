FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY --chown=user app ./app
COPY --chown=user data/wasalt_listings.json data/darglobal_projects.json data/darglobal_company.json ./data/

RUN chown user /home/user/app  # WORKDIR is root-owned; the model cache and embeddings are written here
USER user
ENV PATH="/home/user/app/.venv/bin:$PATH" PORT=7860
# download the embedding model and embed all records at build time, so startup is fast and offline
RUN python -c "import app.search"
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
# PORT: 7860 on HF Spaces; Render/Railway inject their own
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
