# Pull FastAPI Docker image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Install Poetry and configure it to run inside Docker
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python \
    && cd /usr/local/bin \
    && ln -s /opt/poetry/bin/poetry \
    && poetry config virtualenvs.create false

# Copy app dependencies (use poetry.lock* to avoid crashing in case the file doesn't exist yet)
COPY ./app/pyproject.toml ./app/poetry.lock* /app/

# Install app dependencies
RUN poetry install --no-root --no-dev

# Copy app code after installing the dependencies to take advantage of cache
COPY ./app /app

# Copy models
COPY ./model /model

# This does not actually publish the port, but it is for documentation
EXPOSE 8000
