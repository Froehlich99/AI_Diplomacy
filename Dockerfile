FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv "b2[full]"

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY . .

RUN chmod +x entrypoint.sh

ENTRYPOINT ["bash", "entrypoint.sh"]
