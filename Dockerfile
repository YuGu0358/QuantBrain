FROM node:22-bookworm-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip python3-venv build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY package.json ./
COPY alpha_miner ./alpha_miner
COPY service ./service
COPY docs ./docs

RUN python3 -m pip install --break-system-packages --no-cache-dir -r alpha_miner/requirements.txt

ENV PORT=3000
ENV RUNS_DIR=/app/cloud-runs
ENV ALPHA_MINER_ENGINE=python-v2

RUN mkdir -p /app/cloud-runs

EXPOSE 3000

CMD ["npm", "start"]
