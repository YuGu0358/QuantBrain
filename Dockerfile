FROM node:22-bookworm-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip python3-venv build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first so pip install layer is cached independently of source changes
COPY alpha_miner/requirements.txt ./alpha_miner/requirements.txt
RUN python3 -m pip install --break-system-packages -r alpha_miner/requirements.txt

# Copy source (invalidates only the layers below, not pip install)
COPY package.json ./
COPY alpha_miner ./alpha_miner
COPY service ./service
COPY docs ./docs

ENV PORT=3000
ENV RUNS_DIR=/app/cloud-runs
ENV ALPHA_MINER_ENGINE=python-v2

RUN mkdir -p /app/cloud-runs

EXPOSE 3000

CMD ["npm", "start"]
