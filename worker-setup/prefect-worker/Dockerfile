FROM python:3.10.15-slim

WORKDIR /service

# for using git-based deployment
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        tzdata \
        build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

CMD prefect worker start \
    --name ${PREFECT_WORKER_NAME} \
    --pool ${PREFECT_WORK_POOL} \
    --type process