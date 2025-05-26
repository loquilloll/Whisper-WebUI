FROM debian:bookworm-slim AS builder

RUN apt-get update && \
    apt-get install -y curl git python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    mkdir -p /Whisper-WebUI

WORKDIR /Whisper-WebUI

# Copy the main requirements.txt from the project root
COPY requirements.txt ./requirements.txt
# Copy the backend-specific requirements file
COPY backend/requirements-backend.txt ./backend/requirements-backend.txt

# Install Python dependencies
# This layer will be cached if neither requirements.txt nor backend/requirements-backend.txt change.
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install -U -r backend/requirements-backend.txt && \ # This will correctly find ../requirements.txt
    deactivate

# Now copy the application code
COPY backend/ ./backend/
COPY modules/ ./modules/ # Assuming 'modules' is at the root of your build context and needed by the backend


FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y curl ffmpeg python3 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /Whisper-WebUI

COPY . .
COPY --from=builder /Whisper-WebUI/venv /Whisper-WebUI/venv

VOLUME [ "/Whisper-WebUI/models" ]
VOLUME [ "/Whisper-WebUI/outputs" ]
VOLUME [ "/Whisper-WebUI/backend" ]

ENV PATH="/Whisper-WebUI/venv/bin:$PATH"
ENV LD_LIBRARY_PATH=/Whisper-WebUI/venv/lib64/python3.11/site-packages/nvidia/cublas/lib:/Whisper-WebUI/venv/lib64/python3.11/site-packages/nvidia/cudnn/lib

ENTRYPOINT ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
