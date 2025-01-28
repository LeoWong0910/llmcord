FROM --platform=$TARGETPLATFORM python:3.12-slim as builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM --platform=$TARGETPLATFORM python:3.12-slim

WORKDIR /usr/src/app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copy application files
COPY llmcord.py .

# Create data directory
RUN mkdir -p /usr/src/app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Macau

# Set volume
VOLUME ["/usr/src/app/data"]

CMD ["python", "llmcord.py"]
