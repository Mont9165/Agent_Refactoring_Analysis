FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    openjdk-17-jdk \
    curl \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Runtime configuration
ENV PATH="$JAVA_HOME/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/data/huggingface/cache \
    MPLCONFIGDIR=/tmp/matplotlib

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/huggingface/cache \
    && mkdir -p data/filtered/java_repositories \
    && mkdir -p data/analysis/refactoring_instances \
    && mkdir -p outputs/research_questions \
    && mkdir -p tools \
    && mkdir -p /tmp/matplotlib

# Set up RefactoringMiner (will be built on first run)
RUN echo "RefactoringMiner will be automatically set up on first run"

# Create entrypoint script
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port for any potential web services
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
