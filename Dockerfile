FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    openjdk-17-jdk \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/huggingface/cache \
    && mkdir -p data/filtered/java_repositories \
    && mkdir -p data/analysis/refactoring_instances \
    && mkdir -p tools

# Set up RefactoringMiner (will be built on first run)
RUN echo "RefactoringMiner will be automatically set up on first run"

# Create entrypoint script
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV HF_HOME=/app/data/huggingface/cache

# Expose port for any potential web services
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]