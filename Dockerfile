# Use a slim Python 3.11 base for a proper Linux sandbox
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed (e.g., for tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire harness
COPY . .

# Ensure tools package is importable
ENV PYTHONPATH=/app

# Default command runs the agent REPL
CMD ["python", "main.py"]
