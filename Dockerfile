FROM huggingface/transformers-pytorch-gpu:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install additional system dependencies if needed
RUN apt-get update
RUN apt-get install -y wget

# Set working directory
WORKDIR /workspace

# Copy pyproject.toml first for better caching
COPY pyproject.toml /workspace/
RUN mkdir -p /workspace/signwriting_transcription && touch /workspace/signwriting_transcription/__init__.py
RUN pip install -e .[dev]

# Copy project files
COPY . /workspace/

# Entry point to process data and train the model
CMD ["make prepare && make train"]
