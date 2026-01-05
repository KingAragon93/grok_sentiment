# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run the application with functions-framework
CMD exec functions-framework --target=grok_sentiment --port=$PORT
