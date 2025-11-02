FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    transformers==4.35.2 \
    torch==2.1.0 \
    sentence-transformers==2.2.2

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8002

# Run the application
CMD ["python", "app.py"]