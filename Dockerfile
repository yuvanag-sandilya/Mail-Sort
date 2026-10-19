FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --retries 10 --default-timeout=100 -r requirements.txt

# Copy application code
COPY . .

# Train the model during build
RUN python model/train_model.py

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
