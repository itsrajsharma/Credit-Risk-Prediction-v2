FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and artifacts
COPY . .

# Expose the default port (can be overridden by environment variable)
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
