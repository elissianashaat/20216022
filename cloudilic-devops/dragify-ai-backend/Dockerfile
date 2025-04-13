FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Set up a virtual environment and install dependencies in it
RUN python -m venv /app/venv
# Use the venv's pip to install requirements
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port your app will run on
EXPOSE 8000

# Use the Python executable from the virtual environment
CMD ["/app/venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]