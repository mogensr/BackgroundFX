# Use an OpenCV-ready base image
FROM jjanzic/docker-python3-opencv:opencv-4.5.3


WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Environment variables
ENV PORT=5000

# Command to run the application
CMD ["python", "app.py"]
