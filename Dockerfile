# Step 1: Start with a specific, stable Python version
FROM python:3.10-slim

# Step 2: Install system-level dependencies, including Git
RUN apt-get update && apt-get install -y git build-essential

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Copy the requirements file first to leverage Docker's caching
COPY requirements.txt .

# Step 5: Install the Python packages
# This includes installing sam2 directly from GitHub
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your application code
COPY . .

# Step 7: Expose the port that Gradio will run on
EXPOSE 7860

# Step 8: Define the command to run your application
CMD ["gradio", "app.py", "--host", "0.0.0.0", "--port", "7860"]
