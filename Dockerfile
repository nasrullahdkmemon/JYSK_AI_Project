# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container at /app
COPY . /app

# Step 4: Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose port 8080 for cloud deployment (such as GCP)
EXPOSE 8080

# Step 6: Run the application when the container starts
CMD ["python", "app.py"]
