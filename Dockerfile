# Use a lightweight Python image
FROM python:3.9-slim
# Set the working directory in the container
WORKDIR /app
# Copy the current directory contents to the container
COPY . .
# Install the required Python packages
RUN pip install -r requirements.txt
# Expose port 5000
EXPOSE 8000
# Command to run the Flask app
CMD ["python", "app.py"]