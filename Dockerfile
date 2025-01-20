# Use the official Python 3.9 image as the base image
FROM python:3.9

# Set environment variables
ENV PROJECT_ROOT /opt/app/
ENV ACCEPT_EULA=Y

# Set the working directory
WORKDIR $PROJECT_ROOT

# Copy only the requirements file to install dependencies
COPY . /opt/app/

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

RUN pip install --upgrade pip 
RUN apt-get update -y && \
    apt-get install -y vim curl && \
    rm -rf var/lib/apt/lists/* 

# Expose the desired port (e.g., 8000 for a FastAPI app)
EXPOSE 7860

# Default command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]