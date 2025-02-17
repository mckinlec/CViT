# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the entry point to run the Streamlit app
ENTRYPOINT ["streamlit", "run"]

# Specify the default Streamlit app to run
CMD ["streamlit_app.py"]
