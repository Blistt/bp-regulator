# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install bash
RUN apt-get update && apt-get install -y bash

# Create a new conda environment with Python 3.11.7
RUN conda create -n myenv python=3.11.7

# Activate the conda environment and install packages
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

RUN /bin/bash -c "source activate myenv && \
    conda install -c conda-forge \
    xgboost\
    scikit-learn \
    numpy \
    pandas

# Make port 80 available to the world outside this container
EXPOSE 80