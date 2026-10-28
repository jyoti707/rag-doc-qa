# Docker Basics

*Source: Open technical documentation and standard references (public knowledge, paraphrased)*

This document gives a short overview of Docker Basics.

## Key points

A Docker image is a read-only template, while a container is a running instance of an image. The command docker build -t myapp . builds an image from the Dockerfile in the current folder and tags it myapp. The command docker run starts a new container from an image.

The command docker ps lists running containers. The option -p 8501:8501 maps port 8501 of the host to port 8501 of the container. A Dockerfile is a text file with instructions that describe how to build an image.

Volumes store data outside the container so that it persists after the container is removed. Docker Compose uses a YAML file to define and run applications made of several containers.

> Note: This is a simplified educational summary written in our own words from public information. It is not legal, financial or medical advice. Verify details with the official source before relying on them.
