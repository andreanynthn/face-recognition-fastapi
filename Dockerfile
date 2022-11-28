FROM python:3.8-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN mkdir app
WORKDIR /app

COPY . .

RUN pip install -r requirements.txt