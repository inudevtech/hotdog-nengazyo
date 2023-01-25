FROM python:3.10
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt
COPY . .

RUN apt-get update
RUN apt-get install tesseract-ocr libtesseract-dev
RUN apt clean
