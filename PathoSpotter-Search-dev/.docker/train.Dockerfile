FROM tensorflow/tensorflow:2.2.0-gpu

WORKDIR /usr/src/app
COPY .docker/train-requirements.txt ./

RUN pip install --no-cache -r train-requirements.txt