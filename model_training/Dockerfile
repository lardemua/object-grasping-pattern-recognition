FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /model_training

COPY ./preprocessed_dataset /model_training/preprocessed_dataset

COPY ./docker_requirements.txt /model_training/requirements.txt

RUN python3 -m pip install --upgrade pip

RUN pip3 install -r /model_training/requirements.txt

RUN rm /model_training/requirements.txt

COPY *.py /model_training/

RUN mkdir /model_training/results

CMD ["python3", "/model_training/cnn_main.py"]
