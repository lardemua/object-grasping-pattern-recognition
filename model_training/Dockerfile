FROM tensorflow/tensorflow:latest-gpu

WORKDIR /model_training

COPY ./preprocessed_dataset /model_training/preprocessed_dataset

COPY ./docker_requirements.txt /model_training/requirements.txt

RUN pip3 install -r /model_training/requirements.txt

COPY *.py /model_training/

RUN mkdir /model_training/results

CMD ["python3", "/model_training/transformer_main.py"]
