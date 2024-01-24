FROM tensorflow/tensorflow:latest-gpu

WORKDIR /model_training

COPY ./dataset_after_preprocessing /model_training/dataset_after_preprocessings

COPY ./docker_requirements.txt /model_training/requirements.txt

RUN pip3 install -r /model_training/requirements.txt

COPY *.py /model_training/

RUN mkdir /model_training/results

CMD ["python3", "/model_training/transformer_main.py"]
