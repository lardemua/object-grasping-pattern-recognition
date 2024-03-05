docker_build:
	docker build -t pedromiglou/model_training ./model_training

docker_run:
	docker run -v ${HOME}/container_results:/model_training/results --gpus '"device=0"' -d pedromiglou/model_training

docker_run_bash:
	docker run -v ${HOME}/container_results:/model_training/results --gpus '"device=0"' -it pedromiglou/model_training bash
