docker_onnx_build:
	docker build -t language_detection_onnx . -f Dockerfile_onnx

docker_onnx_run:
	docker run -p 8000:8000 language_detection_onnx

docker_cuda_build:
	docker build -t language_detection_cuda . -f Dockerfile_cuda

docker_cuda_run:
	docker run --gpus 0 -p 8000:8000 language_detection_cuda
