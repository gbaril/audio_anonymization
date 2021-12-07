# How to install requirements to use GPU in docker

curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |\
    sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime

sudo systemctl stop docker
sudo systemctl start docker

# How to use the Docker image

docker run -it -v <path_to_input>:/input -v <path_to_tmp_aligned_transcript>:/align -v <path_to_ner_models>:/ner_models -v <path_to_output>:/redact -v <path_to_gold_annotations>:/gold --gpus device=<gpu_num> <docker_name>

docker run -it -v ~/Documents/ner/data/test:/input -v ~/Documents/ner/data/output:/align -v ~/Documents/ner/ner/huggingface/ner_models:/ner_models -v ~/Documents/ner/data/redact:/redact -v ~/Documents/ner/data/gold:/gold --gpus device=0 pipeline