# Automatic Audio Anonymization

This is my source code for my master at École de Technologie Supérieure in partnership with Desjardins.

The goal was to create a proof of concept to determine if it is possible to anonymize french audio recordings.

The code is separated as followed :
- fa : Code used to evaluate two Forced Alignment (FA) algorithms
- ner : Code used to train and evaluate three Named Entity Recognition (NER) models
- pipeline : Code used to create the docker image to anonymize audio recordings

## Pipeline

The pipeline is usable via a Docker image. Please refer to the official [documentation](https://docs.docker.com/get-docker/) for more details on how to install Docker.

### Installation steps

Firstly, build the docker image.

```
cd pipeline
docker build --tag pipeline .
```

Secondly, download the trained NER models from [TODO](google.ca).

Thirdly, allow the docker image to use your GPU.

```
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |\
    sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
sudo systemctl stop docker
sudo systemctl start docker
```

### User manual

Now, you can use the pipeline.

```
docker run -it -v [PATH_TO_DATA_TO_ANONYMIZE]:/input \
-v [PATH_TO_TMP_FA_ALGO_OUTPUT]:/align \
-v [PATH_TO_NER_MODELS_DIR]:/ner_models \
-v [PATH_TO_PIPELINE_OUTPUT]:/redact \
--gpus device=0 pipeline
```

### Input data format

The input directory contains the audio with its corresponding transcription.

The audio file format is wav.
The transcription file format is TextGrid. Both files must have the same name. For example, if the audio file is named *example.wav*, the transcription must be named *example.TextGrid*.

Here is an example of a Textgrid file :

```
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.000000
xmax = 4.803000
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "spkr_1_1-trans"
        xmin = 0.000000
        xmax = 4.803000
        intervals: size = 3
        intervals [1]:
            xmin = 0.000000
            xmax = 0.500000
            text = ""
        intervals [2]:
            xmin = 0.500000
            xmax = 4.303000
            text = "This is an example of someone talking for approximately four seconds"
        intervals [3]:
            xmin = 4.303000
            xmax = 4.803000
            text = ""
```

Note that to work directly with our pipeline, the interval name containing the transcription must be named **spkr_1_1-trans**.