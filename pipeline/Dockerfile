# syntax=docker/dockerfile:1
FROM continuumio/miniconda3
WORKDIR /pipeline

# create environment
RUN conda update --all
RUN conda create -y -n pipeline -c conda-forge python=3.8 openfst pynini ngram baumwelch

# install depedencies
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y libsndfile1 build-essential git vim libopenblas-base ffmpeg

# copy source
COPY src ./
RUN mv docker_config.py config.py
# need to test last command

# make RUN commands use the new environment (https://pythonspeed.com/articles/activate-conda-dockerfile/)
RUN echo "conda activate pipeline" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# install ner
RUN conda install pytorch cudatoolkit=10.2 -c pytorch 
RUN pip install transformers
RUN pip install -U pip setuptools wheel
RUN pip install -U spacy[cuda102]
RUN python -m spacy download fr_dep_news_trf

# install mfa
RUN pip install -U montreal-forced-aligner==2.0.0a22
RUN pip install -U praatio==4.4.0
RUN mfa download acoustic french_prosodylab
RUN mfa download acoustic french_qc 
RUN mfa download dictionary french_prosodylab
RUN mfa thirdparty download

# install fa script dependencies
RUN pip install -U python-Levenshtein==0.12.2 pympi-ling==1.69 Unidecode==1.2.0 fuzzywuzzy==0.18.0 pydub==0.25.1

CMD python main.py