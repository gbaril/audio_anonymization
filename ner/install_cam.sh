source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda remove -y -n ner --all
conda update -y --all
conda create -y -n ner -c conda-forge python=3.8
conda activate ner
python -m pip install --upgrade pip

# Spacy
pip install -U pip setuptools wheel matplotlib scikit-learn
pip install -U spacy[cuda112,transformers,lookups]
python -m spacy download fr_dep_news_trf
python -m spacy download fr_core_news_lg