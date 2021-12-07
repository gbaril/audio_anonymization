source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
rm -r ~/.theano
conda remove -y -n lstm --all
conda update -y --all
conda create -y -n lstm -c conda-forge python=2.7
conda activate lstm
python -m pip install --upgrade pip

sudo apt-get install -y libatlas-base-dev liblapack-dev
conda install -y theano numpy
conda install -y -c conda-forge theano
