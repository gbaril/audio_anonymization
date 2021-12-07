source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda remove -y -n fa --all
rm -r ~/Documents/MFA
conda update -y --all
conda create -y -n fa -c conda-forge python=3.8 openfst pynini ngram baumwelch
conda activate fa
python -m pip install --upgrade pip

# MFA
pip install -U montreal-forced-aligner==2.0.0a22
pip install -U praatio==4.4.0
mfa download acoustic french_prosodylab
mfa download acoustic french_qc 
mfa download dictionary french_prosodylab
mfa thirdparty download

# SPPAS
wget "https://sourceforge.net/projects/sppas/files/SPPAS-3.7-2021-04-13.zip/download" -O sppas.zip
unzip sppas.zip -d SPPAS
rm sppas.zip
mv SPPAS sppas_src
cd sppas_src
sudo python3 sppas/bin/preinstall.py --julius --fra --fraquebec

# FA Script
pip install -U python-Levenshtein==0.12.2 pympi-ling==1.69 Unidecode==1.2.0 fuzzywuzzy==0.18.0 matplotlib