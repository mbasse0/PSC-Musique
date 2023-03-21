# PSC-Musique
PSC sur la génération de musique par intelligence artificielle

Pour utiliser les notebook de test du git : MyBinder.org Attention ! il associe un projet mybinder en freezant le git

Pour produire un fichier requirements.txt, utiliser la bibliothèque pipreqs, cf https://stackoverflow.com/questions/31684375/automatically-create-requirements-txt


Etapes pour faire tourner sur une machine de l'X un script python:
- créer un virtual environment dans un dossier avec la commande : python3 -m venv Path
- activer le venv : source Path/bin/activate
- upgrade pip : python -m pip install --upgrade pip
- télécharger les versions de torch : python -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
- télécharger torchtext : python -m pip install torchtext==0.11.1
- télécharger music21 : python -m pip install music21==6.7.1
- télécharger lightning : python -m pip install lightning


Pour télécharger les databases:
GiantMIDI : https://github.com/bytedance/GiantMIDI-Piano/blob/master/disclaimer.md
Télécharger le lien Google Drive
WJD : https://jazzomat.hfm-weimar.de/download/download.html
Le lien contient : 
- un lien de téléchargement du fichier .db (où les mélodies sont formatées en séquences de string dans une colonne de la database)
- un ZIP contenant tous les MIDI de la Database (plus facile à utiliser pour nous), en descendant
