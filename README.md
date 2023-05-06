# Pistes d'amélioration
- Mettre à jour l'app
- Avoir un rendu musical meilleur :
   - accords ?
   - Giant midi + nouvelle loss ?
   - bon sf2 (un peu jazzy)
   - meilleur fine tune en option sur l'app
- Avoir un meilleur rendu à l'oral
   - API clavier
   - travail théorique sur nos parties
   - préparation de la présentation
   - travail sur le mood / le ton / le langage
   




# PSC-Musique
PSC sur la génération de musique par intelligence artificielle

Pour utiliser les notebook de test du git : MyBinder.org Attention ! il associe un projet mybinder en freezant le git

Pour produire un fichier requirements.txt, utiliser la bibliothèque pipreqs, cf https://stackoverflow.com/questions/31684375/automatically-create-requirements-txt


Etapes pour faire tourner sur une machine de l'X un script python (installation rapide grace au .txt aproprié):
- créer un virtual environment dans un dossier avec la commande : python3 -m venv Path
- activer le venv : source Path/bin/activate
- upgrade pip : python -m pip install --upgrade pip
- télécharger les versions de torch : python -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
- télécharger torchtext : python -m pip install torchtext==0.11.1
- télécharger music21 : python -m pip install music21==6.7.1
- télécharger lightning : python -m pip install pytorch-lightning==1.5.10
- télécharger ray tune : pip install "ray[tune]"


\\

\\


Pour entrainer en DDP sur plusieurs ordis, il faut
- Décommenter la ligne de définition du Trainer version ddp et commenter celle sans ddp puis mettre le bon nombre de devices.


- Se connecter en SSH sur autant de devices, activer le venv et entrer la commande suivante :

(export MASTER_ADDR=Coccyx.polytechnique.fr;
export MASTER_PORT=45547;
export NODE_RANK=0;
python3 main.py)

Les parenthèses permettent d'écrire plusieurs lignes d'un coup (compier tout le paragraphe d'un coup). Deux contraintes :
Un des ordis doit être mis en MASRER_ADDR (à la place de Coccyx)
Chaque ordi doit avoir un NODE_RANK différent (0,1,2,...,N)
