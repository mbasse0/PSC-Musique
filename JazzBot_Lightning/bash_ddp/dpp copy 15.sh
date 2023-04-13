#!/bin/bash
source bin/activate
cd PSC-Musique/
cd JazzBot_Lightning/
export MASTER_ADDR=Coccyx.polytechnique.fr
export MASTER_PORT=45547 
export NODE_RANK=15
python3 main.py