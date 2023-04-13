#!/usr/bin/bash

i=0

liste_ordis=('coccyx' 'cote' 'cubitus' 'cuboide' 'femur' 'frontal' 'humerus' 'malleole' 'metacarpe' 'parietal' 'perone' 'phalange' 'radius' 'rotule' 'sacrum' 'sternum')

for nom in ${liste_ordis[@]}
do
    ssh henri.duprieu@$nom.polytechnique.fr "cd PSCmusique/; source bin/activate; cd PSC-Musique/; cd JazzBot_Lightning/; export MASTER_ADDR=Coccyx.polytechnique.fr; export MASTER_PORT=45547; export NODE_RANK=$i; python3 main.py" &
    ((i+=1))
done

wait