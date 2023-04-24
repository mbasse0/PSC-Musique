#!/usr/bin/env python
# coding: utf-8

# In[26]:


import csv
import numpy as np
import matplotlib.pyplot as plt


def apparition_pitch(filename):
    apparition_tab=[0]*128
    
    with open(filename, mode='r') as csv_file:
    
    # Créer un objet de lecteur CSV
        csv_reader = csv.reader(csv_file)
    
    # Boucle à travers chaque ligne dans le fichier CSV
        for row in csv_reader:
        # Accéder à chaque élément dans la ligne
            i=0
            while i<len(row):
                char=row[i]
                num=int(char[1:])
            
                apparition_tab[num]+=1
                              

                i+=4
            
        return apparition_tab

def apparition_duration(filename):
    apparition_tab=[0]*96
    
    with open(filename, mode='r') as csv_file:
    
    # Créer un objet de lecteur CSV
        csv_reader = csv.reader(csv_file)
    
    # Boucle à travers chaque ligne dans le fichier CSV
        for row in csv_reader:
        # Accéder à chaque élément dans la ligne
            i=1
            while i<len(row):
                char=row[i]
                num=int(char[1:])
            
                apparition_tab[num]+=1
                              

                i+=4
            
        return apparition_tab


def apparition_time(filename):
    apparition_tab=[0]*192
    
    with open(filename, mode='r') as csv_file:
    
    # Créer un objet de lecteur CSV
        csv_reader = csv.reader(csv_file)
    
    # Boucle à travers chaque ligne dans le fichier CSV
        for row in csv_reader:
        # Accéder à chaque élément dans la ligne
            i=2
            while i<len(row):
                char=row[i]
                num=int(char[1:])
                
            
                apparition_tab[num]+=1
                              

                i+=4
           
        return apparition_tab
    
def apparition_velocity(filename):
    apparition_tab=[0]*128
    
    with open(filename, mode='r') as csv_file:
    
    # Créer un objet de lecteur CSV
        csv_reader = csv.reader(csv_file)
    
    # Boucle à travers chaque ligne dans le fichier CSV
        for row in csv_reader:
        # Accéder à chaque élément dans la ligne
            i=3
            while i<len(row):
                char=row[i]
                num=int(char[1:])
            
                apparition_tab[num]+=1
                              

                i+=4
            
        return apparition_tab
    
    
def probacondi_pitch(filename):
    proba_apparition=np.zeros((12,12))
    with open(filename, mode='r') as csv_file:
    
    # Créer un objet de lecteur CSV
        csv_reader = csv.reader(csv_file)
    
    # Boucle à travers chaque ligne dans le fichier CSV
        for row in csv_reader:
        # Accéder à chaque élément dans la ligne
            i=0
            while i<len(row):
                char=row[i]
                num=int(char[1:])
            
                if i+4<len(row):
                    charsuivant=row[i+4]
                    numsuivant=int(charsuivant[1:])
                    proba_apparition[num%12][numsuivant%12]+=1
                    
                              

                i+=4
           
    for i in range(12):
        
        current_tot=np.sum(proba_apparition[i,:])
        proba_apparition[i,:]*=1/current_tot
        
     
    return proba_apparition


def affichehisto(tableau, type_de_token):
    
    n = len(tableau)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(15,10))
    ax.bar(x, tableau)
    ax.set_xlabel(type_de_token)
    ax.set_ylabel("Fréquence d'apparition")
    #ax.set_xticks(x)
    #ax.set_xticklabels(np.arange(1, n+1))
    ax.set_xticks(x[::5])
    ax.set_xticklabels(np.arange(1, n+1)[::5])
    
    '''ce qui suit seulement si on regarde le pitch''' 
    for i in range(n):
        if i % 12 == 0:
            ax.get_children()[i].set_fc('r')
    
    plt.show()

    
def affiche_camembert(filename, pitch):
    #choisissez une classeb de pitch (entre 0 et 11) pour afficher ses proba condi
    probas=probacondi_pitch(filename)[pitch, :]
    fig, ax = plt.subplots()
    ax.pie(probas, labels=range(0,12), autopct='%1.1f%%')
    ax.axis('equal')

    # Affichage du graphique
    plt.show()
    
#affiche_camembert("bigweimar.csv", 0)

#affichehisto(np.array(apparition_velocity("bigweimar.csv")), "Velocity")


#affichehisto(np.array(apparition_time("mediumrare.csv")), "Time")

''' pour afficher les histogramme, mettre affichehisto(np.array(apparition_pitch(nom_du_doc)), quel_type_de_token)
en remplaçant apparition_pitch par la fonction qui t'intéresse. Si on regarde pas le pitch, commenter la partie pour mettre en rouge'''


# In[ ]:




