# Modélisation Thermique d'un Bâtiment

## Description
Ce projet vise à modéliser la thermique d'un bâtiment en utilisant une approche mathématique et une simulation sous Python. L'objectif est d'analyser la consommation énergétique en statique et en dynamique pour différents cas de figure influençant le bâtiment (isolation, ensoleillement, systèmes de régulation thermique, etc.).

## Structure du projet
Le dépôt contient les fichiers suivants :

- **THB_statique.py** : Code principal pour l'analyse statique du bâtiment.
- **THB_dynamique.py** : Code principal pour l'analyse dynamique du bâtiment.
- **dm4bem.py** : Module externe pour la modélisation thermique basée sur le circuit thermique équivalent.
- **Donnes_dynamiques.py** : Contient les paramètres dynamiques utilisés dans la simulation.
- **Rayonnement.py** : Modélisation du rayonnement thermique sur le bâtiment.
- **README.md** : Ce fichier de documentation.

## Installation et Exécution
### Prérequis
Assurez-vous d'avoir **Python 3.x** installé sur votre machine ainsi que les bibliothèques suivantes :
```bash
pip install numpy matplotlib scipy
```

### Exécution de l'analyse statique
```bash
python THB_statique.py
```

### Exécution de l'analyse dynamique
```bash
python THB_dynamique.py
```

## Contributeurs
- Al Khawand Francesco
- Aouame Chaimaa
- Bertrand Agathe
- Creyx Chloé
- Ferrand Shaun


## Licence
Ce projet est distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus de détails.


