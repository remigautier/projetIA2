# Projet d'Analyse de Sentiment

## Description
Ce projet vise à analyser une population d'annotateurs servant de base à un modèle d'analyse de sentiment. L'objectif est d'identifier leurs biais en complétant l'analyse par une étude des vocabulaires annotés.

L'approche adoptée repose sur la programmation probabiliste avec la bibliothèque **Pyro** en Python.

## Fonctionnalités
- Analyse des biais des annotateurs
- Estimation des sentiments "vrais" des textes
- Visualisation des biais et précisions des annotateurs
- Analyse des vocabulaires en fonction des sentiments
- Extraction de caractéristiques textuelles avec TF-IDF
- Génération de nuages de mots colorés selon le sentiment

## Technologies Utilisées
- **Python** (3.x)
- **Pyro** (programmation probabiliste)
- **NLTK** (traitement du langage naturel)
- **Scikit-learn** (TF-IDF et évaluation de modèle)
- **Matplotlib / Seaborn** (visualisation)
- **Pandas / NumPy** (manipulation de données)

## Installation
### Prérequis
Assurez-vous d'avoir **Python 3.x** installé sur votre système.

### Installation des dépendances
Utilisez `pip` pour installer les dépendances nécessaires :
```bash
pip install pyro-ppl numpy pandas nltk matplotlib seaborn scikit-learn wordcloud
```

Téléchargez également les ressources NLTK :
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Utilisation
### 1. Charger les données
Le code attend un fichier CSV contenant des annotations avec les colonnes suivantes :
- `text_id` : identifiant du texte
- `annotator_id` : identifiant de l'annotateur
- `sentiment` : score de sentiment attribué (ex: -2 à +2 ou 0 à 5)

### 2. Exécuter le modèle
Dans un script Python, chargez les données et exécutez le modèle :
```python
from projet2 import AnnotatorBiasAnalyzer
import pandas as pd

data = pd.read_csv("annotations.csv")
analyzer = AnnotatorBiasAnalyzer(data)
analyzer.train(num_iterations=1000, learning_rate=0.01)
analyzer.plot_annotator_biases()
```

### 3. Analyser les résultats
- **Biais des annotateurs** : `analyzer.get_annotator_biases()`
- **Sentiments "vrais" estimés** : `analyzer.get_true_sentiments()`
- **Nuage de mots** : `vocab_analyzer.plot_word_cloud()`

## Structure du Projet
```
/
|-- projet2.py        # Code principal du projet
|-- annotations.csv   # Données d'exemple (si disponible)
|-- figures/         # Dossiers contenant les visualisations générées
|-- README.md        # Documentation
```

## Auteurs
Projet réalisé dans le cadre du cours "Programmation probabiliste, Théorie des jeux & Machine Learning".

## Licence
Ce projet est sous licence MIT. Vous êtes libre de le modifier et de le redistribuer.

