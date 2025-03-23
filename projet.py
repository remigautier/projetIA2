"""
Projet d'Analyse de Sentiment
=============================
Ce projet vise à analyser une population d'annotateurs servant de base à un modèle d'analyse 
de sentiment, afin d'identifier leurs biais, complété par l'analyse des vocabulaires annotés.

Utilisation de Pyro pour la programmation probabiliste en Python.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Téléchargement des ressources NLTK requises
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configuration de Pyro
pyro.set_rng_seed(42)
pyro.clear_param_store()

# Création du dossier pour sauvegarder les graphiques
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Classe pour l'analyse des biais des annotateurs
class AnnotatorBiasAnalyzer:
    def __init__(self, data):  # Correction ici : __init__ au lieu de _init_
        """
        Initialise l'analyseur de biais des annotateurs
        
        Args:
            data (pd.DataFrame): DataFrame contenant les annotations avec colonnes:
                - text_id: identifiant du texte
                - annotator_id: identifiant de l'annotateur
                - sentiment: score attribué (ex: -2 à +2 ou 0 à 5)
        """
        self.data = data
        self.annotators = self.data['annotator_id'].unique()
        self.texts = self.data['text_id'].unique()
        self.num_annotators = len(self.annotators)
        self.num_texts = len(self.texts)
        
        # Création d'un mappage pour les ID des annotateurs et des textes
        self.annotator_to_idx = {ann: idx for idx, ann in enumerate(self.annotators)}
        self.text_to_idx = {text: idx for idx, text in enumerate(self.texts)}
        
        # Préparation des données pour Pyro
        self.annotations = np.zeros((self.num_texts, self.num_annotators))
        self.annotations.fill(np.nan)  # Remplir avec NaN car tous les annotateurs n'annotent pas tous les textes
        
        for _, row in self.data.iterrows():
            ann_idx = self.annotator_to_idx[row['annotator_id']]
            text_idx = self.text_to_idx[row['text_id']]
            self.annotations[text_idx, ann_idx] = row['sentiment']
    
    def model(self):
        """
        Modèle probabiliste pour l'analyse des biais des annotateurs
        
        Le modèle suppose que chaque texte a un sentiment "vrai" latent,
        et chaque annotateur a un biais (tendance à surestimer/sous-estimer)
        et une précision (inverse de la variance).
        """
        # Hyperparamètres sous forme de tenseurs PyTorch
        alpha_precision = torch.tensor(1.0)
        beta_precision = torch.tensor(1.0)
        
        # Priors pour les sentiments "vrais" des textes
        with pyro.plate("texts", self.num_texts):
            true_sentiments = pyro.sample(
                "true_sentiments",
                dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
            )
        
        # Priors pour les biais des annotateurs
        with pyro.plate("annotators", self.num_annotators):
            annotator_biases = pyro.sample(
                "annotator_biases",
                dist.Normal(torch.tensor(0.0), torch.tensor(1.0))
            )
            
            # Priors pour les précisions (inverse de la variance) des annotateurs
            annotator_precisions = pyro.sample(
                "annotator_precisions",
                dist.Gamma(alpha_precision, beta_precision)
            )
        
        # Pour chaque observation non-NaN, échantillonner selon le modèle
        for text_idx in range(self.num_texts):
            for ann_idx in range(self.num_annotators):
                if not np.isnan(self.annotations[text_idx, ann_idx]):
                    # La moyenne est le sentiment vrai plus le biais de l'annotateur
                    mean = true_sentiments[text_idx] + annotator_biases[ann_idx]
                    # La précision est spécifique à l'annotateur
                    precision = annotator_precisions[ann_idx]
                    
                    # Échantillonner l'annotation observée
                    pyro.sample(
                        f"obs_{text_idx}_{ann_idx}",
                        dist.Normal(mean, 1.0 / torch.sqrt(precision)),
                        obs=torch.tensor(self.annotations[text_idx, ann_idx], dtype=torch.float)
                    )
    
    def guide(self):
        """
        Guide pour l'inférence variationnelle
        """
        # Paramètres variationels pour les sentiments vrais
        true_sentiments_loc = pyro.param(
            "true_sentiments_loc",
            torch.zeros(self.num_texts)
        )
        true_sentiments_scale = pyro.param(
            "true_sentiments_scale",
            torch.ones(self.num_texts),
            constraint=dist.constraints.positive
        )
        
        # Paramètres variationels pour les biais des annotateurs
        annotator_biases_loc = pyro.param(
            "annotator_biases_loc",
            torch.zeros(self.num_annotators)
        )
        annotator_biases_scale = pyro.param(
            "annotator_biases_scale",
            torch.ones(self.num_annotators),
            constraint=dist.constraints.positive
        )
        
        # Paramètres variationels pour les précisions des annotateurs
        annotator_precisions_alpha = pyro.param(
            "annotator_precisions_alpha",
            torch.ones(self.num_annotators),
            constraint=dist.constraints.positive
        )
        annotator_precisions_beta = pyro.param(
            "annotator_precisions_beta",
            torch.ones(self.num_annotators),
            constraint=dist.constraints.positive
        )
        
        # Échantillonner les sentiments vrais
        with pyro.plate("texts", self.num_texts):
            true_sentiments = pyro.sample(
                "true_sentiments",
                dist.Normal(true_sentiments_loc, true_sentiments_scale)
            )
        
        # Échantillonner les biais et précisions des annotateurs
        with pyro.plate("annotators", self.num_annotators):
            # Échantillonner les biais des annotateurs
            annotator_biases = pyro.sample(
                "annotator_biases",
                dist.Normal(annotator_biases_loc, annotator_biases_scale)
            )
            
            # Échantillonner les précisions des annotateurs
            annotator_precisions = pyro.sample(
                "annotator_precisions",
                dist.Gamma(annotator_precisions_alpha, annotator_precisions_beta)
            )
    
    def train(self, num_iterations=1000, learning_rate=0.01, show_plot=True, save_plot=True):
        """
        Entraîne le modèle avec SVI (Stochastic Variational Inference)
        
        Args:
            num_iterations: Nombre d'itérations d'entraînement
            learning_rate: Taux d'apprentissage pour l'optimiseur Adam
            show_plot: Si True, affiche le graphique d'évolution de la perte
            save_plot: Si True, sauvegarde le graphique d'évolution de la perte
        
        Returns:
            Une liste des pertes ELBO à chaque itération
        """
        # Optimiseur Adam
        adam = Adam({"lr": learning_rate})
        
        # SVI avec trace ELBO
        svi = SVI(self.model, self.guide, adam, loss=Trace_ELBO())
        
        # Liste pour suivre les pertes
        losses = []
        
        # Entraînement
        for i in range(num_iterations):
            loss = svi.step()
            losses.append(loss)
            
            if (i + 1) % 100 == 0:
                print(f"Itération {i+1}/{num_iterations} | Perte: {loss:.4f}")
        
        # Visualiser les pertes d'entraînement
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Évolution de la perte durant l\'entraînement')
        plt.xlabel('Itération')
        plt.ylabel('Perte ELBO')
        plt.grid(alpha=0.3)
        
        # Sauvegarde du graphique
        if save_plot:
            plt.savefig(os.path.join(FIGURES_DIR, "evolution_perte.png"), dpi=300, bbox_inches='tight')
        
        # Affichage du graphique
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return losses
    
    def get_annotator_biases(self):
        """
        Récupère les biais estimés des annotateurs
        
        Returns:
            Un DataFrame avec les ID d'annotateurs et leurs biais estimés
        """
        biases = pyro.param("annotator_biases_loc").detach().numpy()
        precision_alpha = pyro.param("annotator_precisions_alpha").detach().numpy()
        precision_beta = pyro.param("annotator_precisions_beta").detach().numpy()
        
        # Calcul de la précision moyenne (alpha/beta pour distribution Gamma)
        precision = precision_alpha / precision_beta
        
        # Créer un DataFrame des résultats
        results = pd.DataFrame({
            'annotator_id': self.annotators,
            'bias': biases,
            'precision': precision
        })
        
        return results
    
    def get_true_sentiments(self):
        """
        Récupère les sentiments "vrais" estimés pour chaque texte
        
        Returns:
            Un DataFrame avec les ID de texte et leurs sentiments estimés
        """
        sentiments = pyro.param("true_sentiments_loc").detach().numpy()
        
        # Créer un DataFrame des résultats
        results = pd.DataFrame({
            'text_id': self.texts,
            'true_sentiment': sentiments
        })
        
        return results
    
    def plot_annotator_biases(self, show_plot=True, save_plot=True):
        """
        Visualise les biais des annotateurs et leur précision
        
        Args:
            show_plot: Si True, affiche les graphiques
            save_plot: Si True, sauvegarde les graphiques
        """
        results = self.get_annotator_biases()
        
        # Graphique des biais
        # Trier par biais pour une meilleure visualisation
        results_by_bias = results.sort_values('bias')
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Barplot des biais
        sns.barplot(data=results_by_bias, x='annotator_id', y='bias', ax=ax, color='skyblue')
        
        # Ajouter des lignes horizontales
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Ajouter des labels
        ax.set_title('Biais estimés des annotateurs')
        ax.set_xlabel('Annotateurs')
        ax.set_ylabel('Biais')
        
        # Rotation des labels d'axe x pour une meilleure lisibilité
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Sauvegarde du graphique
        if save_plot:
            plt.savefig(os.path.join(FIGURES_DIR, "biais_annotateurs.png"), dpi=300, bbox_inches='tight')
        
        # Affichage du graphique
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Graphique des précisions
        # Trier par précision pour une meilleure visualisation
        results_by_precision = results.sort_values('precision')
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Barplot des précisions
        sns.barplot(data=results_by_precision, x='annotator_id', y='precision', ax=ax, color='lightgreen')
        
        # Ajouter des labels
        ax.set_title('Précision estimée des annotateurs')
        ax.set_xlabel('Annotateurs')
        ax.set_ylabel('Précision')
        
        # Rotation des labels d'axe x pour une meilleure lisibilité
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Sauvegarde du graphique
        if save_plot:
            plt.savefig(os.path.join(FIGURES_DIR, "precision_annotateurs.png"), dpi=300, bbox_inches='tight')
        
        # Affichage du graphique
        if show_plot:
            plt.show()
        else:
            plt.close()


# Classe pour l'analyse du vocabulaire en fonction des sentiments
class VocabularyAnalyzer:
    def __init__(self, texts, sentiments, language='french'): 
        """
        Initialise l'analyseur de vocabulaire
        
        Args:
            texts (list): Liste des textes à analyser
            sentiments (list): Liste des scores de sentiment (même longueur que texts)
            language (str): Langue pour les stopwords ('french' ou 'english')
        """
        self.texts = texts
        self.sentiments = sentiments
        self.language = language
        self.stopwords = set(stopwords.words(language))
        
        # Initialisation des variables pour stocker les résultats
        self.word_sentiment_scores = {}
        self.vocab_df = None
        self.vectorizer = None
        self.features = None
    
    def preprocess_text(self, text):
        """
        Prétraitement du texte : tokenisation et suppression des stopwords
        
        Args:
            text (str): Texte à prétraiter
            
        Returns:
            list: Liste de tokens (mots) nettoyés
        """
        # Tokenisation
        tokens = word_tokenize(text.lower())
        
        # Suppression des stopwords et de la ponctuation
        tokens = [word for word in tokens if word.isalpha() and word not in self.stopwords]
        
        return tokens
    
    def calculate_word_sentiment_scores(self):
        """
        Calcule le score de sentiment moyen pour chaque mot du corpus
        """
        # Dictionnaire pour stocker la somme des scores et le nombre d'occurrences
        word_sentiment_sum = {}
        word_count = {}
        
        # Parcourir chaque texte avec son score de sentiment
        for text, sentiment in zip(self.texts, self.sentiments):
            tokens = self.preprocess_text(text)
            
            # Mettre à jour les statistiques pour chaque mot
            for word in tokens:
                if word in word_sentiment_sum:
                    word_sentiment_sum[word] += sentiment
                    word_count[word] += 1
                else:
                    word_sentiment_sum[word] = sentiment
                    word_count[word] = 1
        
        # Calculer le score moyen pour chaque mot
        for word in word_sentiment_sum:
            self.word_sentiment_scores[word] = word_sentiment_sum[word] / word_count[word]
        
        # Créer un DataFrame pour une analyse plus facile
        words = list(self.word_sentiment_scores.keys())
        scores = list(self.word_sentiment_scores.values())
        frequencies = [word_count[word] for word in words]
        
        self.vocab_df = pd.DataFrame({
            'word': words,
            'sentiment_score': scores,
            'frequency': frequencies
        })
        
        # Trier par fréquence décroissante
        self.vocab_df = self.vocab_df.sort_values('frequency', ascending=False)
    
    def get_top_words(self, n=20, by_sentiment=True):
        """
        Récupère les n mots les plus positifs et négatifs, ou les plus fréquents
        
        Args:
            n (int): Nombre de mots à récupérer
            by_sentiment (bool): Si True, trie par sentiment; sinon par fréquence
            
        Returns:
            tuple: (DataFrame des mots positifs, DataFrame des mots négatifs)
                   ou simplement DataFrame des mots les plus fréquents
        """
        if self.vocab_df is None:
            self.calculate_word_sentiment_scores()
        
        # Filtrer les mots avec une fréquence minimale pour éviter le bruit
        min_frequency = 3
        filtered_df = self.vocab_df[self.vocab_df['frequency'] >= min_frequency]
        
        if by_sentiment:
            # Trier par score de sentiment
            positive_words = filtered_df.sort_values('sentiment_score', ascending=False).head(n)
            negative_words = filtered_df.sort_values('sentiment_score', ascending=True).head(n)
            return positive_words, negative_words
        else:
            # Trier par fréquence
            top_words = filtered_df.sort_values('frequency', ascending=False).head(n)
            return top_words
    
    def extract_features(self, min_df=5, max_df=0.9):
        """
        Extrait les caractéristiques TF-IDF du corpus
        
        Args:
            min_df (int): Fréquence minimale du document pour inclure un terme
            max_df (float): Fréquence maximale du document pour inclure un terme
            
        Returns:
            scipy.sparse.csr_matrix: Matrice des caractéristiques
        """
        # Initialisation du vectoriseur TF-IDF
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words=self.stopwords,
            ngram_range=(1, 2)  # Unigrammes et bigrammes
        )
        
        # Transformation du corpus
        self.features = self.vectorizer.fit_transform(self.texts)
        
        return self.features
    
    def plot_word_cloud(self, sentiment_threshold=0.0, show_plot=True, save_plot=True):
        """
        Génère et affiche un nuage de mots coloré selon le sentiment
        
        Args:
            sentiment_threshold (float): Seuil pour séparer les mots positifs et négatifs
            show_plot: Si True, affiche le nuage de mots
            save_plot: Si True, sauvegarde le nuage de mots
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("Veuillez installer le package wordcloud: pip install wordcloud")
            return
        
        if self.vocab_df is None:
            self.calculate_word_sentiment_scores()
        
        # Filtrer pour les mots avec une fréquence minimale
        min_frequency = 3
        filtered_df = self.vocab_df[self.vocab_df['frequency'] >= min_frequency]
        
        # Créer un dictionnaire de fréquences
        word_freq = dict(zip(filtered_df['word'], filtered_df['frequency']))
        
        # Créer un dictionnaire pour la colormap basée sur les sentiments
        word_to_sentiment = dict(zip(filtered_df['word'], filtered_df['sentiment_score']))
        
        def color_func(word, **kwargs):
            sentiment = word_to_sentiment.get(word, 0)
            if sentiment > sentiment_threshold:
                # Plus le sentiment est positif, plus la couleur est verte
                r = int(max(0, 255 * (1 - sentiment)))
                g = 255
                b = int(max(0, 255 * (1 - sentiment)))
            else:
                # Plus le sentiment est négatif, plus la couleur est rouge
                r = 255
                g = int(max(0, 255 * (1 + sentiment)))
                b = int(max(0, 255 * (1 + sentiment)))
            return f"rgb({r}, {g}, {b})"
        
        # Générer le nuage de mots
        wc = WordCloud(
            background_color='white',
            max_words=200,
            width=800,
            height=400,
            color_func=color_func
        )
        
        wc.generate_from_frequencies(word_freq)
        
        # Afficher le nuage de mots
        plt.figure(figsize=(16, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nuage de mots coloré selon le sentiment', fontsize=20)
        
        # Sauvegarde du graphique
        if save_plot:
            plt.savefig(os.path.join(FIGURES_DIR, "nuage_mots.png"), dpi=300, bbox_inches='tight')
        
        # Affichage du graphique
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_sentiment_distribution(self, show_plot=True, save_plot=True):
        """
        Trace la distribution des scores de sentiment dans le corpus
        
        Args:
            show_plot: Si True, affiche le graphique
            save_plot: Si True, sauvegarde le graphique
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.sentiments, bins=20, kde=True, color='purple')
        plt.title('Distribution des scores de sentiment')
        plt.xlabel('Score de sentiment')
        plt.ylabel('Fréquence')
        plt.grid(alpha=0.3)
        
        # Sauvegarde du graphique
        if save_plot:
            plt.savefig(os.path.join(FIGURES_DIR, "distribution_sentiment.png"), dpi=300, bbox_inches='tight')
        
        # Affichage du graphique
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_top_words(self, n=10, show_plot=True, save_plot=True):
        """
        Visualise les mots les plus positifs et négatifs
        
        Args:
            n (int): Nombre de mots à afficher
            show_plot: Si True, affiche le graphique
            save_plot: Si True, sauvegarde le graphique
        """
        if self.vocab_df is None:
            self.calculate_word_sentiment_scores()
        
        # Obtenir les mots les plus positifs et négatifs
        positive_words, negative_words = self.get_top_words(n=n)
        
        # Configurer la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Tracer les mots positifs (syntaxe compatible avec seaborn récent)
        sns.barplot(data=positive_words, x='sentiment_score', y='word', ax=ax1, color='green')
        ax1.set_title(f'Top {n} mots positifs')
        ax1.set_xlabel('Score de sentiment')
        
        # Tracer les mots négatifs
        sns.barplot(data=negative_words, x='sentiment_score', y='word', ax=ax2, color='red')
        ax2.set_title(f'Top {n} mots négatifs')
        ax2.set_xlabel('Score de sentiment')
        
        plt.tight_layout()
        
        # Sauvegarde du graphique
        if save_plot:
            plt.savefig(os.path.join(FIGURES_DIR, "top_words.png"), dpi=300, bbox_inches='tight')
        
        # Affichage du graphique
        if show_plot:
            plt.show()
        else:
            plt.close()


# Fonction pour générer des données synthétiques pour la démonstration
def generate_synthetic_data(num_texts=100, num_annotators=5, num_annotations_per_text=3, seed=42):
    """
    Génère des données synthétiques d'annotations de sentiment
    
    Args:
        num_texts: Nombre de textes
        num_annotators: Nombre d'annotateurs
        num_annotations_per_text: Nombre d'annotations par texte
        seed: Graine aléatoire pour la reproductibilité
        
    Returns:
        tuple: (DataFrame des annotations, dictionnaire des textes)
    """
    np.random.seed(seed)
    
    # Générer des caractéristiques latentes pour les textes et les annotateurs
    true_sentiments = np.random.normal(0, 1, num_texts)
    annotator_biases = np.random.normal(0, 0.5, num_annotators)
    annotator_precisions = np.random.gamma(2, 0.5, num_annotators)
    
    # Créer des textes synthétiques basés sur les sentiments
    texts = {}
    
    positive_words = ["excellent", "superbe", "fantastique", "agréable", "heureux", 
                      "merveilleux", "parfait", "plaisir", "satisfait", "réussi"]
    negative_words = ["terrible", "horrible", "mauvais", "médiocre", "déçu", 
                      "catastrophique", "pénible", "frustré", "ennuyeux", "décevant"]
    neutral_words = ["normal", "moyen", "correct", "standard", "habituel", 
                     "ordinaire", "commun", "régulier", "typique", "conventionnel"]
    
    # Vocabulaire commun
    common_words = ["le", "la", "un", "une", "est", "dans", "pour", "avec", "sur", "qui", 
                    "que", "ce", "se", "au", "du", "et", "à", "en", "de"]
    
    for i in range(num_texts):
        sentiment = true_sentiments[i]
        
        # Déterminer le nombre de mots de chaque catégorie selon le sentiment
        if sentiment > 0.5:
            # Texte plutôt positif
            num_pos = np.random.randint(3, 6)
            num_neg = np.random.randint(0, 2)
            num_neut = np.random.randint(1, 4)
        elif sentiment < -0.5:
            # Texte plutôt négatif
            num_pos = np.random.randint(0, 2)
            num_neg = np.random.randint(3, 6)
            num_neut = np.random.randint(1, 4)
        else:
            # Texte neutre
            num_pos = np.random.randint(1, 3)
            num_neg = np.random.randint(1, 3)
            num_neut = np.random.randint(2, 5)
        
        # Sélectionner des mots aléatoires de chaque catégorie
        selected_pos = np.random.choice(positive_words, num_pos, replace=False)
        selected_neg = np.random.choice(negative_words, num_neg, replace=False)
        selected_neut = np.random.choice(neutral_words, num_neut, replace=False)
        selected_common = np.random.choice(common_words, np.random.randint(5, 10), replace=True)
        
        # Mélanger tous les mots et créer un texte
        all_words = list(selected_pos) + list(selected_neg) + list(selected_neut) + list(selected_common)
        np.random.shuffle(all_words)
        text = " ".join(all_words)
        
        texts[f"text_{i}"] = text
    
    # Générer des annotations
    annotations = []
    
    for text_id in range(num_texts):
        # Sélectionner aléatoirement des annotateurs pour ce texte
        selected_annotators = np.random.choice(range(num_annotators), 
                                               min(num_annotations_per_text, num_annotators), 
                                               replace=False)
        
        true_sentiment = true_sentiments[text_id]
        
        for ann_id in selected_annotators:
            # Calculer l'annotation avec bruit
            bias = annotator_biases[ann_id]
            precision = annotator_precisions[ann_id]
            noise = np.random.normal(0, 1.0 / np.sqrt(precision))
            
            # L'annotation est le sentiment vrai plus le biais de l'annotateur plus du bruit
            sentiment_score = true_sentiment + bias + noise
            
            # Normaliser sur une échelle de 1 à 5 pour simuler une annotation réelle
            normalized_score = min(max(int((sentiment_score + 2.5) * 1.0), 1), 5)
            
            annotations.append({
                'text_id': f"text_{text_id}",
                'annotator_id': f"annotator_{ann_id}",
                'sentiment': normalized_score
            })
    
    # Créer un DataFrame
    df = pd.DataFrame(annotations)
    
    return df, texts


# Fonction pour sauvegarder les résultats dans un fichier CSV
def save_results(data, filename):
    """
    Sauvegarde les résultats dans un fichier CSV
    
    Args:
        data (pd.DataFrame): DataFrame à sauvegarder
        filename (str): Nom du fichier (sans extension)
    """
    output_dir = "resultats"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{filename}.csv")
    data.to_csv(filepath, index=False)
    print(f"Résultats sauvegardés dans {filepath}")


# Exemple d'utilisation
def main():
    print("=" * 80)
    print("PROJET D'ANALYSE DE SENTIMENT")
    print("=" * 80)
    
    # Générer des données synthétiques
    print("\nGénération de données synthétiques...")
    annotations_df, texts_dict = generate_synthetic_data(
        num_texts=100,
        num_annotators=8,
        num_annotations_per_text=4
    )
    
    print(f"Données générées: {len(annotations_df)} annotations pour {len(texts_dict)} textes")
    
    # Analyser les biais des annotateurs
    print("\nAnalyse des biais des annotateurs...")
    bias_analyzer = AnnotatorBiasAnalyzer(annotations_df)
    losses = bias_analyzer.train(num_iterations=500)
    
    # Visualiser les biais des annotateurs
    bias_analyzer.plot_annotator_biases()
    
    # Obtenir les sentiments "vrais" estimés
    true_sentiments = bias_analyzer.get_true_sentiments()
    
    # Sauvegarder les résultats
    save_results(bias_analyzer.get_annotator_biases(), "biais_annotateurs")
    save_results(true_sentiments, "sentiments_vrais")
    
    # Préparer les données pour l'analyse du vocabulaire
    texts = [texts_dict[text_id] for text_id in true_sentiments['text_id']]
    sentiments = true_sentiments['true_sentiment'].values
    
    # Analyser le vocabulaire
    print("\nAnalyse du vocabulaire...")
    vocab_analyzer = VocabularyAnalyzer(texts, sentiments)
    vocab_analyzer.calculate_word_sentiment_scores()
    
    # Visualiser la distribution des sentiments
    vocab_analyzer.plot_sentiment_distribution()
    
    # Visualiser les mots les plus positifs et négatifs
    vocab_analyzer.plot_top_words(n=10)
    
    # Essayer de générer un nuage de mots (si wordcloud est installé)
    try:
        vocab_analyzer.plot_word_cloud()
    except Exception as e:
        print(f"Impossible de générer le nuage de mots: {e}")
    
    # Afficher quelques statistiques sur le vocabulaire
    positive_words, negative_words = vocab_analyzer.get_top_words(n=10)
    print("\nMots les plus positifs:")
    print(positive_words[['word', 'sentiment_score', 'frequency']])
    
    print("\nMots les plus négatifs:")
    print(negative_words[['word', 'sentiment_score', 'frequency']])
    
    # Sauvegarder les résultats du vocabulaire
    save_results(vocab_analyzer.vocab_df.sort_values('sentiment_score', ascending=False), "vocabulaire_sentiment")
    
    # Conclusion
    print("\nConclusion de l'analyse:")
    print("1. Nous avons identifié les biais des annotateurs")
    print("2. Nous avons estimé les sentiments 'vrais' des textes")
    print("3. Nous avons analysé le vocabulaire associé aux différents niveaux de sentiment")
    print("\nTous les graphiques ont été sauvegardés dans le dossier 'figures/'")
    print("Tous les résultats numériques ont été sauvegardés dans le dossier 'resultats/'")


if __name__ == "__main__":
    main()