# Génération et Augmentation d'Images avec des GANs

**Auteurs:** Ndeye Maguette MBAYE & DIARISSO Moussa
**Contexte:** Projet Intégrateur - Master 2 Signal et Télécommunications  

## 🎯 Présentation du Projet
 Ce projet a pour but d'étudier et de comparer différentes architectures de réseaux génératifs (GAN) appliquées aux images (datasets MNIST et FashionMNIST) . 

L'objectif est double : 
1.  Comprendre l'évolution des techniques pour stabiliser l'apprentissage des GANs.
2.  Appliquer ces modèles à un cas concret d'**augmentation de données** pour améliorer la classification en situation de manque de données .

## 🧠 Évolution des Architectures
Nous avons implémenté une progression technique en cinq étapes pour stabiliser la génération :
1.  **GAN-MLP (Baseline) :** Modèle basique avec réseaux denses .
2.  **DCGAN :** Intégration des convolutions pour le respect spatial .
3.  **WGAN (Weight Clipping) :** Introduction de la Wasserstein Distance .
4.  **WGAN-GP (Gradient Penalty) :** Résolution des problèmes de convergence.
5.  **Conditional cWGAN-GP :** Ajout d'un conditionnement (Labels) pour générer des images spécifiques à la demande.

## 📈 Le Bonus : Data Augmentation (Cas Pratique)
 L'expérience finale valide l'utilité pratique du modèle face à une pénurie de données (simulation avec seulement 2 000 images réelles). 
*  **Résultat :** L'ajout d'images synthétiques générées par notre Conditional WGAN-GP a permis d'améliorer significativement la précision du classifieur.

*(Insérer ici une image montrant la comparaison avant/après l'augmentation de données)*

## 🛠️ Environnement Technique
*  **Langage & Frameworks :** Python 3.13.9, PyTorch, Torchvision.
*  **Matériel :** Entraînement sur GPU NVIDIA RTX 4060.
*  **Format :** Jupyter Notebooks (`.ipynb`).

## 📊 Métriques d'Évaluation
| Méthode | Objectif |
| :--- | :--- |
| **FID** |  Mesure la distance entre les distributions réelles et générées. |
| **IS** |  Évalue la netteté et la diversité]. |
| **MS-SSIM** |  Détecte le Mode Collapse. |
| **Precision / Recall** |  Quantifie le réalisme et la diversité. |

## 🚀 Installation et Utilisation
1. Cloner le dépôt et installer les dépendances :
   ```bash
   pip install -r requirements.txt
3. Ouvrir les fichiers .ipynb via Jupyter Notebook ou votre IDE (VS Code, etc.).
4. Lancer l'entraînement (les datasets seront automatiquement téléchargés par Torchvision).
   
