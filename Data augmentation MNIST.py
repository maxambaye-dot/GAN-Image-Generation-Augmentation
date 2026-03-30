import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np

# ---  CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
print(f"Expérience lancée sur : {device}")

# ---  PRÉPARATION DES DONNÉES ---
# On normalise entre -1 et 1 (comme le GAN)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

#  Chargement des VRAIES données (MNIST)
# On télécharge tout, mais on n'en utilisera qu'une partie
# CORRECTION IMPORTANTE : target_transform force le label à être un tenseur
# Cela évite les bugs quand on mixera avec les faux labels plus tard
full_train_data = datasets.MNIST(
    root='dataset/', 
    train=True, 
    download=True, 
    transform=transform,
    target_transform=lambda y: torch.tensor(y) 
)

# Données de test (pour vérifier la note finale)
test_data = datasets.MNIST(
    root='dataset/', 
    train=False, 
    download=True, 
    transform=transform,
    target_transform=lambda y: torch.tensor(y)
)

#  Création de la "Pénurie" 
# On ne garde que les 2000 premières images réelles.
indices = range(2000) 
real_subset = Subset(full_train_data, indices)

print("Chargement des images générées...")
try:
    # On charge les fichiers créés à l'étape précédente
    fake_imgs = torch.load('C-WGAN-GP_images_MNIST.pt')
    fake_lbls = torch.load('C-WGAN-GP_labels_MNIST.pt')
    
    # On crée un Dataset PyTorch avec
    fake_dataset = TensorDataset(fake_imgs, fake_lbls)
    print(" Fichiers .pt chargés avec succès !")
    
except FileNotFoundError:
    print(" ERREUR : Les fichiers 'C-WGAN-GP_images_MNIST.pt' ou 'C-WGAN-GP_labels_MNIST' sont introuvables.")
    raise # On arrête tout si pas de fichier

# Création du Dataset "Mixte" (Réel + Fake)
# C'est ici qu'on fait la "Data Augmentation"
augmented_dataset = ConcatDataset([real_subset, fake_dataset])

#  Création des DataLoaders (Les distributeurs de données)
# Loader 1 : Seulement les 2000 vraies
loader_baseline = DataLoader(real_subset, batch_size=batch_size, shuffle=True)
# Loader 2 : Les 2000 vraies + les 10000 fausses
loader_augmented = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
# Loader 3 : Le test
loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print(f" Données Baseline (Réel seul) : {len(real_subset)} images")
print(f" Données Augmentées (Réel + GAN) : {len(augmented_dataset)} images")

# ---  LE MODÈLE 'ÉLÈVE' (Simple CNN) ---
# C'est un petit réseau de neurones classique pour la classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Partie Extraction de caractéristiques (Convolution)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        # Partie Classification (Linéaire)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128), nn.ReLU(),
            nn.Linear(128, 10) # 10 sorties pour les 10 chiffres
        )

    def forward(self, x):
        return self.classifier(self.features(x))

  def train_model(name, train_loader, epochs=20): 
    print(f"\n Entraînement : {name}...")
    model = SimpleCNN().to(device) # On crée un nouveau modèle vierge à chaque fois
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    accuracies = []
    
    for epoch in range(epochs):
        # Mode entraînement
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
        # Mode Test (Évaluation sur les 10 000 images officielles)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader_test:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = torch.max(out.data, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        
        acc = 100 * correct / total
        accuracies.append(acc)
        
        if (epoch+1) % 5 == 0: # Pour afficher la précision à chaque 5 epochs
            print(f"   Epoch {epoch+1}/{epochs} | Accuracy Test: {acc:.2f}%")
        
    return accuracies
# Round 1 : Sans GAN 
scores_baseline = train_model("BASELINE (Seulement Réel)", loader_baseline)

# Round 2 : Avec GAN 
scores_augmented = train_model("AUGMENTÉ (+ Conditional WGAN-GP)", loader_augmented)

plt.figure(figsize=(10, 6))
# Courbe rouge : Sans GAN
plt.plot(scores_baseline, label='Sans GAN (Baseline)', marker='o', linestyle='--', color='red')
# Courbe verte : Avec GAN
plt.plot(scores_augmented, label='Avec Conditional WGAN-GP', marker='o', linewidth=2, color='green')

plt.title("Comparaison des performances du classifieur CNN avec et sans apport de données synthétiques (MNIST)")
plt.xlabel("Époques")
plt.ylabel("Précision (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Affichage des Performances
print(f"\n Accuracy Sans GAN : {scores_baseline[-1]:.2f}%")
print(f" Accuracy Avec C-WGAN-GP : {scores_augmented[-1]:.2f}%")
gain = scores_augmented[-1] - scores_baseline[-1]
print(f" Gain de performance : +{gain:.2f}%")
