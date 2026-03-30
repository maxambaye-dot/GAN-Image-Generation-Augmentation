import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid
import warnings
import time
import matplotlib.pyplot as plt

# --- IMPORTS POUR LES MÉTRIQUES ---
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchvision.models import inception_v3

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration de l'entraînement
lrd = 0.0002         # Taux d'apprentissage du Discriminateur (Learning Rate)
lrg = 0.0002         # Taux d'apprentissage du Générateur
beta1 = 0.5          # Paramètre Beta1 pour l'optimiseur Adam (stabilité du DCGAN)
noise_dim = 100      # Dimension du vecteur latent (bruit en entrée)
batch_size = 64     # Nombre d'images par lot (Batch size)
epochs = 250          # Nombre de cycles d'entraînement

# Configuration de l'architecture
channels_img = 1     # Nombre de canaux de l'image (1 pour Noir & Blanc)
features_d = 64      # Profondeur des filtres du Discriminateur
features_g = 64      # Profondeur des filtres du Générateur

print(f"Entraînement configuré sur : {device}")

# Transformation : Normalisation en [-1, 1] pour coller avec la sortie Tanh du générateur
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Chargement du dataset MNIST
dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
train = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Dataset chargé avec succès !")

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        
        # Le réseau est séquentiel : l'image traverse les couches une par une
        self.disc = nn.Sequential(
            # --- Étape 1 : Analyse globale ---
            # Entrée : Image 28x28
            # Conv2d réduit la taille par 2 (stride=2) -> Sortie 14x14
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2), # On laisse passer un peu de négatif pour ne pas "tuer" les neurones
            
            # --- Étape 2 : Analyse des détails intermédiaires ---
            # Entrée : 14x14 -> Sortie : 7x7
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 2), # La normalisation aide à stabiliser l'apprentissage
            nn.LeakyReLU(0.2),
            
            # --- Étape 3 : Analyse fine ---
            # Entrée : 7x7 -> Sortie : 4x4 (avec kernel=3, stride=2, padding=1)
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),
            
            # --- Étape 4 : Verdict final ---
            # On aplatit les cartes de caractéristiques 4x4 pour passer dans une couche linéaire
            nn.Flatten(),
            nn.Linear(features_d * 4 * 4 * 4, 1) # Sortie : 1 seul chiffre (le score)
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        
        self.net = nn.Sequential(
            # Entrée : Vecteur de bruit (Batch, noise_dim, 1, 1)
            
            # --- Étape 1 : La graine ---
            # On part du bruit (1x1) pour créer un premier petit bloc de 3x3
            nn.ConvTranspose2d(channels_noise, features_g * 4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(), # Le générateur préfère ReLU (sauf à la fin)
            
            # --- Étape 2 : Première croissance ---
            # Passage de 3x3 -> 7x7
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),
            
            # --- Étape 3 : Affinement ---
            # Passage de 7x7 -> 14x14
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g),
            nn.ReLU(),
            
            # --- Étape 4 : Finalisation (Taille MNIST) ---
            # Passage de 14x14 -> 28x28
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1),
            
            # IMPORTANT : Tanh force les pixels entre [-1, 1], 
            # c'est indispensable pour la stabilité du GAN.
            nn.Tanh() 
        )

    def forward(self, x):
        return self.net(x)

# --- Initialisation des Poids ---

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

      disc = Discriminator(channels_img, features_d).to(device)
gen = Generator(noise_dim, channels_img, features_g).to(device)

# On applique notre initialisation pour que les poids partent sur de bonnes bases
initialize_weights(disc)
initialize_weights(gen)

# --- OUTILS DE MESURE ---
print("Préparation des métriques (FID, IS, MS-SSIM, PR)...")
# Ces outils nous aideront à savoir si nos images ressemblent vraiment à des chiffres
fid = FrechetInceptionDistance(feature=2048).to(device)
inception = InceptionScore(feature='logits_unbiased', normalize=False).to(device)

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

pr_inception = inception_v3(pretrained=True, transform_input=False).to(device)
pr_inception.fc = nn.Identity() # On garde les features, pas la classification
pr_inception.eval()

#  Fonctions utilitaires pour le calcul du Precision & Recall (PR)
def compute_pairwise_distance(data_x, data_y=None):
    if data_y is None: data_y = data_x
    dists = torch.cdist(data_x, data_y, p=2)
    return dists

def get_kth_value(unsorted, k, axis=-1):
    indices = torch.topk(unsorted, k, dim=axis, largest=False).indices
    k_smallests = torch.gather(unsorted, axis, indices)
    return k_smallests[:, -1]

def compute_precision_recall(real_features, fake_features, k=3):
    real_dists = compute_pairwise_distance(real_features)
    real_dists.fill_diagonal_(float('inf'))
    real_radii = get_kth_value(real_dists, k)
    fake_dists = compute_pairwise_distance(fake_features)
    fake_dists.fill_diagonal_(float('inf'))
    fake_radii = get_kth_value(fake_dists, k)
    cross_dists = compute_pairwise_distance(real_features, fake_features)
    precision = (cross_dists.T <= real_radii).any(dim=1).float().mean().item()
    recall = (cross_dists <= fake_radii).any(dim=1).float().mean().item()
    return precision, recall

# Petite fonction utilitaire pour créer le bruit (Input du générateur)
def generate_noise(b, n):
    return torch.randn(b, n, 1, 1).to(device)

# --- OPTIMISATION ---
# Scalers pour la précision mixte 
dscaler = GradScaler()
gscaler = GradScaler()

# Optimiseurs Adam avec les paramètres ajustés (learning rate et beta1)
optim_disc = Adam(disc.parameters(), lr=lrd, betas=(beta1, 0.999))
optim_gen = Adam(gen.parameters(), lr=lrg, betas=(beta1, 0.999))

# Fonction de perte (BCEWithLogitsLoss est plus stable numériquement)
criterion = nn.BCEWithLogitsLoss()

# --- HISTORIQUES ---
# Pour stocker les courbes et les afficher à la fin
epoch_losses_d = []
epoch_losses_g = []


# On fige un bruit de départ pour voir comment le générateur évolue sur le MÊME exemple
#fixed_noise = generate_noise(64, noise_dim)

def evaluate_initial_state():
    print("Evaluation de l'état initial (Epoch 0)...")
    
    # Mode évaluation (on fige les modèles, pas d'apprentissage ici)
    disc.eval()
    gen.eval()
    
    running_loss_d = 0.0
    running_loss_g = 0.0
    
    
    # On coupe le calcul des gradients pour économiser de la mémoire
    with torch.no_grad():
        for idx, (real, _) in enumerate(train):
            if idx > 10: break # On regarde juste un petit échantillon (10 batchs)
            
            real = real.to(device)
            batch_size = real.shape[0]
            
            # On génère du bruit et on demande au générateur d'en faire une image
            noise = generate_noise(batch_size, noise_dim)
            fake = gen(noise)
            
            # --- Calcul rapide des scores (juste pour info) ---
            
            # Score sur les vraies images
            dreal = disc(real).view(-1)
            dloss_real = criterion(dreal, torch.ones_like(dreal))
            
            # Score sur les fausses images
            dfake = disc(fake).view(-1)
            dloss_fake = criterion(dfake, torch.zeros_like(dfake))
            
            dloss = (dloss_fake + dloss_real) / 2
            
            # Score du générateur
            gout = disc(fake).view(-1)
            gloss = criterion(gout, torch.ones_like(gout))
            
            # On cumule les résultats
            running_loss_d += dloss.item()
            running_loss_g += gloss.item()
            
            
            
    # Calcul des moyennes
    avg_loss_d = running_loss_d / 11
    avg_loss_g = running_loss_g / 11
    
    
    # On enregistre le point de départ
    epoch_losses_d.append(avg_loss_d)
    epoch_losses_g.append(avg_loss_g)
    
    print(f"  > Init Loss : Discriminateur={avg_loss_d:.4f} | Générateur={avg_loss_g:.4f}")
    

    # --- Affichage visuel ---
    with torch.no_grad():
        random_noise = generate_noise(64, noise_dim) 
        fake_display = gen(random_noise)
        # Normalisation pour l'affichage correct
        grid = make_grid(fake_display, normalize=True, value_range=(-1, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title("Epoch 0 (État Initial - Bruit)")
        plt.imshow(grid_np, cmap='gray')
        plt.show()

    # On remet en mode entraînement 
    disc.train()
    gen.train()

# Exécution du test
evaluate_initial_state()

print('Lancement du DCGAN sur MNIST...')

print("Démarrage de l'entraînement du DCGAN...")

# AJOUT : Initialisation des listes pour stocker l'historique
fid_history = []
is_history = []
mssim_history = []
precision_history = []
recall_history = []

for epoch in range(epochs):
    start_epoch = time.time()
    
    # Remise à zéro des compteurs pour l'époque
    running_loss_d = 0.0
    running_loss_g = 0.0

    # On parcourt le dataset batch par batch
    for idx, (real, _) in enumerate(train):
        
        # Préparation des données
        real = real.to(device) 
        batch_size = real.shape[0]

        # On crée le bruit qui servira de base aux fausses images
        noise = generate_noise(batch_size, noise_dim)
        fake = gen(noise)

        # -------------------------------------
        # 1. Entraînement du Discriminateur
        # -------------------------------------
        with autocast():
            # A. Sur les vraies images
            dreal = disc(real).view(-1)
            
            # Astuce "Label Smoothing" : on vise 0.9 au lieu de 1.0 pour stabiliser
            dloss_real = criterion(dreal, torch.ones_like(dreal) * 0.9)
            
            # B. Sur les fausses images
            dfake = disc(fake.detach()).view(-1)
            dloss_fake = criterion(dfake, torch.zeros_like(dfake))
            
            # Moyenne des deux erreurs
            dloss = (dloss_fake + dloss_real) / 2

        # Mise à jour des poids du Discriminateur
        disc.zero_grad()
        dscaler.scale(dloss).backward()
        dscaler.step(optim_disc)
        dscaler.update()

        # -------------------------------------
        # 2. Entraînement du Générateur
        # -------------------------------------
        with autocast():
            # On repasse les fausses images dans le discriminateur (mis à jour)
            gout = disc(fake).view(-1)
            # Le but du générateur : que le discriminateur dise "Vrai" (1.0)
            gloss = criterion(gout, torch.ones_like(gout))

        # Mise à jour des poids du Générateur
        gen.zero_grad()
        gscaler.scale(gloss).backward()
        gscaler.step(optim_gen)
        gscaler.update()

        # Suivi statistique
        running_loss_d += dloss.item()
        running_loss_g += gloss.item()
        
    
    # --- Fin de l'époque ---
    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    
    # Calcul des moyennes annuelles
    avg_loss_d = running_loss_d / len(train)
    avg_loss_g = running_loss_g / len(train)
    epoch_losses_d.append(avg_loss_d)
    epoch_losses_g.append(avg_loss_g)
    
    
    print(f'--------------------------------------------------')
    print(f'Epoch [{epoch+1}/{epochs}] | Durée: {epoch_time:.2f}s')
    print(f'  > Loss Moyenne : Discriminateur={avg_loss_d:.4f} | Générateur={avg_loss_g:.4f}')
    

    # ---  CALCUL DES MÉTRIQUES (FID, IS, MS-SSIM, PR) ---
    print("  > Calcul des métriques (FID, IS, MS-SSIM, PR)...")
    with torch.no_grad():
        # On prend juste un petit échantillon (15 batches) pour ne pas perdre trop de temps dans le calcul des métriques
        num_batches_metrics = 15
        iter_real = iter(train)
        
        # AJOUT : Listes pour accumuler les features pour PR
        all_real_feats = []
        all_fake_feats = []

        for _ in range(num_batches_metrics):
            try:
                real_batch, _ = next(iter_real)
            except StopIteration:
                iter_real = iter(train)
                real_batch, _ = next(iter_real)
            
            real_batch = real_batch.to(device)
            
            # # Génération d'un lot de fausses images (lot de test)
            noise_metrics = generate_noise(real_batch.shape[0], noise_dim)
            fake_batch = gen(noise_metrics)
            
            # --- 1. PRÉPARATION FID & IS (Uint8 RGB) ---
            # Conversion en format "Image 8-bits" pour le calcul FID/IS
            real_uint8 = ((real_batch.repeat(1, 3, 1, 1) + 1) / 2 * 255).byte()
            fake_uint8 = ((fake_batch.repeat(1, 3, 1, 1) + 1) / 2 * 255).byte()
            
            # Mise à jour des métriques
            fid.update(real_uint8, real=True)
            fid.update(fake_uint8, real=False)
            inception.update(fake_uint8)
            
            # --- 2. PRÉPARATION MS-SSIM (Float 0-1 + RESIZE 256x256) ---
            # DCGAN a déjà des images (N, 1, 28, 28), pas besoin de reshape
            real_float01 = (real_batch + 1) / 2
            fake_float01 = (fake_batch + 1) / 2
            
            # Interpolation à 256x256 pour satisfaire les 5 échelles du MS-SSIM
            real_256 = torch.nn.functional.interpolate(real_float01, size=(256, 256), mode='bilinear', align_corners=False)
            fake_256 = torch.nn.functional.interpolate(fake_float01, size=(256, 256), mode='bilinear', align_corners=False)
            
            ms_ssim.update(fake_256, real_256)

            # --- 3. PRÉPARATION PRECISION/RECALL (Features Inception 299x299) ---
            # On resize en 299x299 pour Inception V3 et on passe en 3 canaux RGB
            real_rgb = real_batch.repeat(1, 3, 1, 1)
            fake_rgb = fake_batch.repeat(1, 3, 1, 1)
            
            real_rgb_float = torch.nn.functional.interpolate(real_rgb, size=(299, 299), mode='bilinear', align_corners=False)
            fake_rgb_float = torch.nn.functional.interpolate(fake_rgb, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Extraction des features
            all_real_feats.append(pr_inception(real_rgb_float))
            all_fake_feats.append(pr_inception(fake_rgb_float))

        # Calcul final des scores pour cette époque
        fid_score = fid.compute()
        is_mean, is_std = inception.compute()
        ms_ssim_score = ms_ssim.compute()

        # Calcul PR sur l'ensemble accumulé
        real_feats_tensor = torch.cat(all_real_feats, dim=0)
        fake_feats_tensor = torch.cat(all_fake_feats, dim=0)
        precision, recall = compute_precision_recall(real_feats_tensor, fake_feats_tensor)
        
        print(f"  >  FID      : {fid_score.item():.4f} (Plus bas = Mieux)")
        print(f"  >  IS       : {is_mean.item():.4f} (Plus haut = Mieux)")
        print(f"  >  MS-SSIM  : {ms_ssim_score.item():.4f} (Plus bas = Mieux)")
        print(f"  >  Precision: {precision:.4f} (Plus haut = Mieux)")
        print(f"  >  Recall   : {recall:.4f} (Plus haut = Mieux)")
        
        # AJOUT : Stockage dans les listes
        fid_history.append(fid_score.item())
        is_history.append(is_mean.item())
        mssim_history.append(ms_ssim_score.item())
        precision_history.append(precision)
        recall_history.append(recall)

        # On remet à zéro pour la prochaine époque
        fid.reset()
        inception.reset()
        ms_ssim.reset()

    # --- Image témoin (Generée par le GAN)---
    with torch.no_grad():
        random_noise = generate_noise(64, noise_dim) 
        fake_display = gen(random_noise)
        grid = make_grid(fake_display, normalize=True, value_range=(-1, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title(f"Image Epoch {epoch+1}/{epochs}")
        plt.imshow(grid_np, cmap='gray')
      # On définit l'axe des X (les époques)
epochs_range = range(0, len(fid_history))

# ==========================================
# FIGURE 1 : FID & IS
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))
plt.title("Évolution des scores : FID & IS (MNIST)", fontsize=14)
ax1.set_xlabel("Époques", fontsize=12)
ax1.grid(True, alpha=0.3)

color_fid = 'tab:red'
ax1.set_ylabel('FID', color=color_fid, fontweight='bold', fontsize=12)
line1 = ax1.plot(epochs_range, fid_history, color=color_fid, label='FID', linewidth=3)
ax1.tick_params(axis='y', labelcolor=color_fid)

ax2 = ax1.twinx()  
color_is = 'tab:blue'
ax2.set_ylabel('IS', color=color_is, fontweight='bold', fontsize=12)
line2 = ax2.plot(epochs_range, is_history, color=color_is, label='Inception Score', linewidth=3)
ax2.tick_params(axis='y', labelcolor=color_is)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=11)

fig.tight_layout()
plt.show()

# ==========================================
# FIGURE 2 : Precision & Recall
# ==========================================
plt.figure(figsize=(10, 6))
plt.title("Évolution des scores : Precision & Recall (MNIST)")
plt.plot(epochs_range, precision_history, label='Precision (Qualité des images)', color='purple', linewidth=2.5)
plt.plot(epochs_range, recall_history, label='Recall (Diversité des images)', color='orange', linewidth=2.5)

plt.xlabel("Époques")
plt.ylabel("Valeurs")
plt.ylim(0, 1.0) 
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 

# ==========================================
# FIGURE 3 : MS-SSIM (Diversité)
# ==========================================
plt.figure(figsize=(10, 6))
plt.title("Analyse du Mode Collapse MS-SSIM (MNIST)")
plt.plot(epochs_range, mssim_history, label='MS-SSIM', color='green', linewidth=2.5)

plt.xlabel("Époques")
plt.ylabel("Valeurs")
plt.axhline(y=0.9, color='red', linestyle=':', label='Seuil critique (Mode Collapse)') # Ligne d'alerte
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 

plt.figure(figsize=(10, 5))
plt.title("Courbe de Perte (DCGAN MNIST)")
# On trace depuis l'époque 0 (état initial) jusqu'à la fin
plt.plot(range(0, epochs+1), epoch_losses_d, label="Discriminateur", linewidth=2)
plt.plot(range(0, epochs+1), epoch_losses_g, label="Générateur", linewidth=2)
plt.xlabel("Époques")
plt.ylabel("Valeur des Pertes(Loss)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
        plt.show()

print('Entraînement terminé avec succès !')
