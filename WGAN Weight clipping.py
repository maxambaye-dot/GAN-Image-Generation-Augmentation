import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Évite les conflits de bibliothèques

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import warnings
import time

#Pour le WGAN, on préfère RMSprop à Adam pour la stabilité
from torch.optim import RMSprop 

# Imports pour les métriques de qualité 
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchvision.models import inception_v3

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Paramètres WGAN ---
lr = 0.00005          # Learning Rate très bas (recommandée)
WEIGHT_CLIP = 0.01    # Seuil de coupure des poids (Clipping)
CRITIC_ITERATIONS = 5 # Le critique s'entraîne 5 fois plus que le générateur

# --- Paramètres Classiques ---
noise_dim = 100       
batch_size = 64
epochs = 250       
channels_img = 1      # 1 canal pour MNIST (Noir & Blanc)
features_d = 64
features_g = 64

print(f"Entraînement configuré sur : {device}")

# Transformation : Normalisation standard pour les GANs
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

# Téléchargement et chargement des données
data = datasets.MNIST(root=r'dataset/', download=True, transform=transform)
train = DataLoader(data, batch_size=batch_size, shuffle=True)

print("Données MNIST chargées !")

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: Image 28x28 -> 14x14
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 14x14 -> 7x7
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 2), 
            nn.LeakyReLU(0.2),
            
            # 7x7 -> 4x4
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),
            
            # Sortie : Score brut (Pas de Sigmoid pour le WGAN !)
            nn.Flatten(),
            nn.Linear(features_d * 4 * 4 * 4, 1) 
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: Vecteur de bruit -> 3x3
            nn.ConvTranspose2d(channels_noise, features_g * 4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),
            
            # 3x3 -> 7x7
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g),
            nn.ReLU(),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Sortie image [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

# Fonction pour initialiser proprement les poids (Loi normale)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

      # Instanciation sur le GPU/CPU
disc = Discriminator(channels_img, features_d).to(device)
gen = Generator(noise_dim, channels_img, features_g).to(device)

# Application de l'initialisation des poids
initialize_weights(disc)
initialize_weights(gen)

# Optimiseurs RMSprop 
optim_disc = RMSprop(disc.parameters(), lr=lr)
optim_gen = RMSprop(gen.parameters(), lr=lr)

# Outils de métriques
print("Préparation des métriques (FID, IS, MS-SSIM, PR)...")
fid = FrechetInceptionDistance(feature=2048).to(device)
inception = InceptionScore(feature='logits_unbiased', normalize=False).to(device)

#  Initialisation MS-SSIM
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

#  Modèle Inception pour extraction de features PR (chargé une seule fois)
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

# Fonction utilitaire pour le bruit
def generate_noise(b, n):
    return torch.randn(b, n, 1, 1).to(device)

# Listes pour stocker l'historique
epoch_losses_d = []
epoch_losses_g = []
epoch_score_real = [] # Pour suivre le score moyen donné aux vraies images
epoch_score_fake = [] # Pour suivre le score moyen donné aux fausses images
#fixed_noise = generate_noise(64, noise_dim)

def evaluate_initial_state():
    print("Evaluation de l'état initial (Epoch 0)...")
    
    # On met les réseaux en mode "test" (fige les stats de BatchNorm)
    disc.eval()
    gen.eval()
    
    # Compteurs pour les pertes (Loss)
    running_loss_d = 0.0
    running_loss_g = 0.0
    
    # Compteurs pour les Scores bruts 
    # sr = Score Real (avis du critique sur les vraies images)
    # sf = Score Fake (avis du critique sur les fausses images)
    running_sr = 0.0
    running_sf = 0.0
    
    with torch.no_grad(): # Pas de calcul de gradient ici, on observe juste
        for idx, (real, _) in enumerate(train):
            if idx > 10: break 
            
            real = real.to(device)
            batch_size = real.shape[0]
            
            # Génération d'images aléatoires
            noise = generate_noise(batch_size, noise_dim)
            fake = gen(noise)
            
            # --- Le Critique donne son avis ---
            # Ce ne sont pas des probabilités (0-1) mais des scores réels (ex: -5, +12...)
            real_pred = disc(real).view(-1)
            fake_pred = disc(fake).view(-1)
            
            # --- Calcul de la Loss WGAN ---
            # Le Critique veut maximiser la différence (Real - Fake).
            # En code, on minimise l'opposé : -(Real - Fake)
            d_loss = -(torch.mean(real_pred) - torch.mean(fake_pred))
            g_loss = -torch.mean(fake_pred)
            
            # Accumulation des résultats
            running_loss_d += d_loss.item()
            running_loss_g += g_loss.item()
            
            # On stocke les scores moyens pour voir si le critique arrive à les séparer
            running_sr += torch.mean(real_pred).item()
            running_sf += torch.mean(fake_pred).item()
            
    # Calcul des moyennes sur les 10 batchs
    avg_loss_d = running_loss_d / 11
    avg_loss_g = running_loss_g / 11
    avg_sr = running_sr / 11
    avg_sf = running_sf / 11
    
    # On sauvegarde les pertes dans l'historique global
    epoch_losses_d.append(avg_loss_d)
    epoch_losses_g.append(avg_loss_g)
    
    print(f"  > Init Loss : D={avg_loss_d:.4f} | G={avg_loss_g:.4f}")
    print(f"  > Init Scores : Real={avg_sr:.2f} | Fake={avg_sf:.2f}")

    # --- Visualisation ---
    with torch.no_grad():
        random_noise = generate_noise(64, noise_dim) 
        fake_display = gen(random_noise)
        grid = make_grid(fake_display, normalize=True, value_range=(-1, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title("Epoch 0 (État Initial - Bruit)")
        plt.imshow(grid_np, cmap='gray')
        plt.show()

    # On remet le mode entraînement
    disc.train()
    gen.train()

# Lancement du test
evaluate_initial_state()


print('Lancement du WGAN (Weight Clipping) sur MNIST...')

# Initialisation des listes pour stocker 
fid_history = []
is_history = []
mssim_history = []
precision_history = []
recall_history = []

for epoch in range(epochs):
    start_epoch = time.time()
    
    # Réinitialisation des compteurs pour l'époque
    running_loss_d = 0.0
    running_loss_g = 0.0
    running_sr = 0.0
    running_sf = 0.0
    
    iter_loader = iter(train)
    num_batches = len(train)

    for i in range(num_batches):
        
        # On récupère un batch d'images réelles
        try:
            real, _ = next(iter_loader)
        except StopIteration:
            # Si on est au bout du dataset, on recommence
            iter_loader = iter(train)
            real, _ = next(iter_loader)
        
        real = real.to(device)
        batch_size = real.shape[0]

        # ---------------------
        # 1. Train Critique (5 itérations pour 1 itération du Générateur)
        # ---------------------

        for _ in range(CRITIC_ITERATIONS):
            noise = generate_noise(batch_size, noise_dim)
            fake = gen(noise) # Le générateur crée une image
            
            # Le critique donne son avis (Score brut, pas de probabilité)
            critic_real = disc(real).view(-1)
            critic_fake = disc(fake).view(-1)
            
            # --- Loss WGAN Simple ---
            # Le but du critique : Maximiser (Score_Vrai - Score_Faux)
            # En PyTorch on minimise, donc on prend l'opposé : -(Score_Vrai - Score_Faux)
            # C'est une approximation de la "Distance de Wasserstein"
            loss_disc = -(torch.mean(critic_real) - torch.mean(critic_fake))
            
            optim_disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            optim_disc.step()

            # --- WEIGHT CLIPPING ---
            # Pour satisfaire la contrainte mathématique (Lipschitz), on force les poids
            # à rester dans une petite boîte [-0.01, 0.01]. C'est brutale mais efficace.
            for p in disc.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # ---------------------
        # 2. Train Generator (1 itération)
        # ---------------------
        # Le générateur veut que le critique donne un score élevé à ses fausses images
        # On reprend le 'fake' de la dernière itération de la boucle ci-dessus
        gen_fake = disc(fake).view(-1)
        
        # On veut Maximiser le score du fake => Minimiser -Score_Fake
        loss_gen = -torch.mean(gen_fake)
        
        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        # Logs et accumulation des scores pour l'affichage
        # On divise par CRITIC_ITERATIONS pour que la moyenne reste cohérente (puisqu'on boucle 5 fois)
        running_loss_d += loss_disc.item() / CRITIC_ITERATIONS
        running_loss_g += loss_gen.item()
        
        # On ne stocke que la dernière valeur des scores pour simplifier
        running_sr += torch.mean(critic_real).item()
        running_sf += torch.mean(critic_fake).item()
    
    # --- Fin de l'époque ---
    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    
    # Calcul des moyennes
    avg_loss_d = running_loss_d / num_batches
    avg_loss_g = running_loss_g / num_batches
    
    # On divise par CRITIC_ITERATIONS car on a sommé 5 fois plus de valeurs pour le critique
    avg_sr = running_sr / (num_batches * CRITIC_ITERATIONS)
    avg_sf = running_sf / (num_batches * CRITIC_ITERATIONS)
    
    epoch_losses_d.append(avg_loss_d)
    epoch_losses_g.append(avg_loss_g)

    print(f'--------------------------------------------------')
    print(f'Epoch [{epoch+1}/{epochs}] | Durée: {epoch_time:.2f}s') 
    print(f'  > Loss : Discriminateur={avg_loss_d:.4f} | Générateur={avg_loss_g:.4f}')
    print(f'  > Scores : Real={avg_sr:.4f} | Fake={avg_sf:.4f}')

    # --- Calcul des métriques de qualité (FID / IS / MS-SSIM / PR) ---
    print("  > Calcul des métriques...")
    with torch.no_grad():
        num_batches_metrics = 5 # On teste sur un petit échantillon pour aller vite
        temp_iter = iter(train)
        
        # AJOUT : Listes pour accumuler les features pour PR
        all_real_feats = []
        all_fake_feats = []

        for _ in range(num_batches_metrics):
            try:
                real_batch, _ = next(temp_iter)
            except StopIteration:
                temp_iter = iter(train)
                real_batch, _ = next(temp_iter)
            
            real_batch = real_batch.to(device)
            noise_metrics = generate_noise(real_batch.shape[0], noise_dim)
            fake_batch = gen(noise_metrics)
            
            # Conversion en format Image (uint8) pour les métriques
            real_uint8 = ((real_batch.repeat(1, 3, 1, 1) + 1) / 2 * 255).byte()
            fake_uint8 = ((fake_batch.repeat(1, 3, 1, 1) + 1) / 2 * 255).byte()
            
            fid.update(real_uint8, real=True)
            fid.update(fake_uint8, real=False)
            inception.update(fake_uint8)
            
            # --- 2. PRÉPARATION MS-SSIM (Float 0-1 + RESIZE 256x256) ---
            real_float01 = (real_batch + 1) / 2
            fake_float01 = (fake_batch + 1) / 2
            
            real_256 = torch.nn.functional.interpolate(real_float01, size=(256, 256), mode='bilinear', align_corners=False)
            fake_256 = torch.nn.functional.interpolate(fake_float01, size=(256, 256), mode='bilinear', align_corners=False)
            
            ms_ssim.update(fake_256, real_256)

            # --- 3. PRÉPARATION PRECISION/RECALL (Features Inception 299x299) ---
            real_rgb = real_batch.repeat(1, 3, 1, 1)
            fake_rgb = fake_batch.repeat(1, 3, 1, 1)
            
            real_rgb_float = torch.nn.functional.interpolate(real_rgb, size=(299, 299), mode='bilinear', align_corners=False)
            fake_rgb_float = torch.nn.functional.interpolate(fake_rgb, size=(299, 299), mode='bilinear', align_corners=False)
            
            all_real_feats.append(pr_inception(real_rgb_float))
            all_fake_feats.append(pr_inception(fake_rgb_float))
            
        # Calcul final des scores
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

        # Reset des métriques pour le prochain tour
        fid.reset()
        inception.reset()
        ms_ssim.reset()

    # --- Visualisation dans le Notebook ---
    with torch.no_grad():
        random_noise = generate_noise(64, noise_dim) 
        fake_display = gen(random_noise)
        grid = make_grid(fake_display, normalize=True, value_range=(-1, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title(f"Images Epoch {epoch+1}/{epochs}")
        plt.imshow(grid_np, cmap='gray')
        plt.show()

print('Entraînement terminé...')

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
plt.title("Analyse du Mode Collapse MS-SSIM (MNIST) ")
plt.plot(epochs_range, mssim_history, label='MS-SSIM', color='green', linewidth=2.5)

plt.xlabel("Époques")
plt.ylabel("Valeurs")
plt.axhline(y=0.9, color='red', linestyle=':', label='Seuil critique (Mode Collapse)') # Ligne d'alerte
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 

plt.figure(figsize=(10, 5))
plt.title("Courbes de Loss WGAN (Clipping) MNIST")
plt.plot(range(0, epochs+1), epoch_losses_d, label="Discriminateur (D)")
plt.plot(range(0, epochs+1), epoch_losses_g, label="Générateur (G)")
plt.xlabel("Époques")
plt.ylabel("Loss Wasserstein")
plt.legend()
plt.grid(True)
plt.show()
