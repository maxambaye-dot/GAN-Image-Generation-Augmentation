import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
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

# Configuration
lr = 0.0002          # Taux d'apprentissage (Learning Rate)
noise_dim = 100      # Dimension du vecteur latent (bruit en entrée du générateur)
in_size = 784        # Taille de l'image aplatie (28x28 = 784)
batch_size = 64      # Nombre d'images par lot
epochs = 250         # Nombre de cycles d'entraînement

print(f"Entraînement configuré sur : {device}")

# Transformation des images : Conversion en tenseur et Normalisation [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Téléchargement et chargement des données
print("Chargement du dataset MNIST...")
dataset = datasets.MNIST(root='dataset/', download=True, transform=transform)
train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Données chargées avec succès.")

class Discriminator(nn.Module):
    """ Réseau chargé de différencier les images réelles des images générées. """
    def __init__(self, in_size):
        super().__init__()
        self.dmodel = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),       # Dropout pour éviter le sur-apprentissage
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)      # Sortie unique (Vrai/Faux)
        )

    def forward(self, x):
        return self.dmodel(x)

class Generator(nn.Module):
    """ Réseau chargé de générer des images à partir d'un vecteur de bruit. """
    def __init__(self, noise_dim, in_size):
        super().__init__()
        self.gmodel = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, in_size),
            nn.Tanh()             # Activation Tanh pour une sortie entre -1 et 1
        )
            
    def forward(self, x):
        return self.gmodel(x)

  # Instanciation des modèles
disc = Discriminator(in_size).to(device)
gen = Generator(noise_dim, in_size).to(device)

# Fonction utilitaire pour générer du bruit
def generate_noise(batch_len, dim):
    return torch.randn((batch_len, dim)).to(device)

# AJOUT : Fonctions utilitaires pour le calcul du Precision & Recall (PR)
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

# Optimiseurs et Fonction de Coût
optim_disc = Adam(disc.parameters(), lr=lr)
optim_gen = Adam(gen.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

# Scalers pour la précision mixte (stabilité et vitesse)
dscaler = GradScaler()
gscaler = GradScaler()

# Initialisation des métriques 
fid = FrechetInceptionDistance(feature=2048).to(device)
inception = InceptionScore(feature='logits_unbiased', normalize=False).to(device)

# AJOUT : Initialisation MS-SSIM
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# AJOUT : Modèle Inception pour extraction de features PR (chargé une seule fois)
pr_inception = inception_v3(pretrained=True, transform_input=False).to(device)
pr_inception.fc = nn.Identity() # On garde les features, pas la classification
pr_inception.eval()

# Bruit fixe pour visualiser la progression du générateur toujours sur la même base
#fixed_noise = generate_noise(64, noise_dim)

# Listes pour stocker les pertes (Loss)
epoch_losses_d = []
epoch_losses_g = []

print("Modèles et optimiseurs initialisés.")

def evaluate_initial_state():
    print("Evaluation de l'état initial (Epoch 0)...")
    
    # On met les réseaux en mode "examen" (pas d'apprentissage ici, juste de l'observation)
    disc.eval()
    gen.eval()
    
    running_loss_d = 0.0
    running_loss_g = 0.0
    
    with torch.no_grad(): # Pas besoin de calculer les gradients, on regarde juste le résultat
        for idx, (real, _) in enumerate(train):
            if idx > 10: break # On regarde juste un petit échantillon (10 batchs), inutile de tout faire
            
            # Mise à plat de l'image pour qu'elle rentre dans notre réseau linéaire (28x28 -> 784 pixels)
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]
            
            # On demande au générateur de créer des images à partir de bruit
            noise = generate_noise(batch_size, noise_dim)
            fake = gen(noise)
            
            # --- Calcul des scores (Loss) ---
            
            # 1. Comment le discriminateur juge les VRAIES images ?
            dreal = disc(real).view(-1)
            dloss_real = criterion(dreal, torch.ones_like(dreal)) # Il devrait dire "1" (Vrai)
            
            # 2. Comment le discriminateur juge les FAUSSES images ?
            dfake = disc(fake).view(-1)
            dloss_fake = criterion(dfake, torch.zeros_like(dfake)) # Il devrait dire "0" (Faux)
            
            # Moyenne des deux erreurs
            dloss = (dloss_fake + dloss_real) / 2
            
            # 3. Score du générateur (réussit-il à tromper le discriminateur ?)
            gout = disc(fake).view(-1)
            gloss = criterion(gout, torch.ones_like(gout)) # Le générateur veut que le disc dise "1"
            
            running_loss_d += dloss.item()
            running_loss_g += gloss.item()
            
            

    # On fait la moyenne 
    avg_loss_d = running_loss_d / 11
    avg_loss_g = running_loss_g / 11
    
    # On enregistre ces valeurs de départ pour nos graphiques futurs
    epoch_losses_d.append(avg_loss_d)
    epoch_losses_g.append(avg_loss_g)
    
    print(f"  > Init Loss : Discriminateur={avg_loss_d:.4f} | Générateur={avg_loss_g:.4f}")
    

    # --- Affichage de l'image témoin ---
    with torch.no_grad():
        random_noise = generate_noise(64, noise_dim) 
        fake_display_flat = gen(random_noise)
        fake_display = fake_display_flat.reshape(-1, 1, 28, 28) # On redonne sa forme carrée à l'image
        grid = make_grid(fake_display, normalize=True, value_range=(-1, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title("Epoch 0 (État Initial - Juste du bruit)")
        plt.imshow(grid_np, cmap='gray')
        plt.show()

    # on remet tout le monde en mode entraînement !
    disc.train()
    gen.train()

evaluate_initial_state()

print("Lancement de l'entraînement...")

# --- BOUCLE D'ENTRAÎNEMENT PRINCIPALE ---
print("Démarrage de l'entraînement du GAN-MLP...")

# AJOUT : Initialisation des listes pour stocker l'historique
fid_history = []
is_history = []
mssim_history = []
precision_history = []
recall_history = []

for epoch in range(epochs):
    start_epoch = time.time()
    
    # Initialisation des compteurs pour cette époque
    running_loss_d = 0.0
    running_loss_g = 0.0


    # --- A. PHASE D'APPRENTISSAGE (Batch par Batch) ---
    for idx, (real, _) in enumerate(train):
        
        # Préparation des données réelles
        # On aplatit l'image (28x28 -> 784) car notre modèle est un simple réseau de neurones (MLP)
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Génération du bruit aléatoire (la matière première du générateur)
        noise = generate_noise(batch_size, noise_dim)
        fake = gen(noise)

        # ---------------------------------------
        # 1. Entraînement du Discriminateur 
        # ---------------------------------------
        with autocast(): # Optimisation automatique pour gagner du temps/mémoire
            # Test sur les images RÉELLES
            dreal = disc(real).view(-1)
            # On veut qu'il prédise 1 (Vrai)
            dloss_real = criterion(dreal, torch.ones_like(dreal))
            
            # Test sur les images FAUSSES (générées)
            # .detach() est crucial ici : on ne veut pas modifier le générateur pour l'instant
            dfake = disc(fake.detach()).view(-1)
            # On veut qu'il prédise 0 (Faux)
            dloss_fake = criterion(dfake, torch.zeros_like(dfake))
            
            # La perte totale du discriminateur est la moyenne des deux erreurs
            dloss = (dloss_fake + dloss_real) / 2

        # Mise à jour des poids du discriminateur
        disc.zero_grad()
        dscaler.scale(dloss).backward()
        dscaler.step(optim_disc)
        dscaler.update()

        # ---------------------------------------
        # 2. Entraînement du Générateur 
        # ---------------------------------------
        with autocast():
            # On repasse les images fausses dans le discriminateur (cette fois sans detach)
            gout = disc(fake).view(-1)
            # Le but du générateur : tromper le disc pour qu'il dise 1 (Vrai)
            gloss = criterion(gout, torch.ones_like(gout))

        # Mise à jour des poids du générateur
        gen.zero_grad()
        gscaler.scale(gloss).backward()
        gscaler.step(optim_gen)
        gscaler.update()

        # Suivi des performances
        running_loss_d += dloss.item()
        running_loss_g += gloss.item()
        
    # Fin de l'époque : calcul du temps écoulé
    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    
    # Calcul des moyennes annuelles
    avg_loss_d = running_loss_d / len(train)
    avg_loss_g = running_loss_g / len(train)
    
    # On stocke l'historique pour les graphiques finaux
    epoch_losses_d.append(avg_loss_d)
    epoch_losses_g.append(avg_loss_g)
    
    
    
    print(f'--------------------------------------------------')
    print(f'Epoch [{epoch+1}/{epochs}] | Durée: {epoch_time:.2f}s')
    print(f'  > Loss Moyenne : Discriminateur={avg_loss_d:.4f} | Générateur={avg_loss_g:.4f}')
    
    
    # --- B. CALCUL DES MÉTRIQUES (FID, IS, PR, MS-SSIM) ---
    print("  > Calcul des métriques (FID, IS, PR, MS-SSIM)...")
    
    # On se met en mode évaluation (pas de gradients)
    with torch.no_grad():
        # On ne calcule pas sur tout le dataset pour gagner du temps (juste 15 batches)
        num_batches_metrics = 15
        iter_real = iter(train)
        
        # Listes pour accumuler les features pour PR
        all_real_feats = []
        all_fake_feats = []

        for _ in range(num_batches_metrics):
            try:
                real_batch, _ = next(iter_real)
            except StopIteration:
                iter_real = iter(train)
                real_batch, _ = next(iter_real)
            
            real_batch = real_batch.to(device)

            # Génération d'un lot de fausses images
            noise_metrics = generate_noise(real_batch.shape[0], noise_dim)
            fake_batch_flat = gen(noise_metrics)
            fake_batch = fake_batch_flat.reshape(-1, 1, 28, 28)
            
            # --- 1. PRÉPARATION FID & IS (Uint8 RGB) ---
            real_rgb = real_batch.repeat(1, 3, 1, 1)
            real_uint8 = ((real_rgb + 1) / 2 * 255).byte()
            
            fake_rgb = fake_batch.repeat(1, 3, 1, 1)
            fake_uint8 = ((fake_rgb + 1) / 2 * 255).byte()
            
            fid.update(real_uint8, real=True)
            fid.update(fake_uint8, real=False)
            inception.update(fake_uint8)

            # --- 2. PRÉPARATION MS-SSIM (Float 0-1 + RESIZE 256x256) ---
            # CORRECTION ICI : On resize en 256x256 (Valeur sûre > 160)
            real_float01 = (real_batch + 1) / 2
            fake_float01 = (fake_batch + 1) / 2
            
            # Interpolation à 256x256 pour satisfaire les 5 échelles du MS-SSIM
            real_256 = torch.nn.functional.interpolate(real_float01, size=(256, 256), mode='bilinear', align_corners=False)
            fake_256 = torch.nn.functional.interpolate(fake_float01, size=(256, 256), mode='bilinear', align_corners=False)
            
            ms_ssim.update(fake_256, real_256)

            # --- 3. PRÉPARATION PRECISION/RECALL (Features Inception 299x299) ---
            # On resize en 299x299 pour Inception V3
            real_rgb_float = torch.nn.functional.interpolate(real_rgb, size=(299, 299), mode='bilinear', align_corners=False)
            fake_rgb_float = torch.nn.functional.interpolate(fake_rgb, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Extraction des features
            all_real_feats.append(pr_inception(real_rgb_float))
            all_fake_feats.append(pr_inception(fake_rgb_float))
            
        # --- CALCUL FINAL DES SCORES ---
        fid_score = fid.compute()
        is_mean, is_std = inception.compute()
        ms_ssim_score = ms_ssim.compute()
        
        # Calcul PR sur l'ensemble accumulé
        real_feats_tensor = torch.cat(all_real_feats, dim=0)
        fake_feats_tensor = torch.cat(all_fake_feats, dim=0)
        precision, recall = compute_precision_recall(real_feats_tensor, fake_feats_tensor)
        
        print(f"  >  FID      : {fid_score.item():.4f} (Plus Bas= Mieux)")
        print(f"  >  IS       : {is_mean.item():.4f} (Plus Haut = Mieux)")
        print(f"  >  MS-SSIM  : {ms_ssim_score.item():.4f} (Plus Bas = Mieux)")
        print(f"  >  Precision: {precision:.4f} (Plus Haut = Mieux)")
        print(f"  >  Recall   : {recall:.4f} (Plus Haut = Mieux)")
        
        # AJOUT : Stockage dans les listes
        fid_history.append(fid_score.item())
        is_history.append(is_mean.item())
        mssim_history.append(ms_ssim_score.item())
        precision_history.append(precision)
        recall_history.append(recall)

        # Reset des métriques pour la prochaine époque
        fid.reset()
        inception.reset()
        ms_ssim.reset()

    # --- C. VISUALISATION INTERMÉDIAIRE ---
    
    with torch.no_grad():
        random_noise = generate_noise(64, noise_dim) 
        fake_display_flat = gen(random_noise)
        
        fake_display = fake_display_flat.reshape(-1, 1, 28, 28)
        
        grid = make_grid(fake_display, normalize=True, value_range=(-1, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title(f"Image Epoch {epoch+1}/{epochs}")
        plt.imshow(grid_np, cmap='gray')
        plt.show()

print('Entraînement terminé avec succès !')

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

# --- Affichage des courbes d'apprentissage ---
plt.figure(figsize=(10, 5))
plt.title("Courbe de Perte (GAN-MLP MNIST)")
# On trace depuis l'époque 0 (état initial) jusqu'à la fin
plt.plot(range(0, epochs+1), epoch_losses_d, label="Discriminateur", linewidth=2)
plt.plot(range(0, epochs+1), epoch_losses_g, label="Générateur", linewidth=2)
plt.xlabel("Époques")
plt.ylabel("Valeur des Perte(Loss)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
