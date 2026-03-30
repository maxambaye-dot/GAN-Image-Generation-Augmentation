# --- 1. IMPORTS ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam
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

lr = 0.0001           # Taux d'apprentissage (Learning rate) 
noise_dim = 100       # Taille du vecteur de bruit z
batch_size = 64
epochs = 250         
channels_img = 1      # MNIST (Noir & Blanc)
features_d = 64
features_g = 64
LAMBDA_GP = 10        # Pénalité de gradient
CRITIC_ITERATIONS = 5 # Ratio Critique/Générateur

# --- Paramètres Conditionnels ---
NUM_CLASSES = 10      # 10 classes (chiffres 0-9)
GEN_EMBEDDING = 100   # Taille du vecteur représentant la classe pour le Générateur
IMG_SIZE = 28         # Taille de l'image 

print(f"Entraînement configuré sur : {device}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.0001           # Taux d'apprentissage (Learning rate) 
noise_dim = 100       # Taille du vecteur de bruit z
batch_size = 64
epochs = 250         
channels_img = 1      # MNIST (Noir & Blanc)
features_d = 64
features_g = 64
LAMBDA_GP = 10        # Pénalité de gradient
CRITIC_ITERATIONS = 5 # Ratio Critique/Générateur

# --- Paramètres Conditionnels ---
NUM_CLASSES = 10      # 10 classes (chiffres 0-9)
GEN_EMBEDDING = 100   # Taille du vecteur représentant la classe pour le Générateur
IMG_SIZE = 28         # Taille de l'image 

print(f"Entraînement configuré sur : {device}")

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        
        # --- Embedding des Labels ---
        # On crée une couche qui transforme un chiffre (ex: 3) en une image 28x28
        self.embed = nn.Embedding(num_classes, img_size * img_size)
        
        # Input: N x (channels_img + 1) x 28 x 28
        # Notez le "+ 1" : c'est le canal ajouté pour l'étiquette
        self.disc = nn.Sequential(
            # Bloc 1
            nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # Bloc 2
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(features_d * 2, affine=True), # InstanceNorm pour WGAN-GP
            nn.LeakyReLU(0.2),
            
            # Bloc 3
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features_d * 4, affine=True),
            nn.LeakyReLU(0.2),
            
            # Aplatissement
            nn.Flatten(),
            
            # Sortie (Score brut)
            nn.Linear(features_d * 4 * 4 * 4, 1) 
        )

    def forward(self, x, labels):
        # x: Image (Batch, 1, 28, 28)
        # labels: (Batch,)
        
        # 1. On transforme le label (étiquette) en une grille 28x28
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        
        # 2. On colle l'étiquette sur l'image (Concaténation sur le canal C)
        # L'image passe de 1 canal à 2 canaux (Image + Label)
        x = torch.cat([x, embedding], dim=1) 
        
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, embed_size):
        super(Generator, self).__init__()
        
        # --- Embedding des Labels ---
        # Transforme le chiffre (0-9) en un vecteur de taille 'embed_size'
        self.embed = nn.Embedding(num_classes, embed_size)
        
        self.net = nn.Sequential(
            # Entrée : Bruit + Embedding
            # On fusionne la taille du bruit et celle de l'embedding
            nn.ConvTranspose2d(channels_noise + embed_size, features_g * 4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),
            
            # Bloc 2
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),
            
            # Bloc 3
            nn.ConvTranspose2d(features_g * 2, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g),
            nn.ReLU(),
            
            # Sortie
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh() 
        )

    def forward(self, x, labels):
        # x: bruit (Batch, noise_dim, 1, 1)
        # labels: (Batch,)
        
        # 1. On transforme le label en vecteur et on le met au format (Batch, embed_size, 1, 1)
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        
        # 2. On colle le bruit et l'étiquette ensemble
        x = torch.cat([x, embedding], dim=1)
        
        return self.net(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

      # On initialise avec les paramètres conditionnels
disc = Discriminator(channels_img, features_d, NUM_CLASSES, IMG_SIZE).to(device)
gen = Generator(noise_dim, channels_img, features_g, NUM_CLASSES, GEN_EMBEDDING).to(device)

initialize_weights(disc)
initialize_weights(gen)

print("Initialisation des métriques (FID, IS, MS-SSIM, PR)...")
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

def generate_noise(b, n):
    return torch.randn(b, n, 1, 1).to(device)

# Optimiseurs (Adam avec beta1=0.0 pour WGAN-GP)
optim_disc = Adam(disc.parameters(), lr=lr, betas=(0.0, 0.9))
optim_gen = Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))

# Listes historiques
epoch_losses_d = []
epoch_losses_g = []

#  Listes pour métriques
fid_history = []
is_history = []
mssim_history = []
precision_history = []
recall_history = []

# Gradient Penalty Function 
def gradient_penalty(critic, real, fake, labels, device="cuda"):
    BATCH_SIZE, C, H, W = real.shape
    
    # Création des images mixées
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    # Mixed scores (IMPORTANT : On passe les labels au critique ici aussi !)
    mixed_scores = critic(interpolated_images, labels)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# On crée 100 vecteurs de bruit (10x10)
fixed_noise = generate_noise(100, noise_dim)
fixed_labels = torch.arange(0, 10).repeat_interleave(10).to(device) 

print('Lancement du Conditional WGAN-GP sur MNIST...')

# --- 6. BOUCLE D'ENTRAÎNEMENT ---
for epoch in range(epochs):
    start_epoch = time.time()
    
    running_loss_d = 0.0
    running_loss_g = 0.0
    
    iter_loader = iter(train)
    num_batches = len(train)

    for i in range(num_batches):
        try:
            # On récupère l'image ET son étiquette
            real, labels = next(iter_loader) 
        except StopIteration:
            iter_loader = iter(train)
            real, labels = next(iter_loader)
        
        real = real.to(device)
        labels = labels.to(device) # On envoie les labels sur le GPU
        batch_size = real.shape[0]

        # ---------------------
        # 1. Entraînment Discriminateur
        # ---------------------
        for _ in range(CRITIC_ITERATIONS):
            noise = generate_noise(batch_size, noise_dim)
            
            # Génération conditionnelle : On dit au générateur quel chiffre faire
            fake = gen(noise, labels) 
            
            # Le critique juge en fonction de l'image ET de l'étiquette fournie
            critic_real = disc(real, labels).view(-1) 
            critic_fake = disc(fake, labels).view(-1)
            
            # Calcul du GP avec les labels
            gp = gradient_penalty(disc, real, fake, labels, device=device)
            
            loss_disc = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            
            optim_disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            optim_disc.step()

        # ---------------------
        # 2. Entraînement Générateur
        # ---------------------
        # Le générateur essaie de tromper le critique pour l'étiquette donnée
        gen_fake = disc(fake, labels).view(-1)
        loss_gen = -torch.mean(gen_fake)
        
        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        # On divise par CRITIC_ITERATIONS pour harmoniser l'échelle
        running_loss_d += loss_disc.item() / CRITIC_ITERATIONS
        running_loss_g += loss_gen.item()
    
    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    
    avg_loss_d = running_loss_d / num_batches
    avg_loss_g = running_loss_g / num_batches
    
    epoch_losses_d.append(avg_loss_d)
    epoch_losses_g.append(avg_loss_g)

    print(f'Epoch [{epoch+1}/{epochs}] | Durée: {epoch_time:.2f}s') 
    print(f'Loss : Discrminateur= {avg_loss_d:.4f} | Générateur: {avg_loss_g:.4f}')

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
                real_batch, labels_batch = next(temp_iter)
            except StopIteration:
                temp_iter = iter(train)
                real_batch, labels_batch = next(temp_iter)
            
            real_batch = real_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            noise_metrics = generate_noise(real_batch.shape[0], noise_dim)
            fake_batch = gen(noise_metrics, labels_batch) # Génération avec labels !
            
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

    # --- Visualisation Conditionnelle ---
    # On utilise nos "fixed_labels" pour vérifier que le modèle génère bien 
    # des 0 sur la ligne 0, des 1 sur la ligne 1, etc.
    with torch.no_grad():
        fake_display = gen(fixed_noise, fixed_labels)
        grid = make_grid(fake_display, nrow=10, normalize=True, value_range=(-1, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.title(f"Image Epoch {epoch+1} (Chaque ligne = un chiffre précis)")
        plt.imshow(grid_np, cmap='gray')
        plt.show()

print('Entraînement terminé.')

# On définit l'axe des X (les époques)
epochs_range = range(0, len(fid_history))

# ==========================================
# FIGURE 1 : FID & IS
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))
plt.title("Évolution des scores : FID & IS", fontsize=14)
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
plt.title("Évolution des scores : Precision & Recall")
plt.plot(epochs_range, precision_history, label='Precision (Qualité)', color='purple', linewidth=2.5)
plt.plot(epochs_range, recall_history, label='Recall (Diversité)', color='orange', linewidth=2.5)

plt.xlabel("Époques")
plt.ylabel("Valeurs")
plt.ylim(0, 1.0) 
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 

# ==========================================
# FIGURE 3 : MS-SSIM
# ==========================================
plt.figure(figsize=(10, 6))
plt.title("Analyse du Mode Collapse (MS-SSIM)")
plt.plot(epochs_range, mssim_history, label='MS-SSIM', color='green', linewidth=2.5)

plt.xlabel("Époques")
plt.ylabel("Valeurs")
plt.axhline(y=0.9, color='red', linestyle=':', label='Seuil critique')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Courbes de Loss Conditional WGAN-GP (MNIST) ")
plt.plot(range(0, len(epoch_losses_d)), epoch_losses_d, label="Discriminateur (D)")
plt.plot(range(0, len(epoch_losses_g)), epoch_losses_g, label="Générateur (G)")
plt.xlabel("Époques")
plt.ylabel("Loss Wasserstein")
plt.legend()
plt.grid(True)
plt.show()

# --- FONCTION DE GÉNÉRATION SUR DEMANDE ---
def dessine_moi_un(chiffre, nombre_d_exemples=10):
    """
    Fonction pour générer des chiffres spécifiques.
    
    Arguments:
        chiffre (int) : Le nombre cible (0 à 9)
        nombre_d_exemples (int) : Nombre d'images à générer
    """
    gen.eval() # Mode évaluation
    
    with torch.no_grad():
        # 1. Bruit aléatoire
        noise = torch.randn(nombre_d_exemples, noise_dim, 1, 1).to(device)
        
        # 2. On définit l'étiquette cible pour toutes les images
        labels = torch.tensor([chiffre] * nombre_d_exemples).to(device)
        
        # 3. Génération conditionnée
        images_generees = gen(noise, labels)
        
        # 4. Affichage
        grid = make_grid(images_generees, nrow=5, normalize=True, value_range=(-1, 1))
        
        plt.figure(figsize=(8, 4))
        plt.axis("off")
        plt.title(f"Génération du chiffre : {chiffre}")
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.show()

# --- EXEMPLES D'UTILISATION ---
print("Génération de chiffres spécifiques :")

dessine_moi_un(0) # Exemple avec le 0
dessine_moi_un(7) # Exemple avec le 7
dessine_moi_un(9) # Exemple avec le 9

import torch

images_par_classe = 1000  # On veut 1000 exemples de chaque chiffre (0 à 9)
total_attendu = images_par_classe * 10

print(f" Démarrage de la production de {total_attendu} images pour chaque classe...")

# Passage en mode évaluation 
gen.eval()

# Listes pour stocker les résultats temporaires
all_images = []
all_labels = []

with torch.no_grad():
    # On boucle de 0 à 9 pour commander chaque chiffre précisément
    for chiffre in range(10):
        
        # 1. Préparation de la commande
        # On génère le bruit pour 200 images d'un coup
        noise = torch.randn(images_par_classe, noise_dim, 1, 1).to(device)
        
        # On crée l'étiquette correspondante (ex: une liste de 200 fois le chiffre "3")
        labels = torch.tensor([chiffre] * images_par_classe).to(device) 
        
        # 2. Fabrication
        # Le générateur produit les images en fonction de l'étiquette demandée
        fake_imgs = gen(noise, labels).cpu() 
        
        # 3. Stockage
        all_images.append(fake_imgs)
        all_labels.append(labels.cpu())
        
        print(f"   > Classe {chiffre} : {images_par_classe} images générées.")

    # 4. Assemblage final
    # On colle tous les morceaux pour faire un seul gros tenseur
    final_images = torch.cat(all_images, dim=0) # Tensor final [2000, 1, 28, 28]
    final_labels = torch.cat(all_labels, dim=0) # Tensor final [2000]

    # 5. Sauvegarde sur le disque
    # Ces fichiers pourront être chargés avec torch.load() dans un autre notebook
    torch.save(final_images, 'C-WGAN-GP_images_MNIST.pt')
    torch.save(final_labels, 'C-WGAN-GP_labels_MNIST.pt')

print(f" Terminé ! Fichiers sauvegardés : 'C-WGAN-GP_images_MNIST.pt' et 'C-WGAN-GP_labels_MNIST.pt'")
