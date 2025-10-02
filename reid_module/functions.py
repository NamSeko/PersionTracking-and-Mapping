import yaml
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import models

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def load_config(file_path):
      with open(file_path, "r") as file:
          return yaml.safe_load(file)
      
def contractive_loss_fn(f1, f2, labels, margin=0.5):
    dist = F.pairwise_distance(f1, f2, keepdim=True)
    loss_pos = labels * dist.pow(2)
    loss_neg = (1 - labels) * F.relu(margin - dist).pow(2)
    loss = torch.mean(loss_pos + loss_neg)
    return loss

class ComputeLoss():
    def __init__(self, alpha=0.3, beta=0.3, gamma=0.3, margin=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)

    def __call__(self, output, f1, f2, labels):
        loss_bce = self.bce_loss(output.squeeze(), labels)
        loss_con = contractive_loss_fn(f1, f2, labels, self.margin)
        loss_cos = self.cosine_loss(f1, f2, 2*labels-1)
        combined_loss = self.alpha * loss_bce + self.beta * loss_con + self.gamma * loss_cos
        return combined_loss

class PersonDataset(data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.df.iloc[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, label

class PersonReIDModel(nn.Module):
    def __init__(self, img_feature_dim=128, his_feature_dim=128, n_split=4, dropout=0.2):
        super(PersonReIDModel, self).__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(backbone.fc.in_features, img_feature_dim)
        self.cnn_encoder = backbone
        self.n_split = n_split
        self.dropout = dropout

        self.histogram_encoder = nn.Sequential(
            nn.Linear(self.n_split*self.n_split*64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, his_feature_dim)
        )
        
        feature_dim = img_feature_dim + his_feature_dim
        self.fc = nn.Sequential(
            nn.Linear(feature_dim*4, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        
    def forward_once(self, img):
        img_feat = self.cnn_encoder(img)
        hist = self.compute_histogram(img) # (B, n_split*n_split, 256)
        hist = hist.view(hist.size(0), -1)
        hist_feat = self.histogram_encoder(hist)
        out = torch.cat((img_feat, hist_feat), dim=1) # (B, feature_dim)
        return out
        
    def forward(self, img1, img2, return_features=False):
        feat1 = self.forward_once(img1) # (B, feature_dim)
        feat2 = self.forward_once(img2) # (B, feature_dim)
        
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        
        diff = torch.abs(feat1 - feat2) # (B, feature_dim)
        prod = feat1 * feat2 # (B, feature_dim)
        combined = torch.cat((feat1, feat2, diff, prod), dim=1) # (B, feature_dim*4)
        out = self.fc(combined) # (B, 1)
        if return_features:
            return out, feat1, feat2
        return out        

    def compute_histogram(self, img_tensor, bins=64):
        gray_img = img_tensor.mean(dim=1)
        
        B, H, W = gray_img.shape
        step_h, step_w = H // self.n_split, W // self.n_split

        # List cho batch
        batch_hist = []

        for b in range(B):
            img = gray_img[b]
            hist_blocks = []
            for i in range(self.n_split):
                for j in range(self.n_split):
                    patch = img[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w]
                    patch = (patch * 255).clamp(0, 255)
                    h_patch = torch.histc(patch, bins=bins, min=0, max=255)
                    # Normalize histogram
                    h_patch = h_patch / (h_patch.sum() + 1e-6)
                    hist_blocks.append(h_patch)
            hist_blocks = torch.stack(hist_blocks, dim=0) # (n_split*n_split, bins)
            batch_hist.append(hist_blocks)
        batch_hist = torch.stack(batch_hist, dim=0) # (B, n_split*n_split, bins)
        return batch_hist
