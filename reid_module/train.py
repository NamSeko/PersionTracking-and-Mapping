from functions import *
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #type: ignore

config = load_config('6p_4c/config.yaml')
data_train_csv = config['data_train']
data_val_csv = config['data_val']
data_test_csv = config['data_test']
label_name = config['label_name']
epochs = int(config['epochs'])
batch_size = int(config['batch_size'])
learning_rate = float(config['learning_rate'])
lr_patience = int(config['lr_patience'])
early_topping_patience = int(config['early_stopping_patience'])

print(f'Train CSV file: {data_train_csv}')
print(f'Validation CSV file: {data_val_csv}')
print(f'Test CSV file: {data_test_csv}')
print(f'Number of epochs: {epochs}')
print(f'Batch size: {batch_size}')
print(f'Learning rate: {learning_rate}')
print(f'LR patience: {lr_patience}')

set_seed(42)

IMG_FEATURE_DIM = 256
HIST_FEATURE_DIM = 128
N_SPLIT = 8
resize = 64
dropout = 0.3        

def train(model, epoch, train_loader, optimizer, device):
    model.to(device)
    model.train()
    compute_loss = ComputeLoss(margin=0.5, alpha=1.0, beta=0.3, gamma=0.3)
    epoch_loss, all_preds, all_labels = 0, [], []
    train_loop = tqdm(train_loader, desc=f'Train Epoch {epoch+1}', unit='batch')
    for (imgs1, imgs2, labels) in train_loop:
        imgs1, imgs2, labels = imgs1.to(device), imgs2.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs, f1, f2 = model(imgs1, imgs2, return_features=True)
        loss = compute_loss(outputs, f1, f2, labels)
        epoch_loss += loss.item() * imgs1.size(0)
        preds = (outputs >= 0.51).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        loss.backward()
        optimizer.step()

    epoch_loss /= len(train_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
    epoch_rec = recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print(f'Train Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Prec={epoch_prec:.4f}, Rec={epoch_rec:.4f}, F1={epoch_f1:.4f}')

    return epoch_loss, [epoch_acc, epoch_prec, epoch_rec, epoch_f1]

def validation(model, epoch, val_loader, device):
    model.to(device)
    model.eval()
    criterion = ComputeLoss(margin=0.5, alpha=1.0, beta=0.3, gamma=0.3)
    val_loss, all_preds, all_labels = 0, [], []
    
    val_loop = tqdm(val_loader, desc=f'Val Epoch {epoch+1}', unit='batch')
    with torch.no_grad():
        for (imgs1, imgs2, labels) in val_loop:
            imgs1, imgs2, labels = imgs1.to(device), imgs2.to(device), labels.float().to(device)
            outputs, f1, f2 = model(imgs1, imgs2, return_features=True)
            loss = criterion(outputs, f1, f2, labels)
            val_loss += loss.item() * imgs1.size(0)
            preds = (outputs >= 0.51).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    val_acc = accuracy_score(all_labels, all_preds)
    val_prec = precision_score(all_labels, all_preds, zero_division=0)
    val_rec = recall_score(all_labels, all_preds, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print(f'Validation Epoch {epoch+1}: Loss={val_loss:.4f}, Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, F1={val_f1:.4f}')

    return val_loss, [val_acc, val_prec, val_rec, val_f1]

train_transform = T.Compose([
    T.Resize((resize, resize)),
    T.RandomRotation(degrees=15),
    T.RandomResizedCrop(size=(resize, resize), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((resize, resize)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = PersonDataset(csv_file=data_train_csv, transform=train_transform)
val_data = PersonDataset(csv_file=data_val_csv, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'Using device: {device}')

model = PersonReIDModel(img_feature_dim=IMG_FEATURE_DIM, his_feature_dim=HIST_FEATURE_DIM, n_split=N_SPLIT, dropout=dropout)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience)

def main():
    epochs_train_loss = []
    epochs_val_loss = []
    epochs_train_metrics_1 = []
    epochs_train_metrics_2 = []
    epochs_train_metrics_3 = []
    epochs_train_metrics_4 = []
    epochs_val_metrics_1 = []
    epochs_val_metrics_2 = []
    epochs_val_metrics_3 = []
    epochs_val_metrics_4 = []
    best_score = 0.0
    best_epoch = -1
    early_stopping_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_metrics = train(model, epoch, train_loader, optimizer, device)
        val_loss, val_metrics = validation(model, epoch, val_loader, device)
        scheduler.step(val_loss)
        epochs_train_loss.append(train_loss)
        epochs_val_loss.append(val_loss)
        epochs_train_metrics_1.append(train_metrics[0])
        epochs_train_metrics_2.append(train_metrics[1])
        epochs_train_metrics_3.append(train_metrics[2])
        epochs_train_metrics_4.append(train_metrics[3])
        epochs_val_metrics_1.append(val_metrics[0])
        epochs_val_metrics_2.append(val_metrics[1])
        epochs_val_metrics_3.append(val_metrics[2])
        epochs_val_metrics_4.append(val_metrics[3])

        A = np.array(epochs_train_loss)
        B = np.array(epochs_val_loss)

        M1 = np.array(epochs_train_metrics_1)
        M2 = np.array(epochs_train_metrics_2)
        M3 = np.array(epochs_train_metrics_3)
        M4 = np.array(epochs_train_metrics_4)
        M5 = np.array(epochs_val_metrics_1)
        M6 = np.array(epochs_val_metrics_2)
        M7 = np.array(epochs_val_metrics_3)
        M8 = np.array(epochs_val_metrics_4)

        np.save(f"{config['checkpoint_path']}/train_loss.npy", A)
        np.save(f"{config['checkpoint_path']}/val_loss.npy", B)
        np.save(f"{config['checkpoint_path']}/train_acc.npy", M1)
        np.save(f"{config['checkpoint_path']}/train_prec.npy", M2)
        np.save(f"{config['checkpoint_path']}/train_rec.npy", M3)
        np.save(f"{config['checkpoint_path']}/train_f1.npy", M4)
        np.save(f"{config['checkpoint_path']}/val_acc.npy", M5)
        np.save(f"{config['checkpoint_path']}/val_prec.npy", M6)
        np.save(f"{config['checkpoint_path']}/val_rec.npy", M7)
        np.save(f"{config['checkpoint_path']}/val_f1.npy", M8)
        val_f1 = val_metrics[3]
        if val_f1 > best_score:
            best_score = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), f"{config['save_model_path']}/best_model.pth")
            print(f"Best model saved at epoch {epoch+1} with F1: {best_score:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_topping_patience}")
            if early_stopping_counter >= early_topping_patience:
                print("Early stopping triggered.")
                break
    print(f"Training complete. Best F1: {best_score:.4f} at epoch {best_epoch+1}")
    

if __name__ == "__main__":
    main()