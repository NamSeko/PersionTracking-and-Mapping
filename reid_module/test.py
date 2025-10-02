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
# config = load_config('market1500/config.yaml')
data_train_csv = config['data_train']
data_val_csv = config['data_val']
data_test_csv = config['data_test']
epochs = int(config['epochs'])
batch_size = int(config['batch_size'])
learning_rate = float(config['learning_rate'])
lr_patience = int(config['lr_patience'])
label_name = config['label_name']

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
dropout = 0.2

def test(model, test_loader, device):
    model.to(device)
    model.eval()
    compute_loss = ComputeLoss(margin=0.5, alpha=1.0, beta=0.3, gamma=0.3)

    test_loss, all_preds, all_labels = 0, [], []
    test_loop = tqdm(test_loader, desc=f'Test', unit='batch')
    with torch.no_grad():
        for (imgs1, imgs2, labels) in test_loop:
            imgs1, imgs2, labels = imgs1.to(device), imgs2.to(device), labels.float().to(device)
            outputs, f1, f2 = model(imgs1, imgs2, return_features=True)
            loss = compute_loss(outputs, f1, f2, labels)
            test_loss += loss.item() * imgs1.size(0)
            preds = (outputs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, zero_division=0)
    test_rec = recall_score(all_labels, all_preds, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f'Test: Loss={test_loss:.4f}, Acc={test_acc:.4f}, Prec={test_prec:.4f}, Rec={test_rec:.4f}, F1={test_f1:.4f}')

    return test_loss, [test_acc, test_prec, test_rec, test_f1]

transform = T.Compose([
    T.Resize((resize, resize)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_data = PersonDataset(csv_file=data_test_csv, transform=transform)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = PersonReIDModel(img_feature_dim=IMG_FEATURE_DIM, his_feature_dim=HIST_FEATURE_DIM, n_split=N_SPLIT)
model.to(device)
# model.load_state_dict(torch.load(f'{config['save_model_path']}/best_model.pth', map_location=device))
# model.load_state_dict(torch.load('6p_4c/models/best_model.pth', map_location=device))
model.load_state_dict(torch.load('market1500/models/best_model.pth', map_location=device))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_patience)

def main():
    epochs_test_loss = []
    epochs_test_metrics = []
    test_loss, test_metrics = test(model, test_loader, device)
    scheduler.step(test_loss)
    epochs_test_loss.append(test_loss)
    epochs_test_metrics.append(test_metrics)
    test_acc, test_prec, test_rec, test_f1 = test_metrics
if __name__ == "__main__":
    main()