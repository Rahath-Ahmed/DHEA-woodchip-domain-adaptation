# ============================================================
# CDAN w/ SDAT for WoodChip Moisture Domain Adaptation
# Method follows val-iisc/SDAT (examples/cdan_sdat.py) exactly:
#   - SAM optimizer, two-step update (task loss smoothed only)
#   - dalib's ConditionalDomainAdversarialLoss (CDAN)
# Place this file in: D:\Woodchip_moisture_content\Adapt_codes\SDAT\examples\
# ============================================================

import os
import sys
import time
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)

# ------------------------------------------------------------
# Repo imports (SDAT). This file must sit in SDAT/examples/ so
# these relative paths resolve, exactly like the original scripts.
# ------------------------------------------------------------
sys.path.append('../')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy as dalib_accuracy
from common.utils.sam import SAM
import common.vision.models as models

# ------------------------------------------------------------
# Your dataset file
# ------------------------------------------------------------
sys.path.append(r"D:\Woodchip_moisture_content\Dataset")
from dataset_1_2 import SourceDataset, TargetDataset, WoodChipTargetEvalDataset

# ============================================================
# 1) SETTINGS  (paper defaults for Office-Home w/ ResNet-50)
# ============================================================
SEED = 42
NUM_CLASSES = 3
LABEL_NAMES = ['Dry', 'Medium', 'Wet']

ARCH = 'resnet50'
BOTTLENECK_DIM = 256
BATCH_SIZE = 32
EPOCHS = 10
ITERS_PER_EPOCH = 200
LR = 0.01
LR_GAMMA = 0.001
LR_DECAY = 0.75
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3
RHO = 0.02                 # SDAT smoothness hyperparameter (paper default for Office-Home)
TRADE_OFF = 1.0            # weight of the CDAN adversarial loss
ENTROPY_CONDITIONING = False
WORKERS = 0                # keep 0 on Windows to avoid multiprocessing issues

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

SAVE_DIR = r"D:\Woodchip_moisture_content\Adapt_codes\SDAT_light\WC_1_2"
os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2) TRANSFORMS
# ============================================================
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

eval_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# ============================================================
# 3) BUILD DATASETS
# We instantiate each dataset twice (train-transform copy and
# eval-transform copy) so the same underlying image, indexed by
# the same position, can be wrapped with different transforms
# without touching dataset_1_2.py.
# ============================================================

class UnlabeledWrapper(Dataset):
    """Wraps TargetDataset (which returns only an image) so it
    yields (image, dummy_label), matching what the SDAT training
    loop expects to unpack from the target iterator."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img = self.base[idx]
        return img, 0


print("Building datasets...")

# ---- Source (labeled) ----
src_train_ds_full = SourceDataset()
src_train_ds_full.transform = train_transform
src_eval_ds_full = SourceDataset()
src_eval_ds_full.transform = eval_transform

src_labels = [lbl for _, lbl in src_eval_ds_full.samples]
src_indices = list(range(len(src_labels)))
src_train_idx, src_val_idx = train_test_split(
    src_indices, test_size=0.2, random_state=SEED, stratify=src_labels
)

source_train_dataset = Subset(src_train_ds_full, src_train_idx)
source_val_dataset = Subset(src_eval_ds_full, src_val_idx)

# ---- Target (unlabeled for adversarial training) ----
tgt_unlabeled_train = TargetDataset()
tgt_unlabeled_train.transform = train_transform
target_train_dataset = UnlabeledWrapper(tgt_unlabeled_train)

# ---- Target (labeled, for evaluation only) ----
tgt_eval_ds_full = WoodChipTargetEvalDataset()
tgt_eval_ds_full.transform = eval_transform

tgt_labels = [lbl for _, lbl in tgt_eval_ds_full.samples]
tgt_indices = list(range(len(tgt_labels)))
_, tgt_test_idx = train_test_split(
    tgt_indices, test_size=0.2, random_state=SEED, stratify=tgt_labels
)
target_test_dataset = Subset(tgt_eval_ds_full, tgt_test_idx)

print(f"Source train: {len(source_train_dataset)} | Source val: {len(source_val_dataset)}")
print(f"Target train (unlabeled): {len(target_train_dataset)} | Target test: {len(target_test_dataset)}")

train_source_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=WORKERS, drop_last=True)
train_target_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=WORKERS, drop_last=True)
val_loader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=WORKERS)
test_loader = DataLoader(target_test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=WORKERS)

train_source_iter = ForeverDataIterator(train_source_loader)
train_target_iter = ForeverDataIterator(train_target_loader)

# ============================================================
# 4) MODEL: backbone + classifier + domain discriminator
# ============================================================
print("=> building model:", ARCH)
backbone = models.__dict__[ARCH](pretrained=True)
classifier = ImageClassifier(backbone, NUM_CLASSES, bottleneck_dim=BOTTLENECK_DIM,
                              pool_layer=None, finetune=True).to(device)
classifier_feature_dim = classifier.features_dim

domain_discri = DomainDiscriminator(
    classifier_feature_dim * NUM_CLASSES, hidden_size=1024).to(device)

base_optimizer = torch.optim.SGD
ad_optimizer = SGD(domain_discri.get_parameters(), LR, momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY, nesterov=True)
optimizer = SAM(classifier.get_parameters(), base_optimizer, rho=RHO, adaptive=False,
                 lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)

lr_scheduler = LambdaLR(optimizer, lambda x: LR * (1. + LR_GAMMA * float(x)) ** (-LR_DECAY))
lr_scheduler_ad = LambdaLR(ad_optimizer, lambda x: LR * (1. + LR_GAMMA * float(x)) ** (-LR_DECAY))

domain_adv = ConditionalDomainAdversarialLoss(
    domain_discri, entropy_conditioning=ENTROPY_CONDITIONING,
    num_classes=NUM_CLASSES, features_dim=classifier_feature_dim,
    randomized=False
).to(device)

# ============================================================
# 5) VALIDATION HELPER (source held-out set, used to pick best checkpoint)
# ============================================================
def validate(loader, model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


# ============================================================
# 6) TRAIN (identical structure to SDAT's cdan_sdat.py train())
# ============================================================
def train_one_epoch(epoch):
    classifier.train()
    domain_adv.train()

    for i in range(ITERS_PER_EPOCH):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s, x_t, labels_s = x_s.to(device), x_t.to(device), labels_s.to(device)

        optimizer.zero_grad()
        ad_optimizer.zero_grad()

        # --- Step 1: task loss only ---
        x = torch.cat((x_s, x_t), dim=0)
        y, f = classifier(x)
        y_s, y_t = y.chunk(2, dim=0)
        cls_loss = F.cross_entropy(y_s, labels_s)
        cls_loss.backward()
        optimizer.first_step(zero_grad=True)

        # --- Step 2: task loss + domain (adversarial) loss at perturbed weights ---
        y, f = classifier(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(y_s, f_s, y_t, f_t)
        loss = cls_loss + transfer_loss * TRADE_OFF
        loss.backward()

        ad_optimizer.step()
        optimizer.second_step(zero_grad=True)
        lr_scheduler.step()
        lr_scheduler_ad.step()

        if i % 200 == 0:
            print(f"Epoch [{epoch}] Iter [{i}/{ITERS_PER_EPOCH}] "
                  f"cls_loss={cls_loss.item():.3f} transfer_loss={transfer_loss.item():.3f} "
                  f"domain_acc={domain_adv.domain_discriminator_accuracy:.2f}")


print("\nTraining CDAN w/ SDAT...")
best_acc = 0.0
best_ckpt_path = os.path.join(SAVE_DIR, "best_checkpoint.pth")

start_time = time.time()
for epoch in range(EPOCHS):
    train_one_epoch(epoch)
    acc = validate(val_loader, classifier)
    print(f"Epoch {epoch}: source val acc = {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(classifier.state_dict(), best_ckpt_path)

print(f"\nBest source val acc: {best_acc:.2f}%")
print(f"Training time: {(time.time() - start_time)/60:.1f} min")

# load best checkpoint
classifier.load_state_dict(torch.load(best_ckpt_path, map_location=device))

# ============================================================
# 7) FULL PREDICTIONS (for metrics table, confusion matrix, viz grid)
# ============================================================
def get_predictions(dataset_with_labels):
    """Runs the classifier over a labeled dataset, returns (imgs_denorm, y_true, y_pred, y_probs)."""
    loader = DataLoader(dataset_with_labels, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    classifier.eval()
    imgs_list, y_true, y_pred, y_probs = [], [], [], []
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    with torch.no_grad():
        for images, labels in loader:
            images_dev = images.to(device)
            output = classifier(images_dev)
            probs = F.softmax(output, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            denorm = images * std + mean
            denorm = denorm.permute(0, 2, 3, 1).clamp(0, 1).numpy()

            imgs_list.append(denorm)
            y_true.append(labels.numpy())
            y_pred.append(preds)
            y_probs.append(probs)

    return (np.concatenate(imgs_list, axis=0),
            np.concatenate(y_true, axis=0),
            np.concatenate(y_pred, axis=0),
            np.concatenate(y_probs, axis=0))


print("\nRunning final predictions...")
# Full source (eval transform), full target (eval transform) - for metrics/viz like reference notebook
Xs_imgs, ys_int, src_preds_vis, src_probs_vis = get_predictions(src_eval_ds_full)
Xt_imgs, yt_int, tgt_preds_vis, tgt_probs_vis = get_predictions(tgt_eval_ds_full)

# Source validation subset & target test subset - for the metrics table (matches train/val/test splits)
_, ys_val_int, val_preds, _ = get_predictions(source_val_dataset)
_, yt_test_int, tgt_preds, _ = get_predictions(target_test_dataset)

# ============================================================
# 8) CLASSIFICATION REPORTS
# ============================================================
print("\nClassification Report (Source Validation):")
print(classification_report(ys_val_int, val_preds, target_names=LABEL_NAMES))

print("Classification Report (Target Test):")
print(classification_report(yt_test_int, tgt_preds, target_names=LABEL_NAMES))

# ============================================================
# 9) METRICS SUMMARY TABLE
# ============================================================
src_acc = accuracy_score(ys_val_int, val_preds)
src_precision = precision_score(ys_val_int, val_preds, average='weighted')
src_recall = recall_score(ys_val_int, val_preds, average='weighted')
src_f1 = f1_score(ys_val_int, val_preds, average='weighted')

tgt_acc = accuracy_score(yt_test_int, tgt_preds)
tgt_precision = precision_score(yt_test_int, tgt_preds, average='weighted')
tgt_recall = recall_score(yt_test_int, tgt_preds, average='weighted')
tgt_f1 = f1_score(yt_test_int, tgt_preds, average='weighted')

print("\nMETRICS SUMMARY")
print("=" * 50)
print(f"{'Metric':<16} {'Source':>10} {'Target':>10}")
print("-" * 50)
print(f"{'Accuracy':<16} {src_acc:>10.3f} {tgt_acc:>10.3f}")
print(f"{'Precision':<16} {src_precision:>10.3f} {tgt_precision:>10.3f}")
print(f"{'Recall':<16} {src_recall:>10.3f} {tgt_recall:>10.3f}")
print(f"{'F1':<16} {src_f1:>10.3f} {tgt_f1:>10.3f}")
print("=" * 50)

# ============================================================
# 10) CONFUSION MATRIX (target, full)
# ============================================================
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

plot_confusion_matrix(yt_int, tgt_preds_vis,
                       'Confusion Matrix — Target Domain (Full) — CDAN w/ SDAT',
                       'confusion_matrix_target_full.jpg')

# ============================================================
# 11) PREDICTION VISUALIZATION GRID
# ============================================================
np.random.seed(SEED)
src_vis_indices = np.random.choice(len(Xs_imgs), 5, replace=False)
tgt_vis_indices = np.random.choice(len(Xt_imgs), 5, replace=False)

def save_visualization_grid(X, y_true, y_pred, y_probs, indices, title, filename):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), dpi=300)
    for ax, idx in zip(axes, indices):
        img = X[idx]
        ax.imshow(img)
        ax.axis('off')
        pred, true = y_pred[idx], y_true[idx]
        conf = y_probs[idx][pred]
        color = 'green' if pred == true else 'red'
        ax.set_title(f"True: {LABEL_NAMES[true]}\nPred: {LABEL_NAMES[pred]}\nConf: {conf:.2f}",
                     color=color, fontsize=10, fontweight='bold')
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

save_visualization_grid(Xs_imgs, ys_int, src_preds_vis, src_probs_vis, src_vis_indices,
                         'CDAN w/ SDAT Predictions (Source Domain)', 'source_predictions_grid.jpg')
save_visualization_grid(Xt_imgs, yt_int, tgt_preds_vis, tgt_probs_vis, tgt_vis_indices,
                         'CDAN w/ SDAT Predictions (Target Domain)', 'target_predictions_grid.jpg')

# ============================================================
# 12) COPY ORIGINAL IMAGES (matching visualized samples)
# ============================================================
src_paths = [p for p, _ in src_eval_ds_full.samples]
tgt_paths = [p for p, _ in tgt_eval_ds_full.samples]

src_folder = os.path.join(SAVE_DIR, "source_samples")
tgt_folder = os.path.join(SAVE_DIR, "target_samples")
os.makedirs(src_folder, exist_ok=True)
os.makedirs(tgt_folder, exist_ok=True)

def copy_original_images(paths, y_true, y_pred, y_probs, indices, folder):
    for i, idx in enumerate(indices):
        true_label = LABEL_NAMES[y_true[idx]]
        pred_label = LABEL_NAMES[y_pred[idx]]
        conf = y_probs[idx][y_pred[idx]]
        ext = os.path.splitext(paths[idx])[1]
        filename = f"{i+1}_True_{true_label}_Pred_{pred_label}_Conf_{conf:.2f}{ext}"
        save_path = os.path.join(folder, filename)
        shutil.copy2(paths[idx], save_path)
        print(f"Saved: {save_path}")

copy_original_images(src_paths, ys_int, src_preds_vis, src_probs_vis, src_vis_indices, src_folder)
copy_original_images(tgt_paths, yt_int, tgt_preds_vis, tgt_probs_vis, tgt_vis_indices, tgt_folder)

# ============================================================
# 13) PREDICTION SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)
src_n_correct = int(np.sum(src_preds_vis == ys_int))
tgt_n_correct = int(np.sum(tgt_preds_vis == yt_int))
print(f"Source Domain: {src_n_correct}/{len(ys_int)} correct ({accuracy_score(ys_int, src_preds_vis):.1%})")
print(f"Target Domain: {tgt_n_correct}/{len(yt_int)} correct ({accuracy_score(yt_int, tgt_preds_vis):.1%})")
print("=" * 60)

# ============================================================
# 14) SAVE ARRAYS
# ============================================================
np.save(os.path.join(SAVE_DIR, "ys_int.npy"), ys_int)
np.save(os.path.join(SAVE_DIR, "yt_int.npy"), yt_int)
np.save(os.path.join(SAVE_DIR, "ys_val_int.npy"), ys_val_int)
np.save(os.path.join(SAVE_DIR, "val_preds.npy"), val_preds)
np.save(os.path.join(SAVE_DIR, "yt_test_int.npy"), yt_test_int)
np.save(os.path.join(SAVE_DIR, "tgt_preds.npy"), tgt_preds)
np.save(os.path.join(SAVE_DIR, "src_probs_vis.npy"), src_probs_vis)
np.save(os.path.join(SAVE_DIR, "src_preds_vis.npy"), src_preds_vis)
np.save(os.path.join(SAVE_DIR, "tgt_probs_vis.npy"), tgt_probs_vis)
np.save(os.path.join(SAVE_DIR, "tgt_preds_vis.npy"), tgt_preds_vis)

print(f"\nAll outputs saved to: {SAVE_DIR}")
