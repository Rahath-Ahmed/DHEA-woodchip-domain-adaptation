import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ============================================================
# 0) DATA PATHS
# ============================================================

SOURCE_PATH = r"D:\Woodchip_moisture_content\Dataset\source_1"
TARGET_PATH = r"D:\Woodchip_moisture_content\Dataset\source_3"

# ============================================================
# 1) LABEL MAPPING
# ============================================================

def moisture_to_class(m):
    """Convert moisture percentage to class index."""
    if m <= 15:
        return 0   # Dry
    elif m <= 35:
        return 1   # Medium
    else:
        return 2   # Wet

FOLDER_TO_CLASS = {
    'dry': 0,
    'medium': 1,
    'wet': 2
}

class_names = ['Dry', 'Medium', 'Wet']


# ============================================================
# 2) SOURCE DATASET (LABELED) — source_1
# ============================================================

class SourceDataset(Dataset):
    def __init__(self, root=SOURCE_PATH):
        self.samples = []

        for moisture in os.listdir(root):
            if not moisture.isdigit():
                continue

            moisture_path = os.path.join(root, moisture)
            if not os.path.isdir(moisture_path):
                continue

            label = moisture_to_class(int(moisture))

            for container in os.listdir(moisture_path):
                container_path = os.path.join(moisture_path, container)
                if not os.path.isdir(container_path):
                    continue

                for img in os.listdir(container_path):
                    img_path = os.path.join(container_path, img)
                    self.samples.append((img_path, label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


# ============================================================
# 3) TARGET DATASET (UNLABELED FOR TRAINING) — source_3
# ============================================================

class TargetDataset(Dataset):
    def __init__(self, root=TARGET_PATH):
        self.samples = []

        for folder in os.listdir(root):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue

            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                self.samples.append(img_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        return self.transform(img)


# ============================================================
# 4) TARGET DATASET WITH LABELS (FOR EVALUATION) — source_3
# ============================================================

class WoodChipTargetEvalDataset(Dataset):
    def __init__(self, root=TARGET_PATH):
        self.samples = []

        for folder in os.listdir(root):
            folder_lower = folder.lower()
            if folder_lower not in FOLDER_TO_CLASS:
                continue

            label = FOLDER_TO_CLASS[folder_lower]
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue

            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                self.samples.append((img_path, label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label
