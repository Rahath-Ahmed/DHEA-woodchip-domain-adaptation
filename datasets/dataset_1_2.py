import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ============================================================
# 0) DATA PATHS
# ============================================================

SOURCE_PATH = r"D:\Woodchip_moisture_content\Dataset\source_1"
TARGET_PATH = r"D:\Woodchip_moisture_content\Dataset\source_2"

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

class_names = ['Dry', 'Medium', 'Wet']


# ============================================================
# 2) SOURCE DATASET (LABELED)
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
# 3) TARGET DATASET (UNLABELED FOR TRAINING)
# ============================================================

class TargetDataset(Dataset):
    def __init__(self, root=TARGET_PATH):
        self.samples = []

        for batch in os.listdir(root):
            batch_path = os.path.join(root, batch)
            if not os.path.isdir(batch_path):
                continue

            for moisture in os.listdir(batch_path):
                moisture_path = os.path.join(batch_path, moisture)
                if not os.path.isdir(moisture_path):
                    continue

                for container in os.listdir(moisture_path):
                    container_path = os.path.join(moisture_path, container)
                    if not os.path.isdir(container_path):
                        continue

                    for img in os.listdir(container_path):
                        img_path = os.path.join(container_path, img)
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
# 4) TARGET DATASET WITH LABELS (FOR EVALUATION)
# ============================================================

class WoodChipTargetEvalDataset(Dataset):
    def __init__(self, root=TARGET_PATH):
        self.samples = []

        for batch in os.listdir(root):
            batch_path = os.path.join(root, batch)
            if not os.path.isdir(batch_path):
                continue

            for level in os.listdir(batch_path):
                if not level.isdigit():
                    continue

                label = moisture_to_class(int(level))
                level_path = os.path.join(batch_path, level)
                if not os.path.isdir(level_path):
                    continue

                for container in os.listdir(level_path):
                    container_path = os.path.join(level_path, container)
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