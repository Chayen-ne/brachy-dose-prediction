import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from torch.cuda.amp import GradScaler, autocast

#################################
# 1. ฟังก์ชันโหลดและเตรียมข้อมูล
#################################

def load_data_and_split(train_dir, test_dir, val_ratio=0.2, random_seed=42):
    """
    โหลดข้อมูลจากโฟลเดอร์ train และ test และแบ่งข้อมูลในโฟลเดอร์ train เป็น train/val
    
    Args:
        train_dir (str): path ไปยังโฟลเดอร์ train (มีทั้ง train+val)
        test_dir (str): path ไปยังโฟลเดอร์ test
        val_ratio (float): สัดส่วนข้อมูลที่จะใช้เป็น validation
        random_seed (int): random seed สำหรับการแบ่งข้อมูล
        
    Returns:
        dict: dictionary ที่มีข้อมูลสำหรับ train, val, test
    """
    # โหลดข้อมูลจากโฟลเดอร์ train (ซึ่งมีทั้ง train+val รวมกัน)
    train_patient_dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))]
    
    # โหลดข้อมูลจากโฟลเดอร์ test
    test_patient_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir) 
                        if os.path.isdir(os.path.join(test_dir, d))]
    
    # ตรวจสอบจำนวนข้อมูล
    print(f"Found {len(train_patient_dirs)} patients in train directory")
    print(f"Found {len(test_patient_dirs)} patients in test directory")
    
    # รวบรวม paths ของไฟล์ในโฟลเดอร์ train
    train_val_ct_paths = []
    train_val_structure_paths = []
    train_val_dose_paths = []
    
    for patient_dir in train_patient_dirs:
        ct_file = os.path.join(patient_dir, "CT_volume.npy")
        structure_file = os.path.join(patient_dir, "Combined_Mask.npy")
        dose_file = os.path.join(patient_dir, f"RD{patient_dir.split('_')[-1]}_preprocessed.npy")
        
        if os.path.exists(ct_file) and os.path.exists(structure_file) and os.path.exists(dose_file):
            train_val_ct_paths.append(ct_file)
            train_val_structure_paths.append(structure_file)
            train_val_dose_paths.append(dose_file)
        else:
            print(f"Warning: Missing files for patient {patient_dir}")
    
    # รวบรวม paths ของไฟล์ในโฟลเดอร์ test
    test_ct_paths = []
    test_structure_paths = []
    test_dose_paths = []
    
    for patient_dir in test_patient_dirs:
        ct_file = os.path.join(patient_dir, "CT_volume.npy")
        structure_file = os.path.join(patient_dir, "Combined_Mask.npy")
        dose_file = os.path.join(patient_dir, f"RD{patient_dir.split('_')[-1]}_preprocessed.npy")
        
        if os.path.exists(ct_file) and os.path.exists(structure_file) and os.path.exists(dose_file):
            test_ct_paths.append(ct_file)
            test_structure_paths.append(structure_file)
            test_dose_paths.append(dose_file)
        else:
            print(f"Warning: Missing files for patient {patient_dir}")
    
    # แบ่งข้อมูลในโฟลเดอร์ train เป็น train/val
    train_ct, val_ct, train_structure, val_structure, train_dose, val_dose = train_test_split(
        train_val_ct_paths, 
        train_val_structure_paths, 
        train_val_dose_paths, 
        test_size=val_ratio, 
        random_state=random_seed
    )
    
    print(f"Split train/val: {len(train_ct)} train samples, {len(val_ct)} validation samples")
    print(f"Test: {len(test_ct_paths)} test samples")
    
    return {
        "train": {"ct": train_ct, "structure": train_structure, "dose": train_dose},
        "val": {"ct": val_ct, "structure": val_structure, "dose": val_dose},
        "test": {"ct": test_ct_paths, "structure": test_structure_paths, "dose": test_dose_paths}
    }

def pad_to_same_size(tensor, target_size):
    """
    Pads a tensor to the target size with zeros.
    Args:
        tensor (torch.Tensor): The input tensor to be padded.
        target_size (tuple): The target size (depth, height, width).
    Returns:
        torch.Tensor: Padded tensor.
    """
    padding = []
    for current, target in zip(reversed(tensor.shape), reversed(target_size)):
        diff = target - current
        padding.extend([diff // 2, diff - diff // 2])
    return F.pad(tensor, padding, mode='constant', value=0)

def get_max_shape(paths):
    """
    Compute the maximum shape across all .npy files in the provided paths.
    Args:
        paths (list of str): List of paths to the .npy files.
    Returns:
        tuple: Maximum shape (depth, height, width).
    """
    max_depth, max_height, max_width = 0, 0, 0
    for path in paths:
        data = np.load(path)
        max_depth = max(max_depth, data.shape[0])
        max_height = max(max_height, data.shape[1])
        max_width = max(max_width, data.shape[2])
    return (max_depth, max_height, max_width)

def normalize_ct(ct_volume, min_val=-1000, max_val=2000):
    """
    Normalize CT volume to [0, 1] range.
    Args:
        ct_volume (numpy.ndarray): The CT volume to normalize.
        min_val (float): Minimum HU value to clip.
        max_val (float): Maximum HU value to clip.
    Returns:
        numpy.ndarray: Normalized CT volume in range [0, 1].
    """
    ct_volume = np.clip(ct_volume, min_val, max_val)
    ct_volume = (ct_volume - min_val) / (max_val - min_val)
    return ct_volume

def normalize_dose(dose_volume, min_dose=None, max_dose=None):
    """
    Normalize dose volume based on metadata or find min/max.
    Args:
        dose_volume (numpy.ndarray): The dose volume to normalize.
        min_dose (float, optional): Minimum dose value. If None, uses array minimum.
        max_dose (float, optional): Maximum dose value. If None, uses array maximum.
    Returns:
        numpy.ndarray: Normalized dose volume in range [0, 1].
    """
    if min_dose is None:
        min_dose = dose_volume.min()
    if max_dose is None:
        max_dose = dose_volume.max()
    
    # Avoid division by zero
    if max_dose == min_dose:
        return np.zeros_like(dose_volume)
    
    normalized_dose = (dose_volume - min_dose) / (max_dose - min_dose)
    return normalized_dose

def verify_data_sample(ct_path, structure_path, dose_path):
    """
    Verify that a single data sample is valid and has consistent dimensions.
    Args:
        ct_path (str): Path to CT volume file.
        structure_path (str): Path to structure mask file.
        dose_path (str): Path to dose volume file.
    Returns:
        bool: True if the data sample is valid, False otherwise.
    """
    try:
        ct = np.load(ct_path)
        structure = np.load(structure_path)
        dose = np.load(dose_path)
        
        # Check dimensions match
        if ct.shape != structure.shape or ct.shape != dose.shape:
            print(f"Dimension mismatch: CT {ct.shape}, Structure {structure.shape}, Dose {dose.shape}")
            return False
        
        # Check for non-finite values
        if not np.isfinite(ct).all():
            print(f"CT contains non-finite values at {ct_path}")
            return False
        if not np.isfinite(structure).all():
            print(f"Structure contains non-finite values at {structure_path}")
            return False
        if not np.isfinite(dose).all():
            print(f"Dose contains non-finite values at {dose_path}")
            return False
        
        return True
    except Exception as e:
        print(f"Error verifying data: {e}")
        return False

def verify_dataset(data_dict, subset="train"):
    """
    Verify all samples in a dataset.
    Args:
        data_dict (dict): Dictionary containing dataset paths.
        subset (str): Subset to verify ('train', 'val', or 'test').
    Returns:
        list: Indices of valid samples.
    """
    valid_indices = []
    ct_paths = data_dict[subset]["ct"]
    structure_paths = data_dict[subset]["structure"]
    dose_paths = data_dict[subset]["dose"]
    
    for i, (ct_path, structure_path, dose_path) in enumerate(zip(ct_paths, structure_paths, dose_paths)):
        if verify_data_sample(ct_path, structure_path, dose_path):
            valid_indices.append(i)
        else:
            print(f"Sample {i} in {subset} set is invalid")
    
    print(f"Verified {subset} set: {len(valid_indices)}/{len(ct_paths)} valid samples")
    return valid_indices

#################################
# 2. Dataset และ DataLoader
#################################

class DosePredictionDataset(Dataset):
    def __init__(self, ct_paths, structure_paths, dose_paths=None, csv_path=None, mode="train", normalize=True):
        self.ct_paths = ct_paths
        self.structure_paths = structure_paths
        self.dose_paths = dose_paths
        self.mode = mode
        self.normalize = normalize

        if csv_path is not None and os.path.exists(csv_path):
            metadata = pd.read_csv(csv_path)
            self.metadata = {row['FileName']: (row['MinDose'], row['DynamicMaxDose']) for _, row in metadata.iterrows()}
        else:
            self.metadata = None

        # Compute the dynamic target size based on the largest tensor in the dataset
        self.target_size = self.compute_max_target_size()
        print(f"Target size for padding: {self.target_size}")

    def compute_max_target_size(self):
        """
        Compute the maximum shape for padding all tensors in the dataset.
        Returns:
            tuple: Maximum shape (depth, height, width).
        """
        max_ct_shape = get_max_shape(self.ct_paths)
        max_structure_shape = get_max_shape(self.structure_paths)
        if self.mode == "train" and self.dose_paths is not None:
            max_dose_shape = get_max_shape(self.dose_paths)
        else:
            max_dose_shape = (0, 0, 0)

        # Determine the overall maximum shape
        return tuple(max(a, b, c) for a, b, c in zip(max_ct_shape, max_structure_shape, max_dose_shape))

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx):
        # Load CT volume and normalize if needed
        ct = np.load(self.ct_paths[idx])
        if self.normalize:
            ct = normalize_ct(ct)
        ct = torch.tensor(ct, dtype=torch.float32)
        
        # Load structure mask
        structure = torch.tensor(np.load(self.structure_paths[idx]), dtype=torch.float32)

        # Pad to the dynamically determined target size
        ct = pad_to_same_size(ct, self.target_size)
        structure = pad_to_same_size(structure, self.target_size)

        # Ensure channel dimension is 2 for input_tensor
        input_tensor = torch.stack([ct, structure], dim=0)

        if self.mode == "train" and self.dose_paths is not None:
            # Load dose and normalize if needed
            dose = np.load(self.dose_paths[idx])
            if self.normalize and self.metadata is not None:
                filename = os.path.basename(self.dose_paths[idx])
                if filename in self.metadata:
                    min_dose, max_dose = self.metadata[filename]
                    dose = normalize_dose(dose, min_dose, max_dose)
                else:
                    dose = normalize_dose(dose)
            elif self.normalize:
                dose = normalize_dose(dose)
                
            dose = torch.tensor(dose, dtype=torch.float32)
            dose = pad_to_same_size(dose, self.target_size)
            
            # Ensure channel dimension is 1 for dose
            dose = dose.unsqueeze(0)
            return input_tensor, dose
        else:
            return input_tensor

def create_datasets_and_loaders(data_dict, csv_path, batch_size=2, normalize=True):
    """
    สร้าง datasets และ dataloaders จากข้อมูลที่โหลดและแบ่งแล้ว
    
    Args:
        data_dict (dict): dictionary ที่มีข้อมูล paths สำหรับ train, val, test
        csv_path (str): path ไปยังไฟล์ metadata csv
        batch_size (int): ขนาด batch
        normalize (bool): ใช้การ normalize ข้อมูลหรือไม่
        
    Returns:
        dict: dictionary ที่มี datasets และ dataloaders
    """
    # สร้าง datasets
    train_dataset = DosePredictionDataset(
        data_dict["train"]["ct"], 
        data_dict["train"]["structure"], 
        data_dict["train"]["dose"], 
        csv_path, 
        mode="train",
        normalize=normalize
    )
    
    val_dataset = DosePredictionDataset(
        data_dict["val"]["ct"], 
        data_dict["val"]["structure"], 
        data_dict["val"]["dose"], 
        csv_path, 
        mode="train",
        normalize=normalize
    )
    
    test_dataset = DosePredictionDataset(
        data_dict["test"]["ct"], 
        data_dict["test"]["structure"], 
        data_dict["test"]["dose"], 
        csv_path, 
        mode="train",
        normalize=normalize
    )
    
    # สร้าง dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True,
        num_workers=2  # ปรับตามจำนวน CPU cores ที่มี
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )
    
    return {
        "datasets": {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        },
        "loaders": {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
    }

#################################
# 3. โมเดล Cascade3DUNet
#################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample  # For matching dimensions if needed

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
    
# Utility Function
def crop_to_match(tensor, target_tensor):
    """
    Crops tensor to match the size of target_tensor along spatial dimensions (D, H, W).
    """
    diff_depth = tensor.size(2) - target_tensor.size(2)
    diff_height = tensor.size(3) - target_tensor.size(3)
    diff_width = tensor.size(4) - target_tensor.size(4)

    # Crop along each dimension
    tensor = tensor[:, :, 
                   diff_depth // 2:tensor.size(2) - diff_depth // 2,
                   diff_height // 2:tensor.size(3) - diff_height // 2,
                   diff_width // 2:tensor.size(4) - diff_width // 2]
    return tensor

# Cascade3DUNet Class
class Cascade3DUNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=1):
        super(Cascade3DUNet, self).__init__()

        # Input projection to match encoder1's input size
        self.input_projection = nn.Conv3d(in_channels, 32, kernel_size=1)

        # Encoder
        self.encoder1 = ResidualBlock(32, 32)
        self.encoder2 = ResidualBlock(32, 64, downsample=nn.Conv3d(32, 64, kernel_size=1))
        self.encoder3 = ResidualBlock(64, 128, downsample=nn.Conv3d(64, 128, kernel_size=1))
        self.encoder4 = ResidualBlock(128, 256, downsample=nn.Conv3d(128, 256, kernel_size=1))
        self.encoder5 = ResidualBlock(256, 512, downsample=nn.Conv3d(256, 512, kernel_size=1))

        # Decoder
        self.decoder4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder4_conv = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.decoder3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3_conv = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.decoder2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2_conv = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.decoder1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder1_conv = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        # Projection layers for skip connections
        self.proj4 = nn.Conv3d(256, 256, kernel_size=1)
        self.proj3 = nn.Conv3d(128, 128, kernel_size=1)
        self.proj2 = nn.Conv3d(64, 64, kernel_size=1)
        self.proj1 = nn.Conv3d(32, 32, kernel_size=1)

        # Final layer
        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Input projection
        x = self.input_projection(x)  # Convert in_channels=2 to out_channels=32

        # Encoder
        enc1 = self.encoder1(x)  # [B, 32, D, H, W]
        enc2 = self.encoder2(F.max_pool3d(enc1, 2))  # [B, 64, D/2, H/2, W/2]
        enc3 = self.encoder3(F.max_pool3d(enc2, 2))  # [B, 128, D/4, H/4, W/4]
        enc4 = self.encoder4(F.max_pool3d(enc3, 2))  # [B, 256, D/8, H/8, W/8]
        enc5 = self.encoder5(F.max_pool3d(enc4, 2))  # [B, 512, D/16, H/16, W/16]

        # Decoder
        x = self.decoder4(enc5)  # [B, 256, D/8, H/8, W/8]
        x = self.decoder4_conv(x)
        x = crop_to_match(x, enc4) + self.proj4(enc4)

        x = self.decoder3(x)  # [B, 128, D/4, H/4, W/4]
        x = self.decoder3_conv(x)
        x = crop_to_match(x, enc3) + self.proj3(enc3)

        x = self.decoder2(x)  # [B, 64, D/2, H/2, W/2]
        x = self.decoder2_conv(x)
        x = crop_to_match(x, enc2) + self.proj2(enc2)

        x = self.decoder1(x)  # [B, 32, D, H, W]
        x = self.decoder1_conv(x)
        x = crop_to_match(x, enc1) + self.proj1(enc1)

        x = self.final_conv(x)  # Final layer
        return x

#################################
# 4. ฟังก์ชันสำหรับเทรนและประเมินผล
#################################

# Compute PSNR
def compute_psnr(prediction, target, max_value=1.0):
    mse = nn.functional.mse_loss(prediction, target, reduction='mean').item()
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_value)) - 10 * torch.log10(torch.tensor(mse))
    return psnr.item()

# Compute MAE
def compute_mae(prediction, target):
    return torch.mean(torch.abs(prediction - target)).item()

# Validation Function
def validate_model_with_mae_list(model, val_loader, criterion, device="cuda"):
    """
    Validate the model and calculate Loss, PSNR, and MAE for the validation dataset.
    Args:
        model (nn.Module): The trained model to validate.
        val_loader (DataLoader): Validation DataLoader.
        criterion (callable): Loss function.
        device (str): Device to use for computation ("cuda" or "cpu").
    Returns:
        float: Validation loss.
        float: Average PSNR.
        list: List of MAE values for all validation samples.
    """
    model.eval()
    val_loss = 0
    psnr_total = 0
    mae_list = []  # เก็บค่า MAE ของทุกตัวอย่างใน Validation

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            # คำนวณ Loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # คำนวณ PSNR
            psnr = compute_psnr(outputs, targets)
            psnr_total += psnr

            # คำนวณ MAE สำหรับแต่ละตัวอย่าง
            mae = torch.mean(torch.abs(outputs - targets)).item()
            mae_list.append(mae)

    val_loss /= len(val_loader)
    psnr_avg = psnr_total / len(val_loader)

    return val_loss, psnr_avg, mae_list

def plot_training_history(df, save_path="training_history.png"):
    """สร้างกราฟแสดงประวัติการฝึก"""
    plt.figure(figsize=(15, 10))
    
    # กราฟ Loss
    plt.subplot(2, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], 'b-', label='Training Loss')
    plt.plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # กราฟ MAE
    plt.subplot(2, 2, 2)
    plt.plot(df['epoch'], df['train_mae_mean'], 'b-', label='Training MAE')
    plt.plot(df['epoch'], df['val_mae_mean'], 'r-', label='Validation MAE')
    plt.fill_between(
        df['epoch'], 
        df['train_mae_mean'] - df['train_mae_std'], 
        df['train_mae_mean'] + df['train_mae_std'], 
        color='b', 
        alpha=0.2
    )
    plt.fill_between(
        df['epoch'], 
        df['val_mae_mean'] - df['val_mae_std'], 
        df['val_mae_mean'] + df['val_mae_std'], 
        color='r', 
        alpha=0.2
    )
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    plt.grid(True)
    
    # กราฟ PSNR
    plt.subplot(2, 2, 3)
    plt.plot(df['epoch'], df['val_psnr'], 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR')
    plt.grid(True)
    
    # กราฟ Learning Rate
    if 'learning_rate' in df.columns:
        plt.subplot(2, 2, 4)
        plt.plot(df['epoch'], df['learning_rate'], 'm-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate over Training')
        plt.yscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_predictions(model, test_loader, num_samples=3, device="cuda", save_dir="predictions"):
    """แสดงตัวอย่างการทำนายของโมเดล"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    samples = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # เลือกตัวอย่างจากแต่ละ batch
            for i in range(min(inputs.size(0), num_samples - len(samples))):
                samples.append({
                    'ct': inputs[i, 0].cpu().numpy(),  # ช่องแรกคือ CT
                    'structure': inputs[i, 1].cpu().numpy(),  # ช่องที่สองคือ structure
                    'true_dose': targets[i, 0].cpu().numpy(),  # ช่องเดียวสำหรับ dose
                    'pred_dose': outputs[i, 0].cpu().numpy()
                })
                
            if len(samples) >= num_samples:
                break
    
    # แสดงตัวอย่าง
    for idx, sample in enumerate(samples):
        plt.figure(figsize=(15, 10))
        
        # เลือกสไลด์กลาง
        slice_idx = sample['ct'].shape[0] // 2
        
        # แสดง CT
        plt.subplot(2, 2, 1)
        plt.imshow(sample['ct'][slice_idx], cmap='gray')
        plt.title(f'CT Slice {slice_idx}')
        plt.colorbar()
        
        # แสดง Structure
        plt.subplot(2, 2, 2)
        plt.imshow(sample['structure'][slice_idx], cmap='viridis')
        plt.title(f'Structure Slice {slice_idx}')
        plt.colorbar()
        
        # แสดง True Dose
        plt.subplot(2, 2, 3)
        plt.imshow(sample['true_dose'][slice_idx], cmap='hot')
        plt.title(f'True Dose Slice {slice_idx}')
        plt.colorbar()
        
        # แสดง Predicted Dose
        plt.subplot(2, 2, 4)
        plt.imshow(sample['pred_dose'][slice_idx], cmap='hot')
        plt.title(f'Predicted Dose Slice {slice_idx}')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sample_prediction_{idx+1}.png')
        plt.close()

def test_model(model, test_loader, criterion, device="cuda"):
    """
    Test the model on the test dataset and calculate metrics.
    
    Args:
        model (nn.Module): The trained model to test.
        test_loader (DataLoader): Test DataLoader.
        criterion (callable): Loss function.
        device (str): Device to use for computation (cuda or cpu).
        
    Returns:
        dict: Dictionary containing test metrics.
    """
    model.eval()
    test_loss = 0
    psnr_total = 0
    mae_list = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Calculate metrics
            psnr_total += compute_psnr(outputs, targets)
            mae = torch.mean(torch.abs(outputs - targets)).item()
            mae_list.append(mae)
            
    # Calculate average metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_psnr = psnr_total / len(test_loader)
    avg_mae = np.mean(mae_list)
    mae_std = np.std(mae_list)
    
    # Print results
    print("\n=== Test Results ===")
    print(f"Loss: {avg_test_loss:.4f}")
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"MAE: {avg_mae:.4f} ± {mae_std:.4f}")
    
    return {
        "loss": avg_test_loss,
        "psnr": avg_psnr,
        "mae": avg_mae,
        "mae_std": mae_std
    }

def train_model(
    model_class,
    data_loaders,
    criterion,
    optimizer_class,
    scheduler_class,
    num_epochs,
    device="cuda",
    patience=10,
    save_model_path="best_model.pth",
    use_mixed_precision=True
):
    """
    ฝึกโมเดลและประเมินผลบน validation set
    
    Args:
        model_class: คลาสของโมเดล
        data_loaders: dictionary ที่มี dataloaders สำหรับ train, val
        criterion: loss function
        optimizer_class: คลาสของ optimizer
        scheduler_class: คลาสของ scheduler
        num_epochs: จำนวน epochs สูงสุด
        device: อุปกรณ์ที่ใช้ (cuda หรือ cpu)
        patience: จำนวน epochs ที่รอก่อนหยุด early stopping
        save_model_path: path สำหรับบันทึกโมเดลที่ดีที่สุด
        use_mixed_precision: ใช้ mixed precision training หรือไม่
        
    Returns:
        model: โมเดลที่ฝึกแล้ว
        history: ประวัติการฝึก
    """
    # แยก dataloaders
    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    
    # สร้างโมเดล
    model = model_class().to(device)
    
    # สร้าง optimizer และ scheduler
    optimizer = optimizer_class(model.parameters())
    scheduler = scheduler_class(optimizer)
    
    # ตั้งค่า mixed precision training
    scaler = GradScaler() if use_mixed_precision else None
    
    # ตัวแปรสำหรับ early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    history = []
    
    # สร้างโฟลเดอร์สำหรับเก็บโมเดล
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    
    # ลูปฝึกโมเดล
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # ขั้นตอนการฝึก
        model.train()
        train_loss = 0
        train_mae_list = []
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                if use_mixed_precision:
                    # Forward pass with mixed precision
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Standard backward pass
                    loss.backward()
                    optimizer.step()
                
                # เก็บค่า metrics
                train_loss += loss.item()
                mae = torch.mean(torch.abs(outputs - targets)).item()
                train_mae_list.append(mae)
                
                # อัพเดท progress bar
                pbar.set_postfix({"Loss": loss.item(), "MAE": mae})
                pbar.update(1)
        
        # คำนวณค่าเฉลี่ย metrics
        train_loss /= len(train_loader)
        train_mae_mean = np.mean(train_mae_list)
        train_mae_std = np.std(train_mae_list)
        
        # ขั้นตอนการ validation
        val_loss, val_psnr, val_mae_list = validate_model_with_mae_list(
            model, val_loader, criterion, device
        )
        val_mae_mean = np.mean(val_mae_list)
        val_mae_std = np.std(val_mae_list)
        
        # อัพเดท scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # คำนวณเวลาที่ใช้ไป
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        # แสดงผลลัพธ์ของ epoch
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss = {train_loss:.4f}, Train MAE = {train_mae_mean:.4f} ± {train_mae_std:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val MAE = {val_mae_mean:.4f} ± {val_mae_std:.4f}, "
              f"PSNR = {val_psnr:.2f}, "
              f"Time = {epoch_time:.1f}s, Total = {total_time/60:.1f}m")
        
        # เก็บประวัติ
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_mae_mean": train_mae_mean,
            "train_mae_std": train_mae_std,
            "val_loss": val_loss,
            "val_mae_mean": val_mae_mean,
            "val_mae_std": val_mae_std,
            "val_psnr": val_psnr,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch_time": epoch_time,
            "total_time": total_time
        })
        
        # ตรวจสอบว่าโมเดลนี้ดีที่สุดหรือไม่
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # บันทึกโมเดลที่ดีที่สุด
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'val_mae': val_mae_mean,
                'val_psnr': val_psnr
            }, save_model_path)
            
            print(f"Model saved at Epoch {epoch+1} with Val Loss: {val_loss:.4f}")
        else:
            early_stop_counter += 1
            
        # ตรวจสอบ early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping at Epoch {epoch+1}")
            break
    
    # บันทึกประวัติการฝึกเป็น CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv("training_history.csv", index=False)
    
    # สร้างกราฟแสดงประวัติการฝึก
    plot_training_history(history_df)
    
    # โหลดโมเดลที่ดีที่สุดก่อนส่งคืน
    checkpoint = torch.load(save_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {checkpoint['epoch']+1}")
    
    return model, history_df

#################################
# 5. ส่วนการเทรนและทดสอบโมเดล
#################################

def main():
    # กำหนด paths
    train_dir = "path/to/train_folder"  # โฟลเดอร์ที่มีข้อมูล train+val รวมกัน
    test_dir = "path/to/test_folder"
    csv_path = "path/to/metadata.csv"
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_model_path = os.path.join(save_dir, "best_model.pth")

    # ตั้งค่า random seed เพื่อความสามารถในการทำซ้ำ
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ตรวจสอบอุปกรณ์
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # โหลดและแบ่งข้อมูล
    data_dict = load_data_and_split(train_dir, test_dir, val_ratio=0.2, random_seed=random_seed)

    # ตรวจสอบความถูกต้องของข้อมูล
    train_valid_indices = verify_dataset(data_dict, "train")
    val_valid_indices = verify_dataset(data_dict, "val")
    test_valid_indices = verify_dataset(data_dict, "test")

    # สร้าง datasets และ dataloaders
    data = create_datasets_and_loaders(data_dict, csv_path, batch_size=2, normalize=True)

    # กำหนด optimizer และ scheduler
    optimizer_class = lambda params: optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    
    # OneCycleLR
    steps_per_epoch = len(data["loaders"]["train"])
    total_steps = 100 * steps_per_epoch  # 100 epochs
    scheduler_class = lambda optimizer: torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3, 
        total_steps=total_steps,
        pct_start=0.3,  # warm-up นาน 30% ของ total steps
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1000  # final_lr = initial_lr/1000
    )

    # Loss function แบบผสม (SmoothL1Loss + MSELoss)
    class CombinedLoss(nn.Module):
        def __init__(self, alpha=0.7):
            super(CombinedLoss, self).__init__()
            self.alpha = alpha
            self.smooth_l1 = nn.SmoothL1Loss()
            self.mse = nn.MSELoss()
            
        def forward(self, pred, target):
            return self.alpha * self.smooth_l1(pred, target) + (1 - self.alpha) * self.mse(pred, target)
    
    criterion = CombinedLoss(alpha=0.7)

    # ฝึกโมเดล
    model, history = train_model(
        model_class=Cascade3DUNet,
        data_loaders={
            "train": data["loaders"]["train"],
            "val": data["loaders"]["val"]
        },
        criterion=criterion,
        optimizer_class=optimizer_class,
        scheduler_class=scheduler_class,
        num_epochs=100,
        device=device,
        patience=15,
        save_model_path=save_model_path,
        use_mixed_precision=True
    )

    # ทดสอบโมเดลบน test set
    test_results = test_model(
        model=model,
        test_loader=data["loaders"]["test"],
        criterion=criterion,
        device=device
    )

    # บันทึกผลลัพธ์การทดสอบ
    with open(os.path.join(save_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    # แสดงตัวอย่างผลการทำนาย
    visualize_predictions(
        model=model,
        test_loader=data["loaders"]["test"],
        num_samples=5,
        device=device,
        save_dir=os.path.join(save_dir, "predictions")
    )

    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()