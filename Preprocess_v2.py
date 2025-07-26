import os
import numpy as np
import skimage.transform
import skimage.exposure
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

def load_numpy_file(file_path):
    """
    Load a numpy file and return its array and shape.
    """
    if os.path.exists(file_path):
        array = np.load(file_path)
        print(f"[INFO] Loaded file: {file_path}, Shape: {array.shape}")
        return array, array.shape
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def clip_ct_to_hu_range(ct_array, min_hu=-1000, max_hu=2000):
    """
    Clip CT array values to the specified HU range.

    Args:
        ct_array (numpy.ndarray): CT image array.
        min_hu (int): Minimum HU value for clipping (default: -1000).
        max_hu (int): Maximum HU value for clipping (default: 2000).

    Returns:
        numpy.ndarray: CT array with values clipped to [min_hu, max_hu].
    """
    # Clip values to the range [min_hu, max_hu]
    clipped_ct = np.clip(ct_array, min_hu, max_hu)
    print(f"[INFO] CT clipped to range [{min_hu}, {max_hu}]")
    return clipped_ct

def rescale_ct_to_range(ct_array, min_hu=-1000, max_hu=2000):
    """
    Rescale CT array values to the range [0, 1].

    Args:
        ct_array (numpy.ndarray): CT image array (clipped HU values).
        min_hu (int): Minimum HU value used for clipping.
        max_hu (int): Maximum HU value used for clipping.

    Returns:
        numpy.ndarray: CT array rescaled to the range [0, 1].
    """
    rescaled_ct = (ct_array - min_hu) / (max_hu - min_hu)
    print(f"[INFO] CT rescaled to range [0, 1]")
    return rescaled_ct

def advanced_preprocessing(ct_array, dose_array, mask_array, 
                           min_hu=-1000, max_hu=2000, do_augmentation=False):
    """
    Advanced preprocessing for medical imaging data.
    Note: This version assumes arrays already have the same shape from DICOM processing.
    
    Args:
        ct_array (np.ndarray): CT volume
        dose_array (np.ndarray): Dose volume
        mask_array (np.ndarray): Mask volume
        min_hu (int): Minimum HU value for clipping
        max_hu (int): Maximum HU value for clipping
        do_augmentation (bool): Whether to apply data augmentation
    
    Returns:
        tuple: Preprocessed CT, Dose, and Mask arrays
    """
    print("[INFO] Starting advanced preprocessing pipeline")
    print(f"[INFO] Input shapes: CT {ct_array.shape}, Dose {dose_array.shape}, Mask {mask_array.shape}")
    
    # 1. Clip CT values to specified HU range
    ct_clipped = clip_ct_to_hu_range(ct_array, min_hu, max_hu)
    
    # 2. Rescale CT to [0, 1]
    ct_rescaled = rescale_ct_to_range(ct_clipped, min_hu, max_hu)
    
    # 3. Ensure binary mask
    mask_binary = (mask_array > 0.5).astype(np.float32)
    
    # 4. Optional data augmentation
    if do_augmentation:
        print("[INFO] Applying data augmentation")
        if np.random.random() > 0.5:
            # Random rotation
            angle = np.random.uniform(-15, 15)
            ct_rescaled = rotate(ct_rescaled, angle, reshape=False)
            dose_array = rotate(dose_array, angle, reshape=False)
            mask_binary = rotate(mask_binary, angle, reshape=False)
        
        if np.random.random() > 0.5:
            # Random flip
            flip_axis = np.random.choice([0, 1, 2])
            ct_rescaled = np.flip(ct_rescaled, axis=flip_axis)
            dose_array = np.flip(dose_array, axis=flip_axis)
            mask_binary = np.flip(mask_binary, axis=flip_axis)
    
    # 5. Normalize dose to [0, 1] if not already normalized
    if np.max(dose_array) > 1.0 or np.min(dose_array) < 0.0:
        if dose_array.max() != dose_array.min():
            dose_normalized = (dose_array - dose_array.min()) / (dose_array.max() - dose_array.min())
        else:
            dose_normalized = dose_array
            print("[WARNING] Dose array has constant values, could not normalize properly")
    else:
        dose_normalized = dose_array
        print("[INFO] Dose array already normalized, skipping normalization")
    
    # 6. Final mask cleaning - ensure binary after all operations
    mask_final = (mask_binary > 0.5).astype(np.float32)
    
    print("[INFO] Advanced preprocessing completed")
    print(f"[INFO] Output shapes: CT {ct_rescaled.shape}, Dose {dose_normalized.shape}, Mask {mask_final.shape}")
    return ct_rescaled, dose_normalized, mask_final

def visualize_preprocessing_results(ct_orig, dose_orig, mask_orig, 
                                    ct_proc, dose_proc, mask_proc, 
                                    patient_id, slice_idx=None, save_path=None):
    """
    Visualize preprocessing results.
    
    Args:
        ct_orig (np.ndarray): Original CT array
        dose_orig (np.ndarray): Original dose array
        mask_orig (np.ndarray): Original mask array
        ct_proc (np.ndarray): Processed CT array
        dose_proc (np.ndarray): Processed dose array
        mask_proc (np.ndarray): Processed mask array
        patient_id (str): Patient ID for display
        slice_idx (int): Slice index to display, if None takes middle slice
        save_path (str): Path to save the visualization, if None just displays it
    """
    # For original arrays, select middle slice if none specified
    if slice_idx is None:
        slice_idx_orig = ct_orig.shape[0] // 2
    else:
        slice_idx_orig = min(slice_idx, ct_orig.shape[0] - 1)
    
    # For processed arrays (which may have different dimensions)
    slice_idx_proc = ct_proc.shape[0] // 2 if slice_idx is None else min(slice_idx, ct_proc.shape[0] - 1)
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Preprocessing Results for Patient {patient_id}", fontsize=16)
    
    titles = ['CT', 'Dose', 'Mask']
    
    # Original images in first row
    axs[0, 0].imshow(ct_orig[slice_idx_orig], cmap='gray')
    axs[0, 0].set_title(f'Original {titles[0]} - Slice {slice_idx_orig}')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(dose_orig[slice_idx_orig], cmap='hot')
    axs[0, 1].set_title(f'Original {titles[1]} - Slice {slice_idx_orig}')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(mask_orig[slice_idx_orig], cmap='binary')
    axs[0, 2].set_title(f'Original {titles[2]} - Slice {slice_idx_orig}')
    axs[0, 2].axis('off')
    
    # Processed images in second row
    axs[1, 0].imshow(ct_proc[slice_idx_proc], cmap='gray')
    axs[1, 0].set_title(f'Processed {titles[0]} - Slice {slice_idx_proc}')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(dose_proc[slice_idx_proc], cmap='hot')
    axs[1, 1].set_title(f'Processed {titles[1]} - Slice {slice_idx_proc}')
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(mask_proc[slice_idx_proc], cmap='binary')
    axs[1, 2].set_title(f'Processed {titles[2]} - Slice {slice_idx_proc}')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{patient_id}_comparison.png")
        plt.savefig(save_file)
        print(f"[INFO] Visualization saved to {save_file}")
    
    plt.show()

def check_sizes_match(ct_array, dose_array, mask_array):
    """
    Check if all arrays have the same shape.
    
    Args:
        ct_array (np.ndarray): CT volume
        dose_array (np.ndarray): Dose volume
        mask_array (np.ndarray): Mask volume
        
    Returns:
        bool: True if all arrays have the same shape, False otherwise
    """
    return (ct_array.shape == dose_array.shape == mask_array.shape)

def unified_preprocessing_pipeline(input_path, output_path, 
                                   min_hu=-1000, max_hu=2000,
                                   visualize=True, 
                                   do_augmentation=False):
    """
    Unified preprocessing pipeline for CT, dose, and mask data.
    This version assumes data already has proper padding from DICOM processing.
    
    Args:
        input_path (str): Path to the folder containing patient data
        output_path (str): Path to save processed data
        min_hu (int): Minimum HU value for CT clipping
        max_hu (int): Maximum HU value for CT clipping
        visualize (bool): Whether to visualize results
        do_augmentation (bool): Whether to apply data augmentation
    """
    print(f"[INFO] Starting unified preprocessing pipeline")
    print(f"[INFO] Input path: {input_path}")
    print(f"[INFO] Output path: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    patient_folders = [f for f in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, f))]
    print(f"[INFO] Found {len(patient_folders)} patient folders")
    
    for patient_folder in patient_folders:
        patient_path = os.path.join(input_path, patient_folder)
        print(f"\n[INFO] Processing patient: {patient_folder}")
        
        try:
            # Create patient output directory
            patient_output_path = os.path.join(output_path, patient_folder)
            os.makedirs(patient_output_path, exist_ok=True)
            
            # Find required files
            ct_file = os.path.join(patient_path, "CT_volume.npy")
            
            # Find mask file
            mask_file = os.path.join(patient_path, "Combined_Mask.npy")
            if not os.path.exists(mask_file):
                # Look for alternative mask files
                mask_candidates = [f for f in os.listdir(patient_path) if "mask" in f.lower() and f.endswith(".npy")]
                if mask_candidates:
                    mask_file = os.path.join(patient_path, mask_candidates[0])
                    print(f"[INFO] Using alternative mask file: {mask_file}")
            
            # Find dose file
            dose_file = os.path.join(patient_path, "Dose.npy")
            if not os.path.exists(dose_file):
                # Look for alternative dose files
                dose_candidates = [f for f in os.listdir(patient_path) if "dose" in f.lower() and f.endswith(".npy")]
                if dose_candidates:
                    dose_file = os.path.join(patient_path, dose_candidates[0])
                    print(f"[INFO] Using alternative dose file: {dose_file}")
            
            # Check if all required files exist
            if not os.path.exists(ct_file):
                print(f"[WARNING] CT file not found for {patient_folder}, skipping...")
                continue
                
            # Load files
            ct_array, ct_shape = load_numpy_file(ct_file)
            
            # Handle missing mask or dose files
            if not os.path.exists(mask_file):
                print(f"[WARNING] Mask file not found for {patient_folder}, creating empty mask")
                mask_array = np.zeros_like(ct_array)
            else:
                mask_array, mask_shape = load_numpy_file(mask_file)
                
            if not os.path.exists(dose_file):
                print(f"[WARNING] Dose file not found for {patient_folder}, creating empty dose")
                dose_array = np.zeros_like(ct_array)
            else:
                dose_array, dose_shape = load_numpy_file(dose_file)
            
            # Check if all arrays have the same shape (should be true after DICOM processing)
            if not check_sizes_match(ct_array, dose_array, mask_array):
                print(f"[WARNING] Array shapes don't match even after DICOM processing.")
                print(f"[WARNING] CT: {ct_array.shape}, Dose: {dose_array.shape}, Mask: {mask_array.shape}")
                print(f"[WARNING] This suggests an issue with the DICOM processing step.")
                continue
            
            # Save original arrays for visualization
            ct_orig = ct_array.copy()
            dose_orig = dose_array.copy()
            mask_orig = mask_array.copy()
            
            # Apply advanced preprocessing (no need for padding as it was done in DICOM processing)
            ct_preprocessed, dose_preprocessed, mask_preprocessed = advanced_preprocessing(
                ct_array, dose_array, mask_array, 
                min_hu=min_hu, max_hu=max_hu,
                do_augmentation=do_augmentation
            )
            
            # Save preprocessed data
            np.save(os.path.join(patient_output_path, "CT_preprocessed.npy"), ct_preprocessed)
            np.save(os.path.join(patient_output_path, "Dose_preprocessed.npy"), dose_preprocessed)
            np.save(os.path.join(patient_output_path, "Mask_preprocessed.npy"), mask_preprocessed)
            print(f"[SUCCESS] Saved preprocessed files for {patient_folder}")
            
            # Visualize results
            if visualize:
                visualize_preprocessing_results(
                    ct_orig, dose_orig, mask_orig,
                    ct_preprocessed, dose_preprocessed, mask_preprocessed,
                    patient_folder,
                    save_path=patient_output_path
                )
                
        except Exception as e:
            print(f"[ERROR] Failed to process {patient_folder}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[INFO] Unified preprocessing pipeline completed")

# Example usage
if __name__ == "__main__":
    # Define input and output paths
    input_path = r"D:\Workhard\preprocessing\DATA\Data_preprocessing\Data_numpy"  # Output from DICOM processing
    output_path = r"D:\Workhard\preprocessing\DATA\Data_preprocessing\Data_numpy_preprocessed"
    
    # Run the unified preprocessing pipeline (assumes data already has proper padding from DICOM processing)
    unified_preprocessing_pipeline(
        input_path=input_path,
        output_path=output_path,
        min_hu=-1000,
        max_hu=2000,
        visualize=True,
        do_augmentation=False  # Set to True if you want augmentation
    )