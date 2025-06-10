# === generate_augmented_audio.py ===
import os
import sys
import argparse
import soundfile as sf
import torchaudio
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import random
from tqdm import tqdm
from pathlib import Path
import re
import shutil

# --- Add project root to path ---
try:
    file = Path(__file__).resolve()
    root = file.parent
    sys.path.append(str(root))
    os.chdir(root)
    #print(f"[DEBUG] Project Root identified as: {root}")
    #print(f"[DEBUG] Current Working Directory set to: {os.getcwd()}")
    #print(f"[DEBUG] Added to sys.path: {str(root)}")

    # --- Import Audio Loader ---
    from audio.audio_utils import get_waveform
    #print("[DEBUG] Successfully imported 'get_waveform' from 'audio.audio_utils'.")

except ImportError:
    print("[ERROR] Could not import 'get_waveform' from 'audio.audio_utils'.")
    print("        Ensure an 'audio' directory exists in the project root")
    print("        containing '__init__.py' and 'audio_utils.py' with the function.")
    print("        Alternatively, define the audio loading logic within this script.")
    def get_waveform(path, normalization=False):
         print(f"[WARN] Using dummy get_waveform for path: {path}. Audio loading will likely fail.")
         try:
             waveform, sr = torchaudio.load(path, normalize=normalization)
             return waveform.numpy(), sr
         except Exception as e:
             print(f"[ERROR] Dummy get_waveform failed during basic load attempt: {e}")
    # sys.exit("[FATAL] Audio loader 'get_waveform' is required. Exiting.")
except Exception as e:
     print(f"[ERROR] Failed during path setup or import: {e}")
     sys.exit(1) # Exit if basic setup fails
# ---------------------------------

#print(f"[DEBUG] TorchAudio backend: {torchaudio.get_audio_backend()}")

# --- Augmentation Functions ---

def augment_time_shift(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Applies random time shift (circular)."""
    func_name = "augment_time_shift"
    #print(f"[DEBUG] Running {func_name}...")
    try:
        if waveform.ndim == 1:
             waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1 and waveform.shape[1] == 1: # Handle [time, 1]
             waveform = waveform.T # Transpose to [1, time]

        if waveform.shape[-1] <= 1:
             print(f"[WARN] {func_name}: Waveform too short for time shift. Length: {waveform.shape[-1]}. Returning original.")
             return waveform.squeeze(0) # Return original shape

        shift_max = waveform.shape[-1]
        shift_amt = random.randint(0, shift_max - 1) # Ensure shift is within bounds
        #print(f"[DEBUG] {func_name}: Shifting by {shift_amt} samples out of {shift_max}.")
        shifted_waveform = torch.roll(waveform, shifts=shift_amt, dims=-1)
        #print(f"[DEBUG] {func_name} finished.")
        return shifted_waveform.squeeze(0) if waveform.ndim == 1 else shifted_waveform
    except Exception as e:
        print(f"[ERROR] {func_name} failed: {e}. Returning original waveform.")
        return waveform.squeeze(0) if waveform.ndim == 1 else waveform # Try to return original shape


def augment_band_pass(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Applies band-pass filter (500-1500 Hz) using separate high/low pass."""
    func_name = "augment_band_pass"
    #print(f"[DEBUG] Running {func_name}...")
    low_cutoff = 500
    high_cutoff = 1500
    try:
        original_dtype = waveform.dtype
        if waveform.dtype != torch.float32 and waveform.dtype != torch.float64:
            #print(f"[DEBUG] {func_name}: Converting waveform from {waveform.dtype} to float32 for filtering.")
            waveform = waveform.float()
        # Apply high-pass first
        #print(f"[DEBUG] {func_name}: Applying highpass filter at {low_cutoff} Hz.")
        filtered_low = torchaudio.functional.highpass_biquad(waveform, sample_rate, low_cutoff)
        # Then apply low-pass to the result
        #print(f"[DEBUG] {func_name}: Applying lowpass filter at {high_cutoff} Hz.")
        filtered_high = torchaudio.functional.lowpass_biquad(filtered_low, sample_rate, high_cutoff)
        #print(f"[DEBUG] {func_name} finished.")
        # return filtered_high.to(original_dtype)
        return filtered_high
    except Exception as e:
        print(f"[ERROR] {func_name} failed: {e}. Returning original waveform.")
        return waveform


def augment_pitch_change(waveform: torch.Tensor, sample_rate: int, n_steps_range=(-2, 2)) -> torch.Tensor:
    """
    Applies pitch shifting by a random number of steps using torchaudio.transforms.PitchShift.
    """
    func_name = "augment_pitch_change"
    #print(f"[DEBUG] Running {func_name}...")
    try:
        original_dtype = waveform.dtype
        if waveform.dtype != torch.float32 and waveform.dtype != torch.float64:
            #print(f"[DEBUG] {func_name}: Converting waveform from {waveform.dtype} to float32 for pitch shift.")
            waveform = waveform.float()

        n_steps = random.uniform(n_steps_range[0], n_steps_range[1])
        #print(f"[DEBUG] {func_name}: Shifting pitch by {n_steps:.2f} steps.")

        pitch_shifter = torchaudio.transforms.PitchShift(sample_rate=sample_rate, n_steps=n_steps)
        shifted_waveform = pitch_shifter(waveform)
        #print(f"[DEBUG] {func_name} finished.")
        # return shifted_waveform.to(original_dtype)
        return shifted_waveform
    except Exception as e:
        print(f"[ERROR] {func_name} failed: {e}. Returning original waveform.")
        return waveform


def augment_time_stretch(waveform: torch.Tensor, sample_rate: int, rate_range=(0.0, 0.0)) -> torch.Tensor:
    """Applies time stretching (speed up/slow down) using torchaudio.functional.speed."""
    func_name = "augment_time_stretch"
    #print(f"[DEBUG] Running {func_name} with rate range {rate_range}...")
    try:
        original_dtype = waveform.dtype
        if waveform.dtype != torch.float32 and waveform.dtype != torch.float64:
             #print(f"[DEBUG] {func_name}: Converting waveform from {waveform.dtype} to float32 for time stretch.")
             waveform = waveform.float()

        if rate_range[0] == rate_range[1] or rate_range[0] > rate_range[1]:
             print(f"[WARN] {func_name}: Invalid rate range {rate_range}. Returning original.")
             return waveform
        rate = random.uniform(rate_range[0], rate_range[1])
        if rate <= 0:
             print(f"[WARN] {func_name}: Invalid stretch rate {rate}. Returning original.")
             return waveform

        #print(f"[DEBUG] {func_name}: Stretching time with rate {rate:.2f}.")
        stretched_waveform, _ = torchaudio.functional.speed(waveform, sample_rate, factor=rate)
        #print(f"[DEBUG] {func_name} finished. New length: {stretched_waveform.shape[-1]}")
        # return stretched_waveform.to(original_dtype)
        return stretched_waveform
    except Exception as e:
        # Add rate info to error
        print(f"[ERROR] {func_name} failed with rate {rate if 'rate' in locals() else 'N/A'}: {e}. Returning original waveform.")
        return waveform


def generate_colored_noise(beta: float, n_samples: int, sample_rate: int, device='cpu') -> torch.Tensor:
    """Generates colored noise with power spectrum 1/f^beta using FFT filtering."""
    func_name = "generate_colored_noise"
    try:
        #print(f"[DEBUG] {func_name}: Generating noise, beta={beta:.2f}, samples={n_samples}, sr={sample_rate}")
        white_noise = torch.randn(n_samples, device=device)
        noise_fft = torch.fft.rfft(white_noise)
        freqs = torch.fft.rfftfreq(n_samples, d=1.0/sample_rate, device=device) # Use passed sample rate

        # Create filter: amplitude ~ 1 / f^(beta/2)
        filter_gain = torch.ones_like(freqs)
        # Avoid division by zero at f=0, set filter gain to 1
        non_zero_freqs_idx = freqs != 0
        # Add a small epsilon to prevent potential log/power issues with very small freqs if beta is large
        freqs_safe = torch.clamp(torch.abs(freqs[non_zero_freqs_idx]), min=1e-9)
        filter_gain[non_zero_freqs_idx] = 1.0 / (freqs_safe ** (beta / 2.0))
        # Handle potential NaN/Inf if beta or freqs cause issues
        filter_gain = torch.nan_to_num(filter_gain, nan=1.0, posinf=1.0, neginf=1.0)

        #print(f"[DEBUG] {func_name}: Applying filter in frequency domain.")
        filtered_fft = noise_fft * filter_gain
        colored_noise = torch.fft.irfft(filtered_fft, n=n_samples)

        noise_rms = torch.sqrt(torch.mean(colored_noise**2))
        if noise_rms > 1e-9:
             target_rms = torch.sqrt(torch.mean(white_noise**2)) # RMS of original N(0,1) noise
             colored_noise = colored_noise * (target_rms / noise_rms)
        else:
             print(f"[WARN] {func_name}: Generated noise RMS is near zero. Skipping normalization.")

        #print(f"[DEBUG] {func_name} finished generation.")
        return colored_noise
    except Exception as e:
        print(f"[ERROR] {func_name} failed: {e}")
        return torch.zeros(n_samples, device=device) # Return zeros on failure


def augment_add_colored_noise(waveform: torch.Tensor, sample_rate: int, beta_range=(0.5, 2.0)) -> torch.Tensor:
    """Adds colored noise with a random beta."""
    func_name = "augment_add_colored_noise"
    #print(f"[DEBUG] Running {func_name}...")
    try:
        original_dtype = waveform.dtype
        if waveform.dtype != torch.float32 and waveform.dtype != torch.float64:
            #print(f"[DEBUG] {func_name}: Converting waveform from {waveform.dtype} to float32.")
            waveform = waveform.float()

        beta = random.uniform(beta_range[0], beta_range[1])
        n_samples = waveform.shape[-1]
        #print(f"[DEBUG] {func_name}: Generating colored noise with beta={beta:.2f} for {n_samples} samples.")

        # Generate noise on the same device as the waveform, pass sample_rate
        colored_noise = generate_colored_noise(beta, n_samples, sample_rate, device=waveform.device)

        signal_rms = torch.sqrt(torch.mean(waveform**2))
        noise_rms = torch.sqrt(torch.mean(colored_noise**2))
        target_snr_db = random.uniform(15, 30)

        if noise_rms > 1e-9 and signal_rms > 1e-9:
             # Calculate required noise scaling factor for target SNR
             required_noise_rms = signal_rms / (10**(target_snr_db / 20.0))
             scaling_factor = required_noise_rms / noise_rms
             #print(f"[DEBUG] {func_name}: Scaling noise by {scaling_factor:.4f} for target SNR ~{target_snr_db:.1f} dB.")
             noisy_waveform = waveform + colored_noise * scaling_factor
        else:
             print(f"[WARN] {func_name}: Signal or noise RMS near zero. Adding noise without scaling.")
             noisy_waveform = waveform + colored_noise # Add directly if RMS calculation fails

        #print(f"[DEBUG] {func_name} finished.")
        # return noisy_waveform.to(original_dtype)
        return noisy_waveform
    except Exception as e:
        print(f"[ERROR] {func_name} failed: {e}. Returning original waveform.")
        return waveform


# --- File Processor ---

def process_file_single_aug(input_path, output_base_dir, fold_num, speaker_id, label, spk_idx, aug_name, aug_func):
    """Loads audio, applies a SINGLE augmentation, saves it, returns path info."""
    # print(f"-- Attempting {aug_name} for: {input_path}") # Optional: detailed logging
    try:
        # Load original waveform, normalize to [-1, 1]
        waveform_np, sr = get_waveform(input_path, normalization=True)
        if waveform_np is None or sr is None:
            print(f"[WARN] Skipping {input_path} for {aug_name} due to loading error.")
            return None
        # Ensure waveform is 2D [channel, time] for consistency
        if waveform_np.ndim == 1:
            waveform_np = waveform_np[np.newaxis, :] # Add channel dim

        waveform = torch.from_numpy(waveform_np) # Shape [1, time]

    except Exception as e:
        print(f"[ERROR] Error loading {input_path} for {aug_name}: {e}. Skipping.")
        return None

    original_filename = Path(input_path).stem
    output_base_dir_path = Path(output_base_dir)
    output_fold_dir = output_base_dir_path / f"fold_{fold_num}" / speaker_id
    output_fold_dir.mkdir(parents=True, exist_ok=True)

    try:
        augmented_waveform = aug_func(waveform, sr)

        # Ensure output is squeezed correctly if needed, but keep channel dim for saving
        if augmented_waveform.ndim == 1:
            augmented_waveform = augmented_waveform.unsqueeze(0) # Ensure [1, time]

        output_filename = f"{original_filename}_{aug_name}.wav"
        output_path = output_fold_dir / output_filename # Use Path object joining

        augmented_waveform_np = augmented_waveform.detach().cpu().squeeze(0).numpy() # Squeeze channel -> [time]

        sf.write(str(output_path), augmented_waveform_np, sr, subtype='FLOAT')

        project_root_path = Path(root).resolve()
        absolute_output_path = output_path.resolve()
        rel_output_path = absolute_output_path.relative_to(project_root_path)

        return {
            "ID": spk_idx,
            "file_path": str(rel_output_path), # Return path as string
            "label": label,
            "spk_ID": speaker_id,
            "augmentation": aug_name
        }
    except Exception as e:
        print(f"[ERROR] Error augmenting/saving {aug_name} for {input_path}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate augmented audio files matching paper descriptions for PC-GITA Vowels training sets.")
    parser.add_argument(
        "--input_folds_dir",
        type=str,
        default="preprocess/PC-GITA_baseline_vowels/processed_output/folds",
        help="Directory containing the original fold CSV files (e.g., train_foldX_online.csv)."
    )
    parser.add_argument(
        "--output_audio_dir",
        type=str,
        default="preprocess/pc_gita_vowels_augmented/audio",
        help="Base directory to save the generated augmented audio files."
    )
    # --- Default metadata output path ---
    parser.add_argument(
        "--output_metadata_dir",
        type=str,
        default="preprocess/pc_gita_vowels_augmented/folds",
        help="Directory to save the metadata CSV of generated files."
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1-10",
        help="Range of folds to process (e.g., '1-10', '1,3,5', '2')."
    )
    parser.add_argument(
        "--njobs",
        default=max(1, os.cpu_count() // 2), # Default to half available cores
        type=int,
        help="Number of parallel jobs for processing."
    )
    parser.add_argument(
        "--init",
        action="store_true",
        default=False,
        help="If specified, removes the existing output audio AND metadata directories first."
    )

    args = parser.parse_args()

    # --- Define Augmentations ---
    augmentations_to_apply = {
        "tshift": augment_time_shift,
        "bpass": augment_band_pass, # Uses high/low pass for 500-1500 Hz
        #"pitch": augment_pitch_change, # Uses standard pitch shift
        #"slow": lambda w, s: augment_time_stretch(w, s, rate_range=(0.2, 0.8))
        #"fast": lambda w, s: augment_time_stretch(w, s, rate_range=(1.2, 2.5))
        "cnoise": lambda w, s: augment_add_colored_noise(w, s, beta_range=(4.0, 6.0)), # Colored noise, beta [4, 6], added with SNR scaling
    }
    print(f"[INFO] Augmentations to apply: {list(augmentations_to_apply.keys())}")

    # --- Parse Fold Range ---
    folds_to_process = []
    try:
        parts = args.folds.split(',')
        for part in parts:
            if '-' in part:
                start, end = map(int, part.split('-'))
                if start > end:
                     raise ValueError(f"Invalid range: start ({start}) > end ({end})")
                folds_to_process.extend(list(range(start, end + 1)))
            else:
                folds_to_process.append(int(part))
        folds_to_process = sorted(list(set(folds_to_process))) # Unique sorted list
        if not folds_to_process:
             raise ValueError("No folds selected.")
        print(f"[INFO] Processing folds: {folds_to_process}")
    except ValueError as e:
        print(f"[ERROR] Invalid fold specification '{args.folds}': {e}. Use format like '1-10' or '1,3,5'.")
        sys.exit(1)

    # --- Clean Output Dirs ---
    output_audio_dir_path = Path(args.output_audio_dir)
    output_metadata_dir_path = Path(args.output_metadata_dir)
    if args.init:
        if output_audio_dir_path.exists():
            print(f"[INFO] --init flag set. Removing existing output audio directory: {output_audio_dir_path}")
            try:
                shutil.rmtree(output_audio_dir_path)
            except OSError as e:
                print(f"[ERROR] Failed to remove directory {output_audio_dir_path}: {e}")
                print("[WARN] Could not remove existing audio directory. Files might be overwritten or mixed.")
        if output_metadata_dir_path.exists():
             print(f"[INFO] --init flag set. Removing existing output metadata directory: {output_metadata_dir_path}")
             try:
                 shutil.rmtree(output_metadata_dir_path)
             except OSError as e:
                 print(f"[ERROR] Failed to remove directory {output_metadata_dir_path}: {e}")
                 print("[WARN] Could not remove existing metadata directory.")

    try:
        output_audio_dir_path.mkdir(parents=True, exist_ok=True)
        output_metadata_dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Ensured output directories exist:")
        print(f"         Audio: {output_audio_dir_path}")
        print(f"      Metadata: {output_metadata_dir_path}")
    except OSError as e:
        print(f"[ERROR] Failed to create output directories: {e}")
        sys.exit(1)

    # --- Create Tasks ---
    all_tasks = []
    print("[INFO] Gathering files and creating augmentation tasks...")
    project_root_path = Path(root)
    #print(f"[DEBUG] Using project root path for resolving inputs: {project_root_path}")

    input_folds_dir_path = Path(args.input_folds_dir)

    files_processed_count = 0
    files_missing_count = 0
    tasks_created_count = 0

    for fold_num in folds_to_process:
        train_csv_path = input_folds_dir_path / f"train_fold{fold_num}_online.csv"
        #print(f"[INFO] Reading training data for fold {fold_num} from {train_csv_path}")
        if not train_csv_path.is_file():
            print(f"[WARN] Training CSV not found for fold {fold_num}: {train_csv_path}. Skipping fold.")
            continue

        try:
            train_df = pd.read_csv(train_csv_path)
            # Check for essential columns
            required_cols = ['file_path', 'spk_ID', 'label']
            missing_cols = [col for col in required_cols if col not in train_df.columns]
            if missing_cols:
                print(f"[ERROR] Missing required columns in {train_csv_path}: {missing_cols}. Skipping fold.")
                continue
            if 'ID' not in train_df.columns:
                print(f"[WARN] Optional 'ID' column missing in {train_csv_path}. Using 'spk_ID' as fallback index.")


            fold_tasks = 0
            fold_files_processed = 0
            fold_files_missing = 0

            for _, row in train_df.iterrows():
                relative_path = Path(row['file_path']) # Treat as Path object
                absolute_path = project_root_path / relative_path

                if absolute_path.is_file():
                    fold_files_processed += 1
                    # Get speaker index, fallback if 'ID' column missing
                    spk_idx = row.get('ID', row['spk_ID']) # Use ID if present, else spk_ID
                    for aug_name, aug_func in augmentations_to_apply.items():
                        all_tasks.append((
                            str(absolute_path),             # Pass path as string
                            str(output_audio_dir_path),     # Pass output dir as string
                            fold_num,
                            row['spk_ID'],
                            row['label'],
                            spk_idx,                        # Use resolved speaker index
                            aug_name,
                            aug_func
                        ))
                        fold_tasks += 1
                else:
                    #print(f"[WARN] Input file not found: {absolute_path} (referenced in {train_csv_path} as '{relative_path}'). Skipping.")
                    fold_files_missing += 1

            if fold_files_missing > 0:
                 print(f"[INFO] Fold {fold_num}: Found {fold_files_processed} existing files, {fold_files_missing} missing files referenced in CSV. Created {fold_tasks} tasks.")
            else:
                 print(f"[INFO] Fold {fold_num}: Found {fold_files_processed} existing files. Created {fold_tasks} tasks.")

            files_processed_count += fold_files_processed
            files_missing_count += fold_files_missing
            tasks_created_count += fold_tasks

        except FileNotFoundError:
             print(f"[ERROR] CSV file itself not found: {train_csv_path}")
             continue # Skip to next fold
        except pd.errors.EmptyDataError:
             print(f"[WARN] CSV file is empty: {train_csv_path}")
             continue
        except KeyError as e:
             print(f"[ERROR] Missing expected column during row processing in {train_csv_path}: {e}.")
             continue # Skip fold potentially
        except Exception as e:
            print(f"[ERROR] Error reading or processing CSV for fold {fold_num} ({train_csv_path}): {e}")
            continue

    if not all_tasks:
        print(f"[ERROR] No valid audio files found or no tasks created across specified folds ({folds_to_process}). Check input paths and CSV files.")
        print(f"[INFO] Total existing files checked: {files_processed_count}")
        print(f"[INFO] Total missing files referenced: {files_missing_count}")
        sys.exit(1)

    print(f"[INFO] Created {len(all_tasks)} total augmentation tasks ({len(augmentations_to_apply)} augmentations for {files_processed_count} original training files across {len(folds_to_process)} folds).")
    if files_missing_count > 0:
        print(f"[WARN] Note: {files_missing_count} files referenced in the CSVs were not found on disk and were skipped.")
    print(f"[INFO] Starting parallel augmentation generation with {args.njobs} workers...")

    # --- Run Parallel Processing ---
    results_list = Parallel(n_jobs=args.njobs)(
        delayed(process_file_single_aug)(*task) # Unpack tuple arguments
        for task in tqdm(all_tasks, desc="Generating Augmentations")
    )

    # --- Summarize Results ---
    successful_results = [result for result in results_list if result is not None]
    failed_count = len(all_tasks) - len(successful_results)

    print("-" * 30)
    print("[INFO] Augmentation Generation Summary:")
    print(f"  Total tasks scheduled: {len(all_tasks)}")
    print(f"  Successfully generated files: {len(successful_results)}")
    print(f"  Failed tasks (errors during load/aug/save): {failed_count}")

    if not successful_results:
        print("[WARN] No augmented files were successfully generated. Check error messages above.")
    else:
         print(f"[INFO] Augmented audio files saved in subfolders under: {args.output_audio_dir}")

    # --- Save Metadata ---
    if successful_results:
        metadata_df = pd.DataFrame(successful_results)
        # Use the dedicated metadata directory path
        metadata_filename = output_metadata_dir_path / "augmented_files_metadata.csv"
        try:
            metadata_df.to_csv(metadata_filename, index=False)
            print(f"[INFO] Saved metadata of generated files to: {metadata_filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save metadata file '{metadata_filename}': {e}")

    print("-" * 30)
    print("[INFO] Augmentation generation finished.")