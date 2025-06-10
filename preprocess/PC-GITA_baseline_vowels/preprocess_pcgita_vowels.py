"""*********************************************************************************************"""
#   Preprocess the Vowels part of the PC-GITA database
#   1) Segmenting: Segment speech materials for each speaker into short utterances.
#   2) Computing/saving features from utterances (parallel processing).
#   3) Making tables (train/test/validation) for wav data (for online feature
#      extraction) and offline feature data (for offline dataloading) using speaker-stratified folds.
#   4) Uses metadata to assign correct speaker IDs and labels (Control=0, Pathological=1).

# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Parvaneh Janbakhshi <parvaneh.janbakhshi@idiap.ch>

# This file is part of pddetection-reps-learning
#
# pddetection-reps-learning is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# pddetection-reps-learning is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pddetection-reps-learning. If not, see <http://www.gnu.org/licenses/>.
"""*********************************************************************************************"""

from pathlib import Path
import sys
from sklearn.model_selection import StratifiedKFold
import soundfile as sf
import os
import re # Import regex
import time # <--- IMPORT TIME MODULE

# ------------------------------ Path change ----------------------------- #
file = Path(__file__).resolve()
parent, root, subroot = file.parent, file.parents[1], file.parents[2]
sys.path.append(str(subroot))
sys.path.append(str(root))
os.chdir(root)
# ------------------------------------------------------------------------- #

from downstream.ds_runner import seed_torch
from audio.audio_utils import get_waveform, get_feat, get_config_args
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import basename
from joblib import Parallel, delayed
import shutil


def segmenting(wav_path, saved_path_dir, MaxSegLen=8):
    """Segmenting long utterances
    Args:
        wav_path (str): path of audio file
        saved_path_dir (str): directory path for saving segmented audio files
        MaxSegLen (int, optional): segmenting length. Defaults to 8.

    Yields:
        (tuple): a tuple containing:
            - (str): path of segmented files
            - (float): length of audio (ms)
    """
    try:
        # --- Optional: Time individual file loading ---
        # start_load = time.time()
        wav, fs = get_waveform(wav_path, normalization=True)
        # end_load = time.time()
        # print(f"    Loading {basename(wav_path)} took {end_load - start_load:.4f}s")
        # --- End Optional Timing ---
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        return # Skip this file if loading fails

    n_samples = len(wav)
    segmentsize = MaxSegLen * fs
    if not os.path.exists(saved_path_dir):
        os.makedirs(saved_path_dir, exist_ok=True)

    # Ensure filename uniqueness if segmenting
    base_name = basename(wav_path).split(Path(wav_path).suffix)[0]
    output_dir = os.path.join(saved_path_dir, base_name) # Save segments in speaker-specific subdirs
    if not os.path.exists(output_dir):
         os.makedirs(output_dir, exist_ok=True)

    # Cast segmentsize to int for range()
    segmentsize_int = int(segmentsize)
    if segmentsize_int <= 0:
        print(f"Warning: Calculated segment size is non-positive ({segmentsize_int}) for {wav_path}. Skipping segmentation.")
        return

    for ind, start in enumerate(range(0, n_samples, segmentsize_int)):
        end = min(start + segmentsize_int, n_samples)
        # Make sure segment is long enough (e.g., at least 1 frame shift)
        # Use original float segmentsize for length check if needed, but int for iteration step
        if (end - start) / fs * 1000 < get_config_args(feat_config_path).get("torchaudio").get("frame_shift", 10.0):
             continue
        new_filename = base_name + "_uttr" + str(ind) + Path(wav_path).suffix
        new_path = os.path.join(
            output_dir, new_filename
        )
        try:
            # --- Optional: Time individual segment writing ---
            # start_write = time.time()
            sf.write(new_path, wav[start:end], fs)
            # end_write = time.time()
            # print(f"    Writing segment {new_filename} took {end_write - start_write:.4f}s")
            # --- End Optional Timing ---
            yield new_path, (end - start) * 1e3 / fs
        except Exception as e:
            print(f"Error writing segment {new_path}: {e}")


def feature_extraction(wav_path, saved_path_dir, feat_config_path):
    """extracting features from audio file
    Args:
        wav_path (str): path of audio file
        saved_path_dir (str): directory path for saving features
        feat_config_path (str): path of feature extraction config file
    Returns:
        (tuple): a tuple containing:
            - (str): path of saved feature file
            - (int): length of feature data (e.g., number of time frames)
            Returns None if feature extraction fails
    """
    try:
        # --- Time feature extraction ---
        start_feat = time.time()
        features = get_feat(wav_path, feat_config_path).T # Transpose to D x T
        end_feat = time.time()
        # --- End feature extraction timing ---
        # You could print this per-file, but it might be too verbose in parallel runs
        # print(f"    Feature extraction for {basename(wav_path)} took {end_feat - start_feat:.4f}s")

        if features is None or features.size == 0:
             print(f"Warning: Feature extraction yielded empty features for {wav_path}")
             return None, 0
    except Exception as e:
        print(f"Error extracting features for {wav_path}: {e}")
        return None, 0

    # Ensure filename uniqueness for features
    base_name = basename(wav_path).split(Path(wav_path).suffix)[0]
    # Extract original speaker ID to create a subdirectory
    speaker_dir_match = re.match(r"(AVPEPUDEA[C]?\d+)", base_name)
    if speaker_dir_match:
        speaker_dir = speaker_dir_match.group(1)
    else:
        # Fallback if pattern doesn't match, though it should based on PC-GITA naming
        speaker_dir = "unknown_speaker"

    output_subdir = os.path.join(saved_path_dir, speaker_dir)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir, exist_ok=True)

    new_filename_base = base_name
    new_path = os.path.join(output_subdir, new_filename_base)
    try:
        # --- Time feature saving ---
        start_save = time.time()
        np.save(new_path, features)
        end_save = time.time()
        # --- End feature saving timing ---
        # print(f"    Saving features for {basename(wav_path)} took {end_save - start_save:.4f}s")
        return new_path + ".npy", features.shape[1]
    except Exception as e:
        print(f"Error saving features for {new_path}: {e}")
        return None, 0


def preprocess_segmenting(
    wav_path,
    ID, # Numerical speaker index
    SPK_ID, # Original speaker ID string (e.g., AVPEPUDEA0001)
    label,
    save_path_seg_wav,
    save_path_feat,
    feat_config_path,
    MaxSegLen=8,
):
    """audio segmenting and feature extraction
    Args:
        wav_path (str): path of wav file
        ID (int): Unique numerical speaker index (0 to N-1)
        SPK_ID (str): Original speaker ID string (e.g., AVPEPUDEA0001)
        label (int): speaker label (0=Control, 1=Pathological)
        save_path_seg_wav (str): path for saving segmented data
        save_path_feat (str): path for saving segmented features
        feat_config_path (str): path of feature extraction config file
        MaxSegLen (int, optional): Audio segmenting length. Defaults to 8.
    """
    # --- Optional: Time the whole function per file ---
    # start_func = time.time()
    # --- End Optional Timing ---

    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    min_wav_length_ms = float(
        get_config_args(feat_config_path).get("torchaudio").get("frame_length")
    )
    # --- Time segmentation part ---
    start_seg_part = time.time()
    segmenting_gen = segmenting(wav_path, save_path_seg_wav, MaxSegLen=MaxSegLen)
    end_seg_part = time.time()
    # print(f"  Segmenting part for {basename(wav_path)} took {end_seg_part - start_seg_part:.4f}s")
    # --- End segmentation timing ---

    df_wav = pd.DataFrame(
        data={
            "ID": [], # Numerical Index
            "file_path": [],
            "length": [],
            "label": [],
            "spk_ID": [], # Original String ID
        }
    )
    df_feat = pd.DataFrame(
        data={
            "ID": [], # Numerical Index
            "file_path": [],
            "length": [],
            "label": [],
            "spk_ID": [], # Original String ID
        }
    )
    if segmenting_gen is None: # Handle potential error in segmenting function
         return df_wav, df_feat

    # --- Time feature extraction loop for segments ---
    start_feat_loop = time.time()
    segment_count = 0
    for new_wav_path, wav_length in segmenting_gen:
        segment_count += 1
        if wav_length >= min_wav_length_ms: # Check segment length
            feat_path, feat_length = feature_extraction(
                new_wav_path, save_path_feat, feat_config_path
            )
            # Skip if feature extraction failed or features are too short
            if feat_path is None or feat_length <= 1: # Need at least 1 frame
                 print(f"Skipping segment {new_wav_path} due to short/failed features (len: {feat_length})")
                 continue

            # Use relative paths from the project root for CSV files
            rel_wav_path = os.path.relpath(new_wav_path, root)
            rel_feat_path = os.path.relpath(feat_path, root)

            df_wav_seg = pd.DataFrame(
                data={
                    "ID": [ID],
                    "file_path": [rel_wav_path],
                    "length": [wav_length], # Wav length in ms
                    "label": [label],
                    "spk_ID": [SPK_ID],
                }
            )
            df_feat_seg = pd.DataFrame(
                data={
                    "ID": [ID],
                    "file_path": [rel_feat_path],
                    "length": [feat_length], # Feature length in frames
                    "label": [label],
                    "spk_ID": [SPK_ID],
                }
            )
            df_wav = pd.concat([df_wav, df_wav_seg], ignore_index=True)
            df_feat = pd.concat([df_feat, df_feat_seg], ignore_index=True)
    end_feat_loop = time.time()
    # if segment_count > 0:
    #     print(f"  Feature extraction loop for {segment_count} segments of {basename(wav_path)} took {end_feat_loop - start_feat_loop:.4f}s")
    # --- End feature extraction loop timing ---

    # --- Optional: End timing the whole function per file ---
    # end_func = time.time()
    # print(f"  Total preprocess_segmenting for {basename(wav_path)} took {end_func - start_func:.4f}s")
    # --- End Optional Timing ---
    return df_wav, df_feat


def preprocess(wav_path, ID, SPK_ID, label, save_path_feat, feat_config_path):
    """audio feature extraction (without segmentation)
    Args:
        wav_path (str): path of wav file
        ID (int): Unique numerical speaker index (0 to N-1)
        SPK_ID (str): Original speaker ID string
        label (int): speaker label (0=Control, 1=Pathological)
        save_path_feat (str): path for saving segmented features
        feat_config_path (str): path of feature extraction config file

    Returns:
        (DataFrame): dataframe including information of feature data
    """
    # --- Optional: Time the whole function per file ---
    # start_func = time.time()
    # --- End Optional Timing ---
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    min_wav_length_ms = float(
        get_config_args(feat_config_path).get("torchaudio").get("frame_length")
    )
    df_feat = pd.DataFrame(
        data={
            "ID": [],
            "file_path": [],
            "length": [],
            "label": [],
            "spk_ID": [],
        }
    )
    try:
        # --- Optional: Time loading ---
        # start_load = time.time()
        wav, fs = get_waveform(wav_path, normalization=True)
        # end_load = time.time()
        # print(f"    Loading {basename(wav_path)} took {end_load - start_load:.4f}s")
        # --- End Optional Timing ---
        wav_length = len(wav) * 1e3 / fs
    except Exception as e:
        print(f"Error loading wav {wav_path}: {e}")
        return df_feat # Return empty dataframe

    new_wav_path = wav_path
    if wav_length >= min_wav_length_ms:
        # --- Time feature extraction part ---
        start_feat_part = time.time()
        feat_path, feat_length = feature_extraction(
            new_wav_path, save_path_feat, feat_config_path
        )
        end_feat_part = time.time()
        # Note: feature_extraction itself has internal timing, this times the call + checks
        # print(f"  Feature extraction call for {basename(wav_path)} took {end_feat_part - start_feat_part:.4f}s")
        # --- End feature extraction timing ---

        # Skip if feature extraction failed or features are too short
        if feat_path is None or feat_length <= 1:
             print(f"Skipping file {new_wav_path} due to short/failed features (len: {feat_length})")
             return df_feat

        # Use relative paths from the project root
        rel_feat_path = os.path.relpath(feat_path, root)

        df_feat_seg = pd.DataFrame(
            data={
                "ID": [ID],
                "file_path": [rel_feat_path],
                "length": [feat_length], # Feature length in frames
                "label": [label],
                "spk_ID": [SPK_ID],
            }
        )
        df_feat = pd.concat([df_feat, df_feat_seg], ignore_index=True)

    # --- Optional: End timing the whole function per file ---
    # end_func = time.time()
    # print(f"  Total preprocess for {basename(wav_path)} took {end_func - start_func:.4f}s")
    # --- End Optional Timing ---
    return df_feat


def remove_dirs(path):
    """removing files in path"""
    if os.path.exists(path):
        print(f"Removing existing directory: {path}")
        # --- Time directory removal ---
        start_remove = time.time()
        shutil.rmtree(path)
        end_remove = time.time()
        print(f"Directory removal took {end_remove - start_remove:.2f} seconds.")
        # --- End timing ---
        # (Keep original error handling logic just in case, though less likely needed now)
        # try:
        #     shutil.rmtree(path)
        # except OSError as e:
        #     print(f"Error removing directory {path}: {e}")
        #     # Handle cases where it might be a file or encounter permission issues
        #     try:
        #         if os.path.isfile(path) or os.path.islink(path):
        #             os.remove(path)
        #         else:
        #             # Attempt removing contents individually if rmtree failed
        #             for filename in os.listdir(path):
        #                 filepath = os.path.join(path, filename)
        #                 try:
        #                     if os.path.isfile(filepath) or os.path.islink(filepath):
        #                         os.remove(filepath)
        #                     elif os.path.isdir(filepath):
        #                         shutil.rmtree(filepath)
        #                 except Exception as e_inner:
        #                     print(f"Failed to remove {filepath}: {e_inner}")
        #             # Try removing the now potentially empty directory again
        #             os.rmdir(path)
        #     except Exception as e_outer:
        #          print(f"Could not fully clean directory {path}: {e_outer}")


def folds_making(feat_path_csv, wav_path_csv, folds_path, feat_type, spk_indices, labels):
    """making fold-wise tables from table of audio and feature data
    Args:
        feat_path_csv (str): path of feature data table
        wav_path_csv (str): path of audio data table (used if segmentation occurred)
        folds_path (str): path of directory of folds tables for saving
        feat_type (str): feature type
        spk_indices (numpy.ndarray): Array of unique numerical speaker indices (0 to N-1)
        labels (numpy.ndarray): Array of corresponding speaker labels (0 or 1)
    """
    # --- Time the entire fold making process ---
    start_folds = time.time()
    print("Starting fold generation...")

    table_feat = pd.read_csv(os.path.join(feat_path_csv))
    table_wav = None
    if wav_path_csv and os.path.exists(wav_path_csv):
         table_wav = pd.read_csv(os.path.join(wav_path_csv))

    folds_num = 10 # As used in the paper
    main_Kfold_obj = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=42) # Use fixed seed for reproducibility
    val_folds_num = folds_num - 1
    val_Kfold_obj = StratifiedKFold(n_splits=val_folds_num, shuffle=True, random_state=42)

    # Ensure spk_indices and labels are unique per speaker for stratification
    unique_spk_indices = np.unique(spk_indices)
    # Find the first label associated with each unique speaker index
    labels_per_unique_spk = [labels[np.where(spk_indices == idx)[0][0]] for idx in unique_spk_indices]
    labels_per_unique_spk = np.array(labels_per_unique_spk)

    print(f"Total unique speakers for stratification: {len(unique_spk_indices)}")
    if len(unique_spk_indices) != len(labels_per_unique_spk):
         print("Warning: Mismatch between unique speaker indices and labels count!")
         # Continue anyway, maybe it resolves, but good to know
         # return # Or exit if this is critical

    fold_indices = list(main_Kfold_obj.split(unique_spk_indices, labels_per_unique_spk))

    for test_fold in range(folds_num):
        # --- Optional: Time individual fold processing ---
        # start_single_fold = time.time()

        train_val_spk_num_idx, test_spk_num_idx = fold_indices[test_fold]

        # Get the actual numerical speaker indices for train/val and test sets
        train_val_spk_indices = unique_spk_indices[train_val_spk_num_idx]
        test_spk_indices = unique_spk_indices[test_spk_num_idx]
        train_val_spk_labels = labels_per_unique_spk[train_val_spk_num_idx]

        # Split train_val into train and val
        # We need to split the train_val_spk_indices based on their labels
        if len(train_val_spk_indices) < val_folds_num:
             print(f"Warning: Not enough samples in train_val set ({len(train_val_spk_indices)}) for {val_folds_num} validation splits in fold {test_fold+1}. Adjusting n_splits for validation.")
             temp_val_folds_num = max(2, len(train_val_spk_indices)) # Ensure at least 2 splits if possible
             temp_val_Kfold_obj = StratifiedKFold(n_splits=temp_val_folds_num, shuffle=True, random_state=42)
             val_split = list(temp_val_Kfold_obj.split(train_val_spk_indices, train_val_spk_labels))
        else:
            val_split = list(val_Kfold_obj.split(train_val_spk_indices, train_val_spk_labels))

        train_nested_num_idx, val_nested_num_idx = val_split[0] # Use the first split for train/val

        train_spk_indices = train_val_spk_indices[train_nested_num_idx]
        val_spk_indices = train_val_spk_indices[val_nested_num_idx]


        print(f"--- Fold {test_fold+1}/{folds_num} ---")
        print(f"  Train speakers: {len(train_spk_indices)}")
        print(f"  Val speakers: {len(val_spk_indices)}")
        print(f"  Test speakers: {len(test_spk_indices)}")

        # Filter the full data tables based on these speaker indices
        train_table_feat = table_feat[table_feat.ID.isin(train_spk_indices)]
        val_table_feat = table_feat[table_feat.ID.isin(val_spk_indices)]
        test_table_feat = table_feat[table_feat.ID.isin(test_spk_indices)]

        if table_wav is not None:
            train_table_wav = table_wav[table_wav.ID.isin(train_spk_indices)]
            val_table_wav = table_wav[table_wav.ID.isin(val_spk_indices)]
            test_table_wav = table_wav[table_wav.ID.isin(test_spk_indices)]

            train_table_wav.to_csv(
                os.path.join(folds_path, f"train_fold{test_fold+1}_online.csv"),
                index=False,
            )
            val_table_wav.to_csv(
                os.path.join(folds_path, f"val_fold{test_fold+1}_online.csv"),
                index=False,
            )
            test_table_wav.to_csv(
                os.path.join(folds_path, f"test_fold{test_fold+1}_online.csv"),
                index=False,
            )

        train_table_feat.to_csv(
            os.path.join(folds_path, f"train_fold{test_fold+1}_{feat_type}_offline.csv"),
            index=False,
        )
        val_table_feat.to_csv(
            os.path.join(folds_path, f"val_fold{test_fold+1}_{feat_type}_offline.csv"),
            index=False,
        )
        test_table_feat.to_csv(
            os.path.join(folds_path, f"test_fold{test_fold+1}_{feat_type}_offline.csv"),
            index=False,
        )
        # --- Optional: End timing individual fold processing ---
        # end_single_fold = time.time()
        # print(f"  Processing fold {test_fold+1} took {end_single_fold - start_single_fold:.2f} seconds.")

    end_folds = time.time()
    print(f"Fold generation finished. Total time: {end_folds - start_folds:.2f} seconds.")
    # --- End timing for fold making ---


def test_loading(main_dir, feat_type, fold=1, set_type='test'):
    """Basic test for loading features from a generated fold file."""
    # --- Time test loading ---
    start_test = time.time()
    print("Performing final test load...")
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    try:
        fold_file = os.path.join(main_dir, "folds", f"{set_type}_fold{fold}_{feat_type}_offline.csv")
        if not os.path.exists(fold_file):
             print(f"Test loading failed: File not found - {fold_file}")
             return
        df = pd.read_csv(fold_file)
        if df.empty:
             print(f"Test loading warning: File is empty - {fold_file}")
             return
        feat_path_rel = df.iloc[0]['file_path']
        feat_path_abs = os.path.join(root, feat_path_rel) # Construct absolute path
        if not os.path.exists(feat_path_abs):
             print(f"Test loading failed: Feature file not found - {feat_path_abs} (relative: {feat_path_rel})")
             return
        features = np.load(feat_path_abs)
        print(f"Successfully loaded test feature from {feat_path_abs}, shape: {features.shape}")
    except Exception as e:
        print(f"Error during test loading: {e}")
    end_test = time.time()
    print(f"Test loading took {end_test - start_test:.4f} seconds.")
    # --- End timing ---


if __name__ == "__main__":

    # --- Overall script timing START ---
    start_overall = time.time()
    print("Starting script...")

    parser = argparse.ArgumentParser(
        description="Preprocess PC-GITA Vowels dataset."
    )
    parser.add_argument(
        "--Database",
        type=str,
        default="preprocess/PC-GITA_per_task_44100Hz/Vowels",
        help="Path to the PC-GITA Vowels directory",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="preprocess/PC-GITA_per_task_44100Hz/Copia de PCGITA_metadata.csv",
        help="Path to the PC-GITA metadata file",
    )
    parser.add_argument(
        "--segmentdata",
        action="store_true", # Changed to store_true, default is False unless specified
        default=False, # Default is NOT to segment
        help="If specified, segments the audio data (default: False, use original files)",
    )
    parser.add_argument(
        "--init",
        action="store_true", # Changed to store_true
        default=False, # Default is not to remove unless specified
        help="If specified, removes previous saved preprocessed data",
    )
    parser.add_argument(
        "--segmentlen",
        default=8,
        type=float,
        help="Maximum length of segments [seconds] (only used if --segmentdata is specified)",
    )
    parser.add_argument(
        "--njobs",
        default=4,
        type=int,
        help="Number of parallel jobs for offline feature extraction",
    )
    parser.add_argument(
        "--output_dir",
        default="preprocess/PCGITA_vowels",
        type=str,
        help="Directory to save processed data (segmented audio, features, tables, folds)",
    )
    parser.add_argument(
        "--config_path",
        default="config/audio_config.yaml", # Changed path relative to root
        type=str,
        help="Path of feature extraction config file",
    )
    args = parser.parse_args() # Use parse_args() without args=[] for command line

    args.Database = os.path.abspath(args.Database)
    args.metadata_file = os.path.abspath(args.metadata_file)
    args.config_path = os.path.abspath(args.config_path)
    seed_torch(0)
    feat_config_path = args.config_path # Store for use in functions

    # --- Load Metadata START ---
    start_metadata = time.time()
    print(f"Loading metadata from: {args.metadata_file}")
    try:
        metadata = pd.read_csv(args.metadata_file)
        # Create a mapping from speaker ID to label (0 for control, 1 for PD)
        speaker_to_label = {}
        for _, row in metadata.iterrows():
            spk_id = row['RECODING ORIGINAL NAME']
            # Check if UPDRS exists and is not NaN/empty to determine if it's a patient
            is_pd = pd.notna(row['UPDRS']) and row['UPDRS'] != ''
            speaker_to_label[spk_id] = 1 if is_pd else 0
        print(f"Loaded metadata for {len(speaker_to_label)} speakers.")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {args.metadata_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        sys.exit(1)
    # --- Load Metadata END ---
    end_metadata = time.time()
    print(f"Metadata loading took {end_metadata - start_metadata:.2f} seconds.")


    # --- Discover Audio Files and Assign Labels/IDs START ---
    start_discovery = time.time()
    wav_paths = []
    speaker_ids_from_files = []
    speaker_labels = []
    print(f"Searching for vowel audio files in: {args.Database}")
    vowel_dirs = ['A', 'E', 'I', 'O', 'U']
    group_dirs = ['Control', 'Patologicas']

    for group in group_dirs:
        for vowel in vowel_dirs:
            current_dir = os.path.join(args.Database, group, vowel)
            if os.path.isdir(current_dir):
                for filename in sorted(os.listdir(current_dir)):
                    if filename.lower().endswith(".wav"):
                        # Extract speaker ID (e.g., AVPEPUDEAC0001 or AVPEPUDEA0059)
                        match = re.match(r"(AVPEPUDEA[C]?\d+)", filename)
                        if match:
                            spk_id_str = match.group(1)
                            if spk_id_str in speaker_to_label:
                                wav_paths.append(os.path.join(current_dir, filename))
                                speaker_ids_from_files.append(spk_id_str)
                                speaker_labels.append(speaker_to_label[spk_id_str])
                            else:
                                print(f"Warning: Speaker ID {spk_id_str} from file {filename} not found in metadata. Skipping.")
                        else:
                             print(f"Warning: Could not extract valid speaker ID from {filename}. Skipping.")
            else:
                 print(f"Warning: Directory not found - {current_dir}")

    if not wav_paths:
         print("Error: No WAV files found or matched with metadata. Please check paths and metadata file.")
         sys.exit(1)

    print(f"Found {len(wav_paths)} vowel wav files corresponding to metadata.")

    # --- Create Unique Numerical Speaker Indices ---
    unique_speaker_ids = sorted(list(set(speaker_ids_from_files)))
    spk_id_to_numerical_idx = {spk_id: i for i, spk_id in enumerate(unique_speaker_ids)}
    speaker_indices_numeric = [spk_id_to_numerical_idx[spk_id] for spk_id in speaker_ids_from_files]

    total_num = len(wav_paths)
    print(f"Total audio files to process: {total_num}")
    print(f"Total unique speakers: {len(unique_speaker_ids)}")
    # --- Discover Audio Files and Assign Labels/IDs END ---
    end_discovery = time.time()
    print(f"File discovery and ID assignment took {end_discovery - start_discovery:.2f} seconds.")


    # --- Setup Output Paths START ---
    start_setup = time.time()
    main_output_dir = args.output_dir # e.g., preprocess/PCGITA_vowels
    feat_config = get_config_args(args.config_path)
    feat_type = feat_config.get("feat_type")
    save_feat_path = os.path.join(main_output_dir, "features_data", feat_type)

    save_wav_path = None
    if args.segmentdata:
         save_wav_path = os.path.join(main_output_dir, "segmented_audio_data")

    if args.init:
        print("Initialization flag set - removing previous processed data...")
        remove_dirs(main_output_dir) # remove_dirs now has its own timing

    os.makedirs(main_output_dir, exist_ok=True)
    os.makedirs(save_feat_path, exist_ok=True)
    if args.segmentdata:
        os.makedirs(save_wav_path, exist_ok=True)
    # --- Setup Output Paths END ---
    end_setup = time.time()
    print(f"Output path setup took {end_setup - start_setup:.2f} seconds.")


    # --- Run Preprocessing (Parallel) START ---
    start_parallel = time.time()
    print(f"Starting parallel preprocessing with {args.njobs} jobs...")
    if args.segmentdata:
        DF = Parallel(n_jobs=args.njobs)(
            delayed(preprocess_segmenting)(
                wav_paths[i],
                speaker_indices_numeric[i],
                speaker_ids_from_files[i],
                speaker_labels[i],
                save_wav_path,
                save_feat_path,
                args.config_path,
                MaxSegLen=args.segmentlen,
            )
            for i in tqdm(range(total_num), desc="Segmenting & Extracting Features") # Add tqdm desc
        )
        # Filter out None results if any file failed
        valid_results = [res for res in DF if res is not None]
        if not valid_results:
             print("Error: No valid results returned from parallel processing.")
             sys.exit(1)
        df_wav_list, df_feat_list = zip(*valid_results)
        final_wav_df = pd.concat(df_wav_list, ignore_index=True)
        final_feat_df = pd.concat(df_feat_list, ignore_index=True)
        wav_csv_path = os.path.join(main_output_dir, "audio_data.csv")
        # --- Save WAV table START ---
        start_save_wav_csv = time.time()
        final_wav_df.to_csv(wav_csv_path, index=False)
        end_save_wav_csv = time.time()
        print(f"Saved segmented audio data table to: {wav_csv_path} (took {end_save_wav_csv - start_save_wav_csv:.2f}s)")
        # --- Save WAV table END ---
    else:
        # Only feature extraction on original files
        df_feat_list = Parallel(n_jobs=args.njobs)(
            delayed(preprocess)(
                wav_paths[i],
                speaker_indices_numeric[i],
                speaker_ids_from_files[i],
                speaker_labels[i],
                save_feat_path,
                args.config_path,
            )
            for i in tqdm(range(total_num), desc="Extracting Features") # Add tqdm desc
        )
        # Filter out None results
        valid_results = [df for df in df_feat_list if df is not None and not df.empty]
        if not valid_results:
             print("Error: No valid feature dataframes returned from parallel processing.")
             sys.exit(1)
        final_feat_df = pd.concat(valid_results, ignore_index=True)
        wav_csv_path = None # No segmented wav table needed

    # --- Run Preprocessing (Parallel) END ---
    end_parallel = time.time()
    print(f"Parallel preprocessing took {end_parallel - start_parallel:.2f} seconds.")

    # --- Save FEAT table START ---
    start_save_feat_csv = time.time()
    feat_csv_path = os.path.join(main_output_dir, f"{feat_type}_features_data.csv")
    final_feat_df.to_csv(feat_csv_path, index=False)
    end_save_feat_csv = time.time()
    print(f"Saved feature data table to: {feat_csv_path} (took {end_save_feat_csv - start_save_feat_csv:.2f}s)")
    # --- Save FEAT table END ---


    # --- Create Folds START (folds_making has internal timing now) ---
    folds_path = os.path.join(main_output_dir, "folds")
    if not os.path.exists(folds_path):
        os.makedirs(folds_path)

    print("Generating train/validation/test folds...")
    # We need the full list of numerical indices and labels corresponding to the *unique* speakers
    unique_spk_numerical_indices = np.array(list(spk_id_to_numerical_idx.values()))
    unique_spk_labels = np.array([speaker_to_label[sid] for sid in unique_speaker_ids])

    folds_making(
        feat_csv_path,
        wav_csv_path, # Pass None if no segmentation
        folds_path,
        feat_type,
        unique_spk_numerical_indices, # Stratify based on unique speakers
        unique_spk_labels,
    )
    print(f"Saved fold definitions in: {folds_path}")
    # --- Create Folds END ---


    # --- Copy folds for Speaker ID task START ---
    start_copy = time.time()
    spkID_folds_path = os.path.join(main_output_dir, "folds_spkID_task")
    if os.path.exists(spkID_folds_path):
        print(f"Removing existing Speaker ID folds directory: {spkID_folds_path}")
        shutil.rmtree(spkID_folds_path)
    try:
        shutil.copytree(folds_path, spkID_folds_path)
        print(f"Copied folds for Speaker ID task to: {spkID_folds_path}")
    except Exception as e:
        print(f"Error copying folds for Speaker ID task: {e}")
    # --- Copy folds for Speaker ID task END ---
    end_copy = time.time()
    print(f"Copying folds took {end_copy - start_copy:.2f} seconds.")

    # --- Final Test Loading START (test_loading has internal timing now) ---
    test_loading(main_output_dir, feat_type, fold=1, set_type='test')
    # --- Final Test Loading END ---

    # --- Overall script timing END ---
    end_overall = time.time()
    print(f"\nPreprocessing finished. Total script execution time: {end_overall - start_overall:.2f} seconds.")