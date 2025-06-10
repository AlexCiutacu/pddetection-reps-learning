"""*********************************************************************************************"""
#   Preprocess PC-GITA Vowels combining baseline folds with augmented data.
#   1) Loads baseline fold definitions (train/val/test).
#   2) Loads metadata for pre-generated augmented audio files.
#   3) Combines original training data with corresponding augmented data for each fold.
#   4) Segments combined audio.
#   5) Computes/saves features for the combined dataset.
#   6) Saves new combined fold definitions (online and offline tables).

# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Parvaneh Janbakhshi <parvaneh.janbakhshi@idiap.ch>

# This file is part of pddetection-reps-learning
# (License details as in the original script)
"""*********************************************************************************************"""

from pathlib import Path
import sys
import os
import re
import time
import argparse
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm
from os.path import basename, join, exists, relpath, abspath, dirname

# ------------------------------ Path change ----------------------------- #
try:
    file = Path(__file__).resolve()
    root = file.parents[2]
    sys.path.append(str(root))
    os.chdir(root) # Set root as CWD
    print(f"[DEBUG] Project Root identified as: {root}")
    print(f"[DEBUG] Current Working Directory set to: {os.getcwd()}")
    print(f"[DEBUG] Added to sys.path: {str(root)}")
except IndexError:
    print("[ERROR] Could not determine project root directory. Script assumes it's two levels above 'preprocess/pc_gita_vowels_augmented/'.")
    sys.exit(1)
except Exception as e:
    print(f"Error setting up paths: {e}")
    sys.exit(1)
# ------------------------------------------------------------------------- #

# --- Import necessary functions from the original script ---
try:
    from downstream.ds_runner import seed_torch
    from audio.audio_utils import get_waveform, get_feat, get_config_args


    def segmenting(wav_path, saved_path_dir, MaxSegLen=8, feat_config_path_global=None):
        """Segmenting long utterances"""
        try:
            wav, fs = get_waveform(wav_path, normalization=True)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return # Skip this file if loading fails

        n_samples = len(wav)
        segmentsize = MaxSegLen * fs
        if not os.path.exists(saved_path_dir):
            os.makedirs(saved_path_dir, exist_ok=True)

        base_name = basename(wav_path).split(Path(wav_path).suffix)[0]
        relative_input_path = Path(relpath(wav_path, root))
        try:
             speaker_id_seg = relative_input_path.parent.name
             fold_dir_seg = relative_input_path.parent.parent.name
             output_base_for_file = join(saved_path_dir, fold_dir_seg, speaker_id_seg)
        except IndexError:
             print(f"[WARN] Could not parse fold/speaker from path {relative_input_path}. Saving segments directly under {saved_path_dir}")
             output_base_for_file = saved_path_dir

        output_dir = join(output_base_for_file, base_name + "_segments") # Save segments in specific subdirs
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        segmentsize_int = int(segmentsize)
        if segmentsize_int <= 0:
            print(f"Warning: Calculated segment size is non-positive ({segmentsize_int}) for {wav_path}. Skipping segmentation.")
            return

        if feat_config_path_global:
             min_len_ms = float(get_config_args(feat_config_path_global).get("torchaudio", {}).get("frame_shift", 10.0))
        else:
             min_len_ms = 10.0 # Default fallback
             print("[WARN] feat_config_path_global not available to segmenting, using default min_len_ms=10.0")


        for ind, start in enumerate(range(0, n_samples, segmentsize_int)):
            end = min(start + segmentsize_int, n_samples)
            segment_len_ms = (end - start) * 1e3 / fs
            if segment_len_ms < min_len_ms:
                 continue
            new_filename = base_name + "_uttr" + str(ind) + Path(wav_path).suffix
            new_path = join(output_dir, new_filename)
            try:
                sf.write(new_path, wav[start:end], fs)
                yield new_path, segment_len_ms # yield absolute path
            except Exception as e:
                print(f"Error writing segment {new_path}: {e}")

    def feature_extraction(wav_path, saved_path_dir, feat_config_path):
        """extracting features from audio file (Copied)"""
        try:
            features = get_feat(wav_path, feat_config_path).T # Transpose to D x T
            if features is None or features.size == 0:
                print(f"Warning: Feature extraction yielded empty features for {wav_path}")
                return None, 0
        except Exception as e:
            print(f"Error extracting features for {wav_path}: {e}")
            return None, 0

        base_name = basename(wav_path).split(Path(wav_path).suffix)[0]
        relative_input_path = Path(relpath(wav_path, root))
        try:
            path_parts = relative_input_path.parts
            if "segmented_audio_data" in path_parts:
                 # segmented_audio_data/fold_X/speaker_Y/base_segments/base_uttrZ.wav
                 fold_dir_feat = path_parts[-4] # fold_X
                 speaker_dir_feat = path_parts[-3] # speaker_Y
            elif "audio" in path_parts and "augmented" in path_parts[-5]: # Heuristic for augmented path
                 fold_dir_feat = path_parts[-3] # fold_X
                 speaker_dir_feat = path_parts[-2] # speaker_Y
            elif "Vowels" in path_parts: # Original files (heuristic)
                 speaker_dir_match = re.match(r"(AVPEPUDEA[C]?\d+)", base_name)
                 if speaker_dir_match:
                      speaker_dir_feat = speaker_dir_match.group(1)
                      fold_dir_feat = "unknown_fold"
                 else:
                      speaker_dir_feat = "unknown_speaker"
                      fold_dir_feat = "unknown_fold"
            else: # Fallback
                speaker_dir_feat = "unknown_speaker"
                fold_dir_feat = "unknown_fold"

            output_subdir = join(saved_path_dir, fold_dir_feat, speaker_dir_feat)

        except Exception as path_e:
            print(f"[WARN] Error parsing path {relative_input_path} for feature saving: {path_e}. Using flat structure.")
            output_subdir = saved_path_dir # Fallback

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir, exist_ok=True)

        new_filename_base = base_name
        new_path = join(output_subdir, new_filename_base + ".npy") # Save as .npy

        try:
            np.save(new_path.replace('.npy',''), features)
            return new_path, features.shape[1] # Return absolute path
        except Exception as e:
            print(f"Error saving features for {new_path}: {e}")
            return None, 0

    def process_single_file(args_tuple):
        """Wrapper for parallel processing: segments & extracts features."""
        (abs_wav_path, ID, SPK_ID, label, segment, seg_len,
         save_seg_path, save_feat_path, feat_config_path_local) = args_tuple

        results_list = []

        try:
            # Check if input file exists before processing
            if not exists(abs_wav_path):
                 print(f"[WARN] Input file not found in process_single_file: {abs_wav_path}. Skipping.")
                 return results_list # Return empty list for this file

            if segment:
                segmenting_gen = segmenting(abs_wav_path, save_seg_path, MaxSegLen=seg_len, feat_config_path_global=feat_config_path_local)
                if segmenting_gen is None:
                    return results_list # Error during segmentation

                for abs_seg_path, wav_length_ms in segmenting_gen:
                    # Extract features for the segment
                    abs_feat_path, feat_length = feature_extraction(
                        abs_seg_path, save_feat_path, feat_config_path_local
                    )
                    if abs_feat_path and feat_length > 1:
                        rel_seg_path = relpath(abs_seg_path, root)
                        rel_feat_path = relpath(abs_feat_path, root)
                        results_list.append({
                            "ID": ID,
                            "spk_ID": SPK_ID,
                            "label": label,
                            "wav_path": rel_seg_path, # Relative path to segmented wav
                            "wav_length_ms": wav_length_ms,
                            "feat_path": rel_feat_path, # Relative path to feature file
                            "feat_length": feat_length, # Feature frames
                            "original_wav_path": relpath(abs_wav_path, root) # Keep track of original source
                        })
                    else:
                        print(f"[WARN] Skipping segment {abs_seg_path} due to feature extraction issue.")
            else:
                # No segmentation, process original file directly
                wav, fs = get_waveform(abs_wav_path, normalization=True)
                wav_length_ms = len(wav) * 1e3 / fs
                min_len_ms = float(get_config_args(feat_config_path_local).get("torchaudio",{}).get("frame_length", 25.0))

                if wav_length_ms >= min_len_ms:
                    abs_feat_path, feat_length = feature_extraction(
                        abs_wav_path, save_feat_path, feat_config_path_local
                    )
                    if abs_feat_path and feat_length > 1:
                        rel_wav_path = relpath(abs_wav_path, root)
                        rel_feat_path = relpath(abs_feat_path, root)
                        results_list.append({
                            "ID": ID,
                            "spk_ID": SPK_ID,
                            "label": label,
                            "wav_path": rel_wav_path, # Relative path to original wav
                            "wav_length_ms": wav_length_ms,
                            "feat_path": rel_feat_path, # Relative path to feature file
                            "feat_length": feat_length, # Feature frames
                            "original_wav_path": rel_wav_path # Original is itself
                        })
                    else:
                         print(f"[WARN] Skipping file {abs_wav_path} due to feature extraction issue.")
                else:
                    print(f"[WARN] Skipping file {abs_wav_path} due to insufficient length ({wav_length_ms:.2f}ms).")

        except Exception as e:
            print(f"[ERROR] Failed processing {abs_wav_path}: {e}")
            import traceback
            traceback.print_exc()

        return results_list # List of dicts for segments/file


    def remove_dirs(path):
        """removing files in path"""
        if os.path.exists(path):
            print(f"Removing existing directory: {path}")
            start_remove = time.time()
            try:
                shutil.rmtree(path)
            except OSError as e:
                 print(f"[ERROR] Failed to remove directory {path}: {e}")
            end_remove = time.time()
            print(f"Directory removal took {end_remove - start_remove:.2f} seconds.")

    def test_loading(main_dir, feat_type, fold=1, set_type='test'):
        """Basic test for loading features from a generated fold file."""
        start_test = time.time()
        print(f"Performing final test load for fold {fold}, set {set_type}...")
        try:
            fold_file = join(main_dir, "folds", f"{set_type}_fold{fold}_{feat_type}_offline.csv")
            if not exists(fold_file):
                print(f"Test loading failed: File not found - {fold_file}")
                return
            df = pd.read_csv(fold_file)
            if df.empty:
                print(f"Test loading warning: File is empty - {fold_file}")
                return
            feat_path_col = 'feat_path' if 'feat_path' in df.columns else 'file_path'
            feat_path_rel = df.iloc[0][feat_path_col]
            feat_path_abs = join(root, feat_path_rel) # Construct absolute path from root

            if not exists(feat_path_abs):
                print(f"Test loading failed: Feature file not found - {feat_path_abs} (relative: {feat_path_rel})")
                feat_path_abs_alt = abspath(feat_path_rel)
                if exists(feat_path_abs_alt):
                     print(f"Test loading found feature file at alternative absolute path: {feat_path_abs_alt}")
                     feat_path_abs = feat_path_abs_alt
                else:
                     print(f"Test loading still failed to find feature file.")
                     return

            features = np.load(feat_path_abs)
            print(f"Successfully loaded test feature from {feat_path_abs}, shape: {features.shape}")
        except Exception as e:
            print(f"Error during test loading: {e}")
            import traceback
            traceback.print_exc()
        end_test = time.time()
        print(f"Test loading took {end_test - start_test:.4f} seconds.")

except ImportError as e:
    print(f"Error importing required functions: {e}")
    print("Ensure 'audio.audio_utils' and 'downstream.ds_runner' are available.")
    sys.exit(1)


if __name__ == "__main__":

    start_overall = time.time()
    print("Starting Augmented Preprocessing Script...")

    parser = argparse.ArgumentParser(
        description="Preprocess PC-GITA Vowels dataset, combining baseline folds with augmented data."
    )
    # --- Inputs ---
    parser.add_argument(
        "--baseline_folds_dir",
        type=str,
        required=True,
        help="Path to the directory containing the original baseline fold CSVs (e.g., .../processed_output/folds)",
    )
    parser.add_argument(
        "--augmented_metadata_csv",
        type=str,
        required=True,
        help="Path to the metadata CSV file for the generated augmented audio (e.g., .../augmented/folds/augmented_files_metadata.csv)",
    )
    parser.add_argument(
        "--config_path",
        default="config/audio_config.yaml",
        type=str,
        help="Path to the feature extraction config file (relative to project root)",
    )
    # --- Outputs ---
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="New directory to save processed data (features, combined tables, combined folds)",
    )
    # --- Options ---
    parser.add_argument(
        "--segmentdata",
        action="store_true",
        default=False,
        help="If specified, segments the audio data (both original and augmented)",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        default=False,
        help="If specified, removes the target output directory before starting",
    )
    parser.add_argument(
        "--segmentlen",
        default=8,
        type=float,
        help="Maximum length of segments in seconds (used if --segmentdata is specified)",
    )
    parser.add_argument(
        "--njobs",
        default=max(1, os.cpu_count() // 2), # Default to half cores
        type=int,
        help="Number of parallel jobs for feature extraction",
    )

    args = parser.parse_args()

    # Resolve paths relative to the project root (CWD)
    args.baseline_folds_dir = abspath(args.baseline_folds_dir)
    args.augmented_metadata_csv = abspath(args.augmented_metadata_csv)
    args.config_path = abspath(args.config_path)
    args.output_dir = abspath(args.output_dir)
    seed_torch(0)
    feat_config_path = args.config_path
    # --- Load Augmented Metadata ---
    start_metadata = time.time()
    print(f"Loading augmented metadata from: {args.augmented_metadata_csv}")
    try:
        aug_metadata_df = pd.read_csv(args.augmented_metadata_csv)
        # Ensure necessary columns exist
        required_aug_cols = ['ID', 'file_path', 'label', 'spk_ID', 'augmentation']
        missing_cols = [col for col in required_aug_cols if col not in aug_metadata_df.columns]
        if missing_cols:
             print(f"[ERROR] Augmented metadata CSV is missing columns: {missing_cols}")
             sys.exit(1)
        print(f"Loaded {len(aug_metadata_df)} augmented file entries.")
        # Convert relative paths in augmented metadata to absolute paths for processing
        aug_metadata_df['abs_path'] = aug_metadata_df['file_path'].apply(lambda x: join(root, x))
        # Check if these absolute paths exist
        non_existent_aug = aug_metadata_df[~aug_metadata_df['abs_path'].apply(exists)]
        if not non_existent_aug.empty:
             print(f"[WARN] {len(non_existent_aug)} augmented files listed in CSV do not exist on disk:")
             for _, row in non_existent_aug.head().iterrows(): # Print a few examples
                  print(f"  - {row['abs_path']} (from {row['file_path']})")

    except FileNotFoundError:
        print(f"Error: Augmented metadata file not found at {args.augmented_metadata_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading augmented metadata file: {e}")
        sys.exit(1)
    end_metadata = time.time()
    print(f"Augmented metadata loading took {end_metadata - start_metadata:.2f} seconds.")


    # --- Setup Output Paths ---
    start_setup = time.time()
    main_output_dir = args.output_dir
    feat_config = get_config_args(args.config_path)
    feat_type = feat_config.get("feat_type")
    save_feat_path = join(main_output_dir, "features_data", feat_type)
    save_seg_path = join(main_output_dir, "segmented_audio_data") if args.segmentdata else None
    save_folds_path = join(main_output_dir, "folds")

    if args.init:
        print("Initialization flag set - removing previous processed data...")
        remove_dirs(main_output_dir)

    os.makedirs(main_output_dir, exist_ok=True)
    os.makedirs(save_feat_path, exist_ok=True)
    os.makedirs(save_folds_path, exist_ok=True)
    if save_seg_path:
        os.makedirs(save_seg_path, exist_ok=True)
    end_setup = time.time()
    print(f"Output path setup took {end_setup - start_setup:.2f} seconds.")
    print(f"  Features will be saved under: {save_feat_path}")
    if save_seg_path:
        print(f"  Segmented audio will be saved under: {save_seg_path}")
    print(f"  Fold definitions will be saved under: {save_folds_path}")


    # --- Combine Folds and Collect Files for Processing ---
    start_combine = time.time()
    print("Combining baseline folds with augmented data...")
    files_to_process_map = {} # Store as {abs_path: (ID, SPK_ID, label)}
    combined_fold_dataframes_online = {} # Store as {(fold, set): dataframe}

    num_folds = 10
    for fold_num in range(1, num_folds + 1):
        print(f" Processing Fold {fold_num}/{num_folds}")
        fold_files_added = {'train': 0, 'val': 0, 'test': 0}
        aug_files_added_to_train = 0

        for split in ["train", "val", "test"]:
            baseline_csv_path = join(args.baseline_folds_dir, f"{split}_fold{fold_num}_online.csv")
            if not exists(baseline_csv_path):
                print(f" [WARN] Baseline fold file not found: {baseline_csv_path}. Skipping {split} for fold {fold_num}.")
                combined_fold_dataframes_online[(fold_num, split)] = pd.DataFrame() # Store empty DF
                continue

            try:
                df_orig = pd.read_csv(baseline_csv_path)
                # Ensure baseline paths are absolute for processing map
                df_orig['abs_path'] = df_orig['file_path'].apply(lambda x: join(root, x))

                # Store original files for processing
                for _, row in df_orig.iterrows():
                    files_to_process_map[row['abs_path']] = (row['ID'], row['spk_ID'], row['label'])
                    fold_files_added[split] += 1

                # If it's the training split, add augmented files
                if split == "train":
                    training_spk_ids = set(df_orig['spk_ID'].unique())
                    aug_mask = aug_metadata_df['spk_ID'].isin(training_spk_ids) & \
                               aug_metadata_df['abs_path'].apply(lambda x: f"fold_{fold_num}" in Path(x).parts)

                    df_aug_filtered = aug_metadata_df[aug_mask].copy()

                    if not df_aug_filtered.empty:
                        # Add augmented files for processing
                        for _, row in df_aug_filtered.iterrows():
                           if exists(row['abs_path']):
                                files_to_process_map[row['abs_path']] = (row['ID'], row['spk_ID'], row['label'])
                                aug_files_added_to_train += 1
                           else:
                                print(f"  [WARN] Filtered augmented file missing: {row['abs_path']}")


                        df_aug_for_concat = df_aug_filtered[['ID', 'file_path', 'label', 'spk_ID']].copy()

                        df_combined = pd.concat([df_orig[['ID', 'file_path', 'label', 'spk_ID']], df_aug_for_concat], ignore_index=True)
                        combined_fold_dataframes_online[(fold_num, split)] = df_combined
                        print(f"  Train: {len(df_orig)} original + {aug_files_added_to_train} augmented = {len(df_combined)} total")
                    else:
                        print(f"  Train: {len(df_orig)} original + 0 augmented")
                        combined_fold_dataframes_online[(fold_num, split)] = df_orig[['ID', 'file_path', 'label', 'spk_ID']] # Keep original if no aug found
                else:
                    combined_fold_dataframes_online[(fold_num, split)] = df_orig[['ID', 'file_path', 'label', 'spk_ID']]
                    print(f"  {split.capitalize()}: {len(df_orig)} original")

            except Exception as e:
                print(f" [ERROR] Failed processing {baseline_csv_path}: {e}")
                combined_fold_dataframes_online[(fold_num, split)] = pd.DataFrame() # Store empty DF on error

    end_combine = time.time()
    print(f"Fold combination took {end_combine - start_combine:.2f} seconds.")
    print(f"Total unique files collected for processing: {len(files_to_process_map)}")


    # --- Prepare Parallel Processing Tasks ---
    tasks = []
    print("Preparing parallel processing tasks...")
    for abs_path, (id_val, spk_id_val, label_val) in files_to_process_map.items():
        tasks.append((
            abs_path, id_val, spk_id_val, label_val, args.segmentdata, args.segmentlen,
            save_seg_path, save_feat_path, feat_config_path
        ))

    if not tasks:
         print("[ERROR] No tasks created for parallel processing. Exiting.")
         sys.exit(1)

    # --- Run Parallel Feature Extraction ---
    start_parallel = time.time()
    print(f"Starting parallel feature extraction with {args.njobs} jobs...")
    results_nested = Parallel(n_jobs=args.njobs)(
        delayed(process_single_file)(task) for task in tqdm(tasks, desc="Extracting Features")
    )

    # Flatten results and create a map for easy lookup
    # Map: original_wav_path -> list of processed segment/file info dicts
    processed_files_map = {}
    total_processed_segments = 0
    for result_list in results_nested:
        if result_list: # Only process if the list is not empty
            total_processed_segments += len(result_list)
            original_path = result_list[0]['original_wav_path'] # All items in list share the same original
            if original_path not in processed_files_map:
                 processed_files_map[original_path] = []
            processed_files_map[original_path].extend(result_list)

    end_parallel = time.time()
    print(f"Parallel processing finished. Took {end_parallel - start_parallel:.2f} seconds.")
    print(f"Successfully processed {len(processed_files_map)} original files into {total_processed_segments} segments/files with features.")

    if not processed_files_map:
         print("[ERROR] No files were successfully processed. Check logs for errors. Exiting.")
         sys.exit(1)

    # --- Generate Final Fold CSVs (Online and Offline) ---
    start_saving_folds = time.time()
    print("Generating final fold CSV files...")

    for fold_num in range(1, num_folds + 1):
        for split in ["train", "val", "test"]:
            online_df = combined_fold_dataframes_online.get((fold_num, split))
            if online_df is None or online_df.empty:
                print(f"  Skipping Fold {fold_num} / {split} due to missing initial data.")
                continue

            offline_data_list = []
            processed_wav_paths = [] # For the 'online' df final save

            # Iterate through the combined/original online dataframe
            for _, row in online_df.iterrows():
                original_rel_path = row['file_path']
                # Look up processing results using the relative path as key
                processed_results = processed_files_map.get(original_rel_path)

                if processed_results:
                    # Add all processed segments/files originating from this row
                    for processed_info in processed_results:
                         # For offline table
                         offline_data_list.append({
                             "ID": processed_info["ID"],
                             "file_path": processed_info["feat_path"], # Feature path for offline
                             "length": processed_info["feat_length"], # Feature length
                             "label": processed_info["label"],
                             "spk_ID": processed_info["spk_ID"],
                         })
                         # For final online table
                         processed_wav_paths.append({
                              "ID": processed_info["ID"],
                              "file_path": processed_info["wav_path"], # Segment/original wav path
                              "length": processed_info["wav_length_ms"], # Wav/Segment length
                              "label": processed_info["label"],
                              "spk_ID": processed_info["spk_ID"],
                         })

                else:
                    pass # Don't add it to the final CSVs

            if not offline_data_list:
                 print(f"  [WARN] No successfully processed files for Fold {fold_num}/{split}. Skipping CSV generation.")
                 continue

            # Create final DataFrames
            offline_df_final = pd.DataFrame(offline_data_list)
            online_df_final = pd.DataFrame(processed_wav_paths)

            # Save CSVs
            online_csv_path = join(save_folds_path, f"{split}_fold{fold_num}_online.csv")
            offline_csv_path = join(save_folds_path, f"{split}_fold{fold_num}_{feat_type}_offline.csv")

            try:
                online_df_final.to_csv(online_csv_path, index=False)
                offline_df_final.to_csv(offline_csv_path, index=False)
                # print(f"  Saved: {basename(online_csv_path)} ({len(online_df_final)} rows)")
                # print(f"  Saved: {basename(offline_csv_path)} ({len(offline_df_final)} rows)")
            except Exception as e:
                print(f"  [ERROR] Failed to save CSVs for Fold {fold_num}/{split}: {e}")

    end_saving_folds = time.time()
    print(f"Fold CSV generation took {end_saving_folds - start_saving_folds:.2f} seconds.")


    # --- Copy folds for Speaker ID task ---
    start_copy = time.time()
    spkID_folds_path = join(main_output_dir, "folds_spkID_task")
    remove_dirs(spkID_folds_path) # Remove if exists
    try:
        shutil.copytree(save_folds_path, spkID_folds_path)
        print(f"Copied folds for Speaker ID task to: {spkID_folds_path}")
    except Exception as e:
        print(f"Error copying folds for Speaker ID task: {e}")
    end_copy = time.time()
    print(f"Copying folds took {end_copy - start_copy:.2f} seconds.")


    # --- Final Test Loading ---
    test_loading(main_output_dir, feat_type, fold=1, set_type='test')


    # --- Overall script timing END ---
    end_overall = time.time()
    print(f"\nAugmented Preprocessing finished. Total script execution time: {end_overall - start_overall:.2f} seconds.")
    print(f"Output data saved in: {args.output_dir}")