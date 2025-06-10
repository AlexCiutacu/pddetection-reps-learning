import numpy as np
import os
import pandas as pd
import glob
import re

# --- Experiment Configurations ---
_DEFAULT_PAPER_BASELINE = {
    "acc": 66.20,
    "acc_std": 1.17,
    "auc": 0.77,
    "auc_std": 0.02,
}

EXPERIMENT_CONFIGS = [
    {
        "name": "PCGITA Vowels Baseline",
        "path_key": ["results", "pcgita_vowels_baseline", "ds-pcgita_vowels_baseline", "outputs"],
        "paper_baseline": _DEFAULT_PAPER_BASELINE,
    },
    {
        "name": "PCGITA Vowels Augmented",
        "path_key": ["results", "pcgita_vowels_augmented", "ds-pcgita_vowels_augmented", "outputs"],
        "paper_baseline": _DEFAULT_PAPER_BASELINE,
    },
    {
        "name": "Test DS",
        "path_key": ["results", "test", "ds-test", "outputs"],
        "paper_baseline": None,
    }
]

def load_and_extract_metrics_from_file(file_path, fold_num):
    """Loads data from a single .npy file and extracts metrics."""
    try:
        data_array = np.load(file_path, allow_pickle=True)
        if data_array.ndim == 0:
            extracted_object = data_array.item()
            # print(f"  Extracted Object Type: {type(extracted_object)}")

            if isinstance(extracted_object, dict) and 'test' in extracted_object:
                test_results_list = extracted_object['test']
                # print(f"  'test' key found. Value type: {type(test_results_list)}")

                if isinstance(test_results_list, list) and len(test_results_list) >= 4:
                    try:
                        accuracy_percent = float(test_results_list[3])
                        auc = float(test_results_list[-1])
                        # print(f"  Fold {fold_num} Metrics Extracted: Accuracy={accuracy_percent:.2f}%, AUC={auc:.4f}")
                        return {'accuracy_percent': accuracy_percent, 'auc': auc}, None
                    except (ValueError, TypeError, IndexError) as e:
                        error_msg = f"Error extracting metrics from 'test' list: {e}. Content: {test_results_list[:7]}..."
                        return {'raw_test_list': test_results_list}, error_msg
                else:
                    error_msg = f"'test' value is not a list or is too short. Content: {test_results_list}"
                    return {'raw_test_content': test_results_list}, error_msg
            else:
                error_msg = f"Extracted object is not a dictionary or missing 'test' key. Object: {extracted_object}"
                return {'raw_object': extracted_object}, error_msg
        else:
            error_msg = f"Expected a 0-dimensional array, but got {data_array.ndim} dimensions."
            return {'raw_array': data_array}, error_msg
    except FileNotFoundError:
        return None, f"File not found at '{os.path.abspath(file_path)}'"
    except Exception as e:
        import traceback
        # traceback.print_exc()
        return None, f"An error occurred loading or processing file: {e}"

def process_experiment(experiment_config):
    """Processes all result files for a single experiment configuration."""
    experiment_name = experiment_config["name"]
    # Construct platform-independent path
    base_path = os.path.join(*experiment_config["path_key"])
    paper_baseline = experiment_config.get("paper_baseline")

    print(f"\n\n--- Processing Experiment: {experiment_name} ---")
    print(f"Attempting to load .npy files from base path: {os.path.abspath(base_path)}")
    print("-" * 50)

    fold_metrics = {}
    npy_file_paths = sorted(glob.glob(os.path.join(base_path, "output_*.npy")))

    if not npy_file_paths:
        print(f"  No 'output_*.npy' files found in {base_path}. Skipping.")
        print("-" * 50)
        return

    processed_files_count = 0
    for file_path in npy_file_paths:
        filename = os.path.basename(file_path)
        match = re.search(r"output_(\d+)\.npy", filename)
        if not match:
            print(f"  Skipping file with unexpected name format: {filename}")
            continue
        
        fold_num = int(match.group(1))
        # print(f"--- Processing File: {filename} (Fold {fold_num}) ---")
        
        metrics, error = load_and_extract_metrics_from_file(file_path, fold_num)
        if metrics:
            fold_metrics[fold_num] = metrics
        if error:
            print(f"  Fold {fold_num} ({filename}): {error}")
        # print("-" * 30)
        processed_files_count +=1

    if processed_files_count == 0:
        print(f"  No files matching 'output_<number>.npy' pattern were successfully processed in {base_path}.")
        print("-" * 50)
        return

    # print("=" * 50)
    # print(f"Finished processing .npy files for experiment: {experiment_name}.")

    if not fold_metrics:
        print(f"\nNo metrics could be extracted for {experiment_name}.")
        return

    metrics_df = pd.DataFrame.from_dict(fold_metrics, orient='index')
    metrics_df.index.name = "Fold"
    metrics_df = metrics_df.sort_index()

    print(f"\n--- {experiment_name}: Extracted Metrics Per Fold ---")
    print(metrics_df)

    print(f"\n--- {experiment_name}: Aggregated Metrics (Mean +/- Std Dev) ---")
    if 'accuracy_percent' in metrics_df.columns and 'auc' in metrics_df.columns:
        metrics_df['accuracy_percent'] = pd.to_numeric(metrics_df['accuracy_percent'], errors='coerce')
        metrics_df['auc'] = pd.to_numeric(metrics_df['auc'], errors='coerce')
        valid_metrics_df = metrics_df.dropna(subset=['accuracy_percent', 'auc'])

        if not valid_metrics_df.empty:
            mean_accuracy_percent = valid_metrics_df['accuracy_percent'].mean()
            std_accuracy_percent = valid_metrics_df['accuracy_percent'].std()
            mean_auc = valid_metrics_df['auc'].mean()
            std_auc = valid_metrics_df['auc'].std()

            print(f"PD Classification Accuracy: {mean_accuracy_percent:.2f} ± {std_accuracy_percent:.2f} %")
            print(f"PD Classification AUC:      {mean_auc:.2f} ± {std_auc:.2f}")

            if paper_baseline:
                print(f"\n--- {experiment_name}: Comparison with Paper Baseline ---")
                print(f"Your Results:   Accuracy={mean_accuracy_percent:.2f}±{std_accuracy_percent:.2f} %, AUC={mean_auc:.2f}±{std_auc:.2f}")
                print(f"Paper Baseline: Accuracy={paper_baseline['acc']:.2f}±{paper_baseline['acc_std']:.2f} %, AUC={paper_baseline['auc']:.2f}±{paper_baseline['auc_std']:.2f}")
            else:
                print(f"\n--- {experiment_name}: Paper baseline comparison not configured or N/A. ---")
        else:
            print("Could not calculate mean/std dev. No valid numeric metrics found after processing.")
            # print("DataFrame with attempted numeric conversion (may contain NaNs):")
            # print(metrics_df)
    else:
        print("Could not calculate mean/std dev. 'accuracy_percent' or 'auc' columns missing/empty.")
        # print("DataFrame content:")
        # print(metrics_df)
    print("=" * 70)

def main():
    """Main function to run the results processing for all configured experiments."""
    print(f"Script's CWD: {os.getcwd()}")
    print("=" * 70)

    for config in EXPERIMENT_CONFIGS:
        process_experiment(config)
    
    print("\n\nAll configured experiments processed.")

if __name__ == "__main__":
    main()
