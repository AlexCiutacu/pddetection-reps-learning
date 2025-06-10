    #!/bin/bash
    set -e

    # --- Configuration ---
    EXP_NAME="pcgita_vowels_augmented"
    FOLDS=$(seq 1 10)
    UPSTREAM_ARGS="--newinit --valmonitor --auxiltr --auxlossw1 -0.01 --auxlossw2 0.01"
    DOWNSTREAM_ARGS="--valmonitor --newinit --mode train"

    # --- Upstream Training Loop ---
    echo "Starting Upstream Training for experiment: $EXP_NAME"
    for FOLD in $FOLDS; do
      echo "--- Running Upstream Fold $FOLD ---"
      python train_upstream.py --expname "$EXP_NAME" --fold "$FOLD" $UPSTREAM_ARGS
      echo "--- Finished Upstream Fold $FOLD ---"
    done
    echo "Finished All Upstream Training."

    # --- Downstream Training Loop ---
    echo "Starting Downstream Training for experiment: $EXP_NAME"
    for FOLD in $FOLDS; do
      echo "--- Running Downstream Fold $FOLD ---"
      python train_downstream.py --expname "$EXP_NAME" --fold "$FOLD" $DOWNSTREAM_ARGS
      echo "--- Finished Downstream Fold $FOLD ---"
    done
    echo "Finished All Downstream Training."

    echo "All folds completed for experiment: $EXP_NAME"