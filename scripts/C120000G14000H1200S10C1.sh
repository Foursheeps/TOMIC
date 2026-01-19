#!/bin/bash
# Script to run all training scripts sequentially
# Usage: bash run_all.sh

set -e  # Exit on error
# export CUDA_VISIBLE_DEVICES=1

# ==========================================
# Configuration: Set Python and paths
# ==========================================
PYTHON=/home/LYan/miniconda3/envs/clip/bin/python

# ==========================================
# Configuration: Set data paths
# ==========================================
DATA_PATH=/your/path/to/processed_data

# ==========================================
# Configuration: Set output paths
# ==========================================
DEFAULT_ROOT_DIR_BASE=${DATA_PATH}/outputs

# ==========================================
# Configuration: Set script save path
# ==========================================
SCRIPT_SAVE_PATH=/your/path/to/TOMIC/scripts

# ==========================================
# Configuration: Set script paths
# ==========================================
DATA_GEN_SCRIPT=/your/path/to/TOMIC/process_data/create_syncdata.py
ADDA_SCRIPT=/your/path/to/TOMIC/train_val_scripts/main_adda.py
DANN_SCRIPT=/your/path/to/TOMIC/train_val_scripts/main_dann.py
DSN_SCRIPT=/your/path/to/TOMIC/train_val_scripts/main_dsn.py
USUAL_SCRIPT=/your/path/to/TOMIC/train_val_scripts/main_usual.py

# ==========================================
# Configuration: Data generation parameters
# ==========================================
CELLS_PER_ORGAN="dict(Liver=40000,Lung=30000,Stomach=20000,Peritoneum=30000)"
# Calculate total number of cells from cells_per_organ dict
# Extract numbers from the dict string and sum them
N_CELLS=$(echo "$CELLS_PER_ORGAN" | grep -oE '[0-9]+' | awk '{sum+=$1} END {print sum}')

N_GENES=14000
N_HIGHLY_VARIABLE_GENES=1200
CLASS_SEP=10
RANDOM_STATE=42
OVERWRITE=0

DATA_FILE_NAME=C${N_CELLS}G${N_GENES}H${N_HIGHLY_VARIABLE_GENES}S${CLASS_SEP}

# Update the data path and copy this file to new data path
NEW_DATA_PATH=${DATA_PATH}/${DATA_FILE_NAME}

# CP this file to new data path 
# cp $0 $SCRIPT_SAVE_PATH/${DATA_FILE_NAME}.sh

# ==========================================
# Configuration: Training parameters
# ==========================================
TRAIN_MODELS="['mlp', 'patch', 'expr', 'name']"
DEVICES=1
RUN_TRAINING=1
RUN_TESTING=1

# Control which training scripts to run (1 = run, 0 = skip)
RUN_ADDA=0
RUN_DANN=0
RUN_DSN=1
RUN_USUAL=0


# Learning rate and scheduler
LR=1e-4
MAX_EPOCHS=80
PRETRAIN_EPOCHS=40
SCHEDULER_TYPE="warmupcosine"
WARMUP_RATIO=0.05

# Batch sizes and data loading
TRAIN_BATCH_SIZE=64
TEST_BATCH_SIZE=64
NUM_WORKERS=4

# PyTorch Lightning Trainer configuration
SEED=2025
PRECISION="16-mixed"
LOG_EVERY_N_STEPS=50
VAL_CHECK_INTERVAL=
CHECK_VAL_EVERY_N_EPOCH=1

# Callback configuration
PATIENCE=15
MONITOR_METRIC="val/total_loss"
MODE="min"
SAVE_TOP_K=2

# ADDA-specific callback configuration
PRETRAIN_PATIENCE=5
PRETRAIN_MONITOR_METRIC="val/pretrain_accuracy"
PRETRAIN_MODE="max"
ADVERSARIAL_PATIENCE=
ADVERSARIAL_MONITOR_METRIC=
ADVERSARIAL_MODE=

# Other parameters
BINGINGS="[None, 50]"
USUAL_TRAIN_DOMAINS="['source', 'target', 'both']"
CHECKPOINT_PATH=

# ==========================================
# Configuration: Model parameters
# ==========================================

# Base model arguments
DROPOUT=0.1
ACTIVATION="gelu"

# MLP model arguments
HIDDEN_DIMS="[512, 256, 128]"

# Transformer model arguments
HIDDEN_SIZE=128
PATCH_SIZE=40
NUM_HEADS=8
NUM_LAYERS=6

# Dual Transformer model arguments
NUM_HEADS_CROSS_ATTN=8
NUM_LAYERS_CROSS_ATTN=2
NUM_HEADS_ENCODER=8
NUM_LAYERS_ENCODER=2

# ==========================================
# Configuration: Domain adaptation parameters
# ==========================================

# DANN-specific parameters
DANN_GAMMA=0.1

# DSN-specific parameters
DSN_ALPHA=4.0
DSN_BETA=0.25
DSN_GAMMA=0.1

# Export variables so they can be used by child scripts
export DATA_PATH
export DEFAULT_ROOT_DIR_BASE

echo "=========================================="
echo "Configuration:"
echo "  DATA_PATH: $DATA_PATH"
echo "  DEFAULT_ROOT_DIR_BASE: $DEFAULT_ROOT_DIR_BASE"
echo "  NEW_DATA_PATH: $NEW_DATA_PATH"
echo "  N_CELLS: $N_CELLS"
echo "  N_GENES: $N_GENES"
echo "  N_HIGHLY_VARIABLE_GENES: $N_HIGHLY_VARIABLE_GENES"
echo "  CLASS_SEP: $CLASS_SEP"
echo ""
echo "Training scripts to run:"
echo "  RUN_ADDA: $RUN_ADDA"
echo "  RUN_DANN: $RUN_DANN"
echo "  RUN_DSN: $RUN_DSN"
echo "  RUN_USUAL: $RUN_USUAL"
echo "=========================================="
echo ""

# ==========================================
# Step 1: Generate synthetic data
# ==========================================
echo "Step 1: Generating synthetic data..."

$PYTHON $DATA_GEN_SCRIPT \
    --output_base_dir $DATA_PATH \
    --cells_per_organ "$CELLS_PER_ORGAN" \
    --n_genes $N_GENES \
    --n_highly_variable_genes $N_HIGHLY_VARIABLE_GENES \
    --class_sep $CLASS_SEP \
    --random_state $RANDOM_STATE \
    --overwrite $OVERWRITE

echo "  NEW_DATA_PATH: $NEW_DATA_PATH"
echo "=========================================="
echo ""


echo ""
echo "=========================================="
echo "Step 2: Running training scripts..."
echo "=========================================="

# ==========================================
# Step 2: Run ADDA training
# ==========================================
if [ "$RUN_ADDA" -eq 1 ]; then
    echo ""
    echo "2. Running ADDA training..."

    $PYTHON $ADDA_SCRIPT \
    --train_models "$TRAIN_MODELS" \
    --data_path "$NEW_DATA_PATH" \
    --default_root_dir "${DEFAULT_ROOT_DIR_BASE}/adda" \
    --run_training $RUN_TRAINING \
    --run_testing $RUN_TESTING \
    --lr $LR \
    --max_epochs $MAX_EPOCHS \
    --pretrain_epochs $PRETRAIN_EPOCHS \
    --scheduler_type "$SCHEDULER_TYPE" \
    --warmup_ratio $WARMUP_RATIO \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --devices $DEVICES \
    --precision "$PRECISION" \
    --log_every_n_steps $LOG_EVERY_N_STEPS \
    ${VAL_CHECK_INTERVAL:+--val_check_interval $VAL_CHECK_INTERVAL} \
    ${CHECK_VAL_EVERY_N_EPOCH:+--check_val_every_n_epoch $CHECK_VAL_EVERY_N_EPOCH} \
    --patience $PATIENCE \
    --monitor_metric "$MONITOR_METRIC" \
    --mode "$MODE" \
    --save_top_k $SAVE_TOP_K \
    ${PRETRAIN_PATIENCE:+--pretrain_patience $PRETRAIN_PATIENCE} \
    --pretrain_monitor_metric "$PRETRAIN_MONITOR_METRIC" \
    --pretrain_mode "$PRETRAIN_MODE" \
    ${ADVERSARIAL_PATIENCE:+--adversarial_patience $ADVERSARIAL_PATIENCE} \
    ${ADVERSARIAL_MONITOR_METRIC:+--adversarial_monitor_metric "$ADVERSARIAL_MONITOR_METRIC"} \
    ${ADVERSARIAL_MODE:+--adversarial_mode "$ADVERSARIAL_MODE"} \
    --dropout $DROPOUT \
    --activation "$ACTIVATION" \
    --hidden_dims "$HIDDEN_DIMS" \
    --hidden_size $HIDDEN_SIZE \
    --patch_size $PATCH_SIZE \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --num_heads_cross_attn $NUM_HEADS_CROSS_ATTN \
    --num_layers_cross_attn $NUM_LAYERS_CROSS_ATTN \
    --num_heads_encoder $NUM_HEADS_ENCODER \
    --num_layers_encoder $NUM_LAYERS_ENCODER \
    --bingings "$BINGINGS" \
    ${CHECKPOINT_PATH:+--checkpoint_path "$CHECKPOINT_PATH"}
else
    echo ""
    echo "2. Skipping ADDA training (RUN_ADDA=0)"
fi

# ==========================================
# Step 3: Run DANN training
# ==========================================
if [ "$RUN_DANN" -eq 1 ]; then
    echo ""
    echo "3. Running DANN training..."

    $PYTHON $DANN_SCRIPT \
    --train_models "$TRAIN_MODELS" \
    --data_path "$NEW_DATA_PATH" \
    --default_root_dir "${DEFAULT_ROOT_DIR_BASE}/dann" \
    --run_training $RUN_TRAINING \
    --run_testing $RUN_TESTING \
    --lr $LR \
    --max_epochs $MAX_EPOCHS \
    --scheduler_type "$SCHEDULER_TYPE" \
    --warmup_ratio $WARMUP_RATIO \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --devices $DEVICES \
    --precision "$PRECISION" \
    --log_every_n_steps $LOG_EVERY_N_STEPS \
    ${VAL_CHECK_INTERVAL:+--val_check_interval $VAL_CHECK_INTERVAL} \
    ${CHECK_VAL_EVERY_N_EPOCH:+--check_val_every_n_epoch $CHECK_VAL_EVERY_N_EPOCH} \
    --patience $PATIENCE \
    --monitor_metric "$MONITOR_METRIC" \
    --mode "$MODE" \
    --save_top_k $SAVE_TOP_K \
    --gamma $DANN_GAMMA \
    --dropout $DROPOUT \
    --activation "$ACTIVATION" \
    --hidden_dims "$HIDDEN_DIMS" \
    --hidden_size $HIDDEN_SIZE \
    --patch_size $PATCH_SIZE \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --num_heads_cross_attn $NUM_HEADS_CROSS_ATTN \
    --num_layers_cross_attn $NUM_LAYERS_CROSS_ATTN \
    --num_heads_encoder $NUM_HEADS_ENCODER \
    --num_layers_encoder $NUM_LAYERS_ENCODER \
    --bingings "$BINGINGS" \
    ${CHECKPOINT_PATH:+--checkpoint_path "$CHECKPOINT_PATH"}
else
    echo ""
    echo "3. Skipping DANN training (RUN_DANN=0)"
fi

# ==========================================
# Step 4: Run DSN training
# ==========================================
if [ "$RUN_DSN" -eq 1 ]; then
    echo ""
    echo "4. Running DSN training..."

    $PYTHON $DSN_SCRIPT \
    --train_models "$TRAIN_MODELS" \
    --data_path "$NEW_DATA_PATH" \
    --default_root_dir "${DEFAULT_ROOT_DIR_BASE}/dsn" \
    --run_training $RUN_TRAINING \
    --run_testing $RUN_TESTING \
    --lr $LR \
    --max_epochs $MAX_EPOCHS \
    --scheduler_type "$SCHEDULER_TYPE" \
    --warmup_ratio $WARMUP_RATIO \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --devices $DEVICES \
    --precision "$PRECISION" \
    --log_every_n_steps $LOG_EVERY_N_STEPS \
    ${VAL_CHECK_INTERVAL:+--val_check_interval $VAL_CHECK_INTERVAL} \
    ${CHECK_VAL_EVERY_N_EPOCH:+--check_val_every_n_epoch $CHECK_VAL_EVERY_N_EPOCH} \
    --patience $PATIENCE \
    --monitor_metric "$MONITOR_METRIC" \
    --mode "$MODE" \
    --save_top_k $SAVE_TOP_K \
    --alpha $DSN_ALPHA \
    --beta $DSN_BETA \
    --gamma $DSN_GAMMA \
    --dropout $DROPOUT \
    --activation "$ACTIVATION" \
    --hidden_dims "$HIDDEN_DIMS" \
    --hidden_size $HIDDEN_SIZE \
    --patch_size $PATCH_SIZE \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --num_heads_cross_attn $NUM_HEADS_CROSS_ATTN \
    --num_layers_cross_attn $NUM_LAYERS_CROSS_ATTN \
    --num_heads_encoder $NUM_HEADS_ENCODER \
    --num_layers_encoder $NUM_LAYERS_ENCODER \
    --bingings "$BINGINGS" \
    ${CHECKPOINT_PATH:+--checkpoint_path "$CHECKPOINT_PATH"}
else
    echo ""
    echo "4. Skipping DSN training (RUN_DSN=0)"
fi

# ==========================================
# Step 5: Run Usual training
# ==========================================
if [ "$RUN_USUAL" -eq 1 ]; then
    echo ""
    echo "5. Running Usual training..."

    $PYTHON $USUAL_SCRIPT \
    --train_domains "$USUAL_TRAIN_DOMAINS" \
    --train_models "$TRAIN_MODELS" \
    --data_path "$NEW_DATA_PATH" \
    --default_root_dir "${DEFAULT_ROOT_DIR_BASE}/usual" \
    --run_training $RUN_TRAINING \
    --run_testing $RUN_TESTING \
    --lr $LR \
    --max_epochs $MAX_EPOCHS \
    --scheduler_type "$SCHEDULER_TYPE" \
    --warmup_ratio $WARMUP_RATIO \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --devices $DEVICES \
    --precision "$PRECISION" \
    --log_every_n_steps $LOG_EVERY_N_STEPS \
    ${VAL_CHECK_INTERVAL:+--val_check_interval $VAL_CHECK_INTERVAL} \
    ${CHECK_VAL_EVERY_N_EPOCH:+--check_val_every_n_epoch $CHECK_VAL_EVERY_N_EPOCH} \
    --patience $PATIENCE \
    --monitor_metric "val/accuracy" \
    --mode "max" \
    --save_top_k $SAVE_TOP_K \
    --dropout $DROPOUT \
    --activation "$ACTIVATION" \
    --hidden_dims "$HIDDEN_DIMS" \
    --hidden_size $HIDDEN_SIZE \
    --patch_size $PATCH_SIZE \
    --num_heads $NUM_HEADS \
    --num_layers $NUM_LAYERS \
    --num_heads_cross_attn $NUM_HEADS_CROSS_ATTN \
    --num_layers_cross_attn $NUM_LAYERS_CROSS_ATTN \
    --num_heads_encoder $NUM_HEADS_ENCODER \
    --num_layers_encoder $NUM_LAYERS_ENCODER \
    --bingings "$BINGINGS" \
    ${CHECKPOINT_PATH:+--checkpoint_path "$CHECKPOINT_PATH"}
else
    echo ""
    echo "5. Skipping Usual training (RUN_USUAL=0)"
fi


# CP this file to ouput path
OUTPUT_PATH=${DEFAULT_ROOT_DIR_BASE}/C${N_CELLS}G${N_GENES}H${N_HIGHLY_VARIABLE_GENES}S${CLASS_SEP}.sh
cp $0 $OUTPUT_PATH

echo ""
echo "=========================================="
echo "All training scripts completed!"
echo "=========================================="
