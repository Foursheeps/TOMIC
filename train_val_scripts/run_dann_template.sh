PYTHON=/home/LYan/miniconda3/envs/clip/bin/python
SCRIPT=/your/path/to/TOMIC/train_val_scripts/main_dann.py
DATA_PATH=${DATA_PATH:-/your/path/to/TOMIC/process_data/Synthetic/C7000G1200H400S10}
DEFAULT_ROOT_DIR=${DEFAULT_ROOT_DIR_BASE:-/your/path/to/TOMIC/train_val_scripts/outputs}/dann

$PYTHON $SCRIPT \
    --train_models "['mlp', 'patch', 'expr', 'name', 'dual']" \
    --data_path $DATA_PATH \
    --default_root_dir $DEFAULT_ROOT_DIR \
    --run_training 0 \
    --run_testing 0 \
    --devices 2 \
    --train_batch_size 256 \
    --max_epochs 80 \
    --patience 10 \
    --gamma 0.1 \
    --patch_size 40 \
    --bingings "[None, 50]"
