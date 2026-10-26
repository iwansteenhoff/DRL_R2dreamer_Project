#!/bin/bash
# Experiment 2: Reacher hard with images + full state (position, velocity, to_target).
# The state alone is enough to solve the task here. The image-only baseline is the
# existing env=dmc_vision env.task=dmc_reacher_hard run, so it is not trained again.
# Run from the repository root, inside tmux:
#   bash runs/reacher_state_sufficient.sh [seed] [env steps]
# Running it again with a bigger step budget continues the saved runs, e.g.
#   bash runs/reacher_state_sufficient.sh 0 1.01e6

# ==== Settings ====
GPU_ID=0
SEED=${1:-0}
STEPS=${2:-5.1e5}
DROPOUT=0.5 # chance that a training sequence loses either the image or the state
export CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU_ID
PYTHON=.venv/bin/python

PLAIN=logdir/reacher_hard_images_state_seed${SEED}
DROP=logdir/reacher_hard_images_state_dropout_seed${SEED}

train_and_evaluate() {
    logdir=$1
    shift
    # continue from the last checkpoint if this run was started before
    resume=False
    if [ -f "$logdir/latest.pt" ]; then
        resume=True
    fi
    $PYTHON train.py env=dmc_multimodal seed=$SEED logdir=$logdir device=cuda:0 env.steps=$STEPS resume=$resume "$@"
    # one ablation folder per step budget, so later evaluations never clash with earlier ones
    $PYTHON evaluate_ablations.py $logdir --episodes 100 --seed-start 100000 --videos 3 \
        --output-dir $logdir/ablations_steps${STEPS}
}

# images + state, no dropout
train_and_evaluate $PLAIN

# images + state with modality dropout
train_and_evaluate $DROP model.modality_dropout=$DROPOUT
