#!/bin/bash
# MemFlow
python -u train_MemFlowNet.py --name MemFlowNet --stage things --validation sintel_train --restore_ckpt ckpts/skflow-things.pth --gpus 2 --GPU_ids 0,1 --DDP
python -u train_MemFlowNet.py --name MemFlowNet --stage sintel --restore_ckpt ckpts/MemFlowNet_things.pth --gpus 2 --GPU_ids 0,1 --DDP
python -u train_MemFlowNet.py --name MemFlowNet --stage kitti --restore_ckpt ckpts/MemFlowNet_sintel.pth --gpus 2 --GPU_ids 0,1 --DDP
python -u train_MemFlowNet.py --name MemFlowNet --stage spring_only --validation spring_subset_val --restore_ckpt ckpts/MemFlowNet_sintel.pth --gpus 2 --GPU_ids 0,1 --DDP

# MemFlow-T
python -u train_MemFlowNet_T.py --name MemFlowNet_T --stage things --validation sintel_train --restore_ckpt ckpts/twins_skflow.pth --gpus 2 --GPU_ids 0,1 --DDP
python -u train_MemFlowNet_T.py --name MemFlowNet_T --stage things_kitti --validation kitti --restore_ckpt ckpts/twins_skflow.pth --gpus 2 --GPU_ids 0,1 --DDP
python -u train_MemFlowNet_T.py --name MemFlowNet_T --stage sintel --restore_ckpt ckpts/MemFlowNet_T_things.pth --gpus 2 --GPU_ids 0,1 --DDP
python -u train_MemFlowNet_T.py --name MemFlowNet_T --stage kitti --restore_ckpt ckpts/MemFlowNet_T_sintel.pth --gpus 2 --GPU_ids 0,1 --DDP

# MemFlow-P
python -u train_MemFlowNet_P.py --name MemFlowNet_P --stage things --restore_ckpt ckpts/skflow-things.pth --gpus 1 --GPU_ids 0 --DDP