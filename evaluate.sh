#!/bin/bash
# MemFlow
python -u train_MemFlowNet.py --name MemFlowNet --stage things --validation kitti sintel_train --restore_ckpt ckpts/MemFlowNet_things.pth --eval_only --DDP
python -u train_MemFlowNet.py --name MemFlowNet --stage sintel --validation sintel_submission --restore_ckpt ckpts/MemFlowNet_sintel.pth --eval_only --DDP
python -u train_MemFlowNet.py --name MemFlowNet --stage kitti --validation kitti_submission --restore_ckpt ckpts/MemFlowNet_kitti.pth --eval_only --DDP
python -u train_MemFlowNet.py --name MemFlowNet --stage spring_only --validation spring_submission --restore_ckpt ckpts/MemFlowNet_spring.pth --eval_only --DDP

# MemFlow-T
python -u train_MemFlowNet_T.py --name MemFlowNet_T --stage things --validation sintel_train --restore_ckpt ckpts/MemFlowNet_T_things.pth --eval_only --DDP
python -u train_MemFlowNet_T.py --name MemFlowNet_T --stage things_kitti --validation kitti --restore_ckpt ckpts/MemFlowNet_T_things_kitti.pth --eval_only --DDP
python -u train_MemFlowNet_T.py --name MemFlowNet_T --stage sintel --validation sintel_submission --restore_ckpt ckpts/MemFlowNet_T_sintel.pth --eval_only --DDP
python -u train_MemFlowNet_T.py --name MemFlowNet_T --stage kitti --validation kitti_submission --restore_ckpt ckpts/MemFlowNet_T_kitti.pth --eval_only --DDP

# MemFlow-P
python -u train_MemFlowNet_P.py --name MemFlowNet_P --stage things --validation kitti sintel_train things --restore_ckpt ckpts/MemFlowNet_P_things.pth --eval_only --DDP