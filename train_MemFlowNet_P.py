from __future__ import print_function, division
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import core.datasets_video as datasets
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger
from core.utils.logger import Logger
import random
from core.Networks import build_network
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import evaluate_MemFlowNet_predict
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(gpu, cfg):
    rank = cfg.node_rank * cfg.gpus + gpu
    torch.cuda.set_device(rank)

    if cfg.DDP:
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=cfg.world_size,
                                rank=rank,
                                group_name='mtorch')
        model = nn.SyncBatchNorm.convert_sync_batchnorm(build_network(cfg)).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    loss_func = sequence_loss

    if 'freeze_encoder' in cfg and cfg.freeze_encoder:
        print("[Freeze feature, context  and qk encoder]")
        for param in model.module.cnet.parameters():
            param.requires_grad = False
        for param in model.module.fnet.parameters():
            param.requires_grad = False
        for param in model.module.att.parameters():
            param.requires_grad = False

    if rank == 0:
        loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt

        current_dict = model.module.state_dict()
        pretrained_dict = {k.replace('module.', ''): v for k, v in ckpt_model.items() if k.replace('module.', '') in current_dict}
        if cfg.restore_steps > 0 or cfg.eval_only:
            model.module.cnet = model.module.cnet.reset_output(128)

        missing_unexpected = model.module.load_state_dict(pretrained_dict, strict=False)
        if rank == 0:
            print(missing_unexpected)
        if cfg.restore_steps == 0:
            model.module.cnet = model.module.cnet.reset_output(128)

    model.train()

    if cfg.eval_only:
        # loading optical flow estimate model
        from configs.things_memflownet import get_cfg
        estimate_cfg = get_cfg()
        estimate_cfg.restore_ckpt = 'ckpts/MemFlowNet_things.pth'
        estimate_model = nn.SyncBatchNorm.convert_sync_batchnorm(build_network(estimate_cfg)).cuda()
        estimate_model = nn.parallel.DistributedDataParallel(estimate_model, device_ids=[rank])
        print("[Loading ckpt from {}]".format(estimate_cfg.restore_ckpt))
        ckpt = torch.load(estimate_cfg.restore_ckpt, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            estimate_model.load_state_dict(ckpt_model, strict=True)
        else:
            estimate_model.module.load_state_dict(ckpt_model, strict=True)

        if rank == 0:
            for val_dataset in cfg.validation:
                results = {}
                if val_dataset == 'sintel_train':
                    results.update(evaluate_MemFlowNet_predict.validate_sintel(model.module, cfg, rank,
                                                                               estimate_model.module, estimate_cfg))
                elif val_dataset == 'things':
                    results.update(evaluate_MemFlowNet_predict.validate_things(model.module, cfg, rank,
                                                                               estimate_model.module, estimate_cfg))
                elif val_dataset == 'kitti':
                    results.update(evaluate_MemFlowNet_predict.validate_kitti(model.module, cfg, rank,
                                                                               estimate_model.module, estimate_cfg))
                print(results)
        return

    if cfg.DDP:
        train_sampler, train_loader = datasets.fetch_dataloader(cfg, DDP=cfg.DDP, rank=rank)
    else:
        train_loader = datasets.fetch_dataloader(cfg, DDP=cfg.DDP, rank=rank)

    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    epoch = 0
    if cfg.restore_steps > 1:
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.total_steps = cfg.restore_steps - 1
        total_steps = cfg.restore_steps
        epoch = ckpt['epoch']
        for _ in range(total_steps):
            scheduler.step()

    should_keep_training = True
    while should_keep_training:

        epoch += 1
        if cfg.DDP:
            train_sampler.set_epoch(epoch)

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            if 'concat_flow' in cfg[cfg.network] and cfg[cfg.network].concat_flow:
                images, flows, valids, forward_warped_flow = [x.cuda() for x in data_blob]
            else:
                images, flows, valids = [x.cuda() for x in data_blob]
            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                images = (images + stdv * torch.randn(*images.shape).cuda()).clamp(0.0, 255.0)

            output = {}
            # flow prediction
            images = 2 * (images / 255.0) - 1.0
            b = images.shape[0]
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision, dtype=torch.bfloat16):
                # B*C*N-1*H*W,                    B*N-1*C*H*W
                query, key, inp = model.module.encode_context(images[:, :-1, ...])

                coords0, coords1, fmaps = model.module.encode_features(images)
                values = None
                video_flow_predictions = []  # frame by frame
                for ti in range(0, cfg.input_frames - 1):
                    if ti > 0:
                        flow = flows[:, ti-1]
                        new_size = (flow.shape[2] // 8, flow.shape[3] // 8)
                        flow = F.interpolate(flow, size=new_size, mode='bilinear', align_corners=True) / 8
                        current_value = model.module.get_motion_feature(flow, coords1, fmaps[:, ti-1:ti+1])
                        current_value = current_value.unsqueeze(2)
                        values = current_value if values is None else torch.cat([values, current_value], dim=2)

                    if ti <= cfg.num_ref_frames:
                        ref_values = values
                        ref_keys = key[:, :, :ti]
                    else:
                        indices = [torch.randperm(ti)[:cfg.num_ref_frames] for _ in range(b)]
                        ref_values = torch.stack([
                            values[bi, :, indices[bi]] for bi in range(b)
                        ], 0)
                        ref_keys = torch.stack([
                            key[bi, :, indices[bi]] for bi in range(b)
                        ], 0)

                    # predict flow from frame ti to frame ti+1
                    if 'concat_flow' in cfg[cfg.network] and cfg[cfg.network].concat_flow:
                        flow_pr = model.module.predict_flow(inp[:, ti], query[:, :, ti], ref_keys, ref_values, forward_warp_flow=forward_warped_flow[:, ti])
                    else:
                        flow_pr = model.module.predict_flow(inp[:, ti], query[:, :, ti], ref_keys, ref_values)

                    video_flow_predictions.append(flow_pr.unsqueeze(0))
                # loss function
                video_flow_predictions = torch.stack(video_flow_predictions, dim=2)  # Iter, B, N-1, 2, H, W

                loss, metrics, _ = loss_func(video_flow_predictions, flows, valids, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            metrics['scale'] = scaler.get_scale()
            if rank == 0:
                logger.push(metrics)

            if total_steps % cfg.val_freq == cfg.val_freq - 1 and rank == 0:
                print('start validation')
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps + 1, cfg.name)
                torch.save({
                    'iteration': total_steps,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'model': model.module.state_dict(),
                }, PATH)
                results = {}
                for val_dataset in cfg.validation:
                    if val_dataset == 'sintel_train':
                        results.update(evaluate_MemFlowNet_predict.validate_sintel(model.module, cfg, rank))
                    elif val_dataset == 'kitti':
                        results.update(evaluate_MemFlowNet_predict.validate_kitti(model.module, cfg, rank))
                logger.write_dict(results)
                model.train()

            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()
    if rank == 0:
        PATH = cfg.log_dir + f'/{cfg.name}.pth'
        torch.save(model.module.state_dict(), PATH)
    cleanup()
    return PATH


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MemFlowNet_P', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    # DDP
    parser.add_argument('--nodes', type=int, default=1, help='how many machines')
    parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--node_rank', type=int, default=0, help='the id of this machine')
    parser.add_argument('--DDP', action='store_true', help='DDP')
    parser.add_argument('--eval_only', action='store_true', default=False, help='eval only')

    args = parser.parse_args()

    if args.stage == 'things':
        from configs.things_memflownet_p import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel_memflownet_p import get_cfg

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    if args.DDP:
        args.world_size = args.nodes * args.gpus
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '22324'
    else:
        args.world_size = 1

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    if not cfg.eval_only:
        loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    # initialize random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    mp.spawn(train, nprocs=args.world_size, args=(cfg,))
