from __future__ import print_function, division
import argparse
from loguru import logger as loguru_logger
import random
from core.Networks import build_network
import sys
sys.path.append('core')
from PIL import Image
import os
import numpy as np
import torch
from utils import flow_viz
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
from inference import inference_core_skflow as inference_core


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def inference(cfg):
    model = build_network(cfg).cuda()
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
        if 'module' in list(ckpt_model.keys())[0]:
            for key in ckpt_model.keys():
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            model.load_state_dict(ckpt_model, strict=True)
        else:
            model.load_state_dict(ckpt_model, strict=True)

    model.eval()

    print(f"preparing image...")
    print(f"Input image sequence dir = {cfg.seq_dir}")
    image_list = sorted(os.listdir(cfg.seq_dir))

    imgs = [frame_utils.read_gen(os.path.join(cfg.seq_dir, path)) for path in image_list]
    imgs = [np.array(img).astype(np.uint8) for img in imgs]
    # grayscale images
    if len(imgs[0].shape) == 2:
        imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
    else:
        imgs = [img[..., :3] for img in imgs]
    imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]

    images = torch.stack(imgs)

    processor = inference_core.InferenceCore(model, config=cfg)
    # 1, T, C, H, W
    images = images.cuda().unsqueeze(0)

    padder = InputPadder(images.shape)
    images = padder.pad(images)

    images = 2 * (images / 255.0) - 1.0
    flow_prev = None
    results = []
    print(f"start inference...")
    for ti in range(images.shape[1] - 1):
        flow_low, flow_pre = processor.step(images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                                            add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
        flow_pre = padder.unpad(flow_pre[0]).cpu()
        results.append(flow_pre)
        if 'warm_start' in cfg and cfg.warm_start:
            flow_prev = forward_interpolate(flow_low[0])[None].cuda()

    if not os.path.exists(cfg.vis_dir):
        os.makedirs(cfg.vis_dir)

    print(f"save results...")
    N = len(results)
    for idx in range(N):
        flow_img = flow_viz.flow_to_image(results[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{:04}_to_{:04}.png'.format(cfg.vis_dir, idx + 1, idx + 2))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='MemFlowNet', choices=['MemFlowNet', 'MemFlowNet_T'], help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--seq_dir', default='default')
    parser.add_argument('--vis_dir', default='default')

    args = parser.parse_args()
    if args.name == "MemFlowNet":
        if args.stage == 'things':
            from configs.things_memflownet import get_cfg
        elif args.stage == 'sintel':
            from configs.sintel_memflownet import get_cfg
        elif args.stage == 'spring_only':
            from configs.spring_memflownet import get_cfg
        elif args.stage == 'kitti':
            from configs.kitti_memflownet import get_cfg
        else:
            raise NotImplementedError
    elif args.name == "MemFlowNet_T":
        if args.stage == 'things':
            from configs.things_memflownet_t import get_cfg
        elif args.stage == 'things_kitti':
            from configs.things_memflownet_t_kitti import get_cfg
        elif args.stage == 'sintel':
            from configs.sintel_memflownet_t import get_cfg
        elif args.stage == 'kitti':
            from configs.kitti_memflownet_t import get_cfg
        else:
            raise NotImplementedError

    cfg = get_cfg()
    cfg.update(vars(args))

    # initialize random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    inference(cfg)
