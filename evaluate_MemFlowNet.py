import sys

sys.path.append('core')

from PIL import Image
import os
import numpy as np
import torch
from utils import flow_viz
import core.datasets_video as datasets
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from inference import inference_core_skflow as inference_core
from tqdm import tqdm
import imageio


@torch.no_grad()
def validate_sintel(model, cfg, rank=0):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    for dstype in ['final', "clean"]:
        val_dataset = datasets.MpiSintelTrain(dstype=dstype, return_gt=True)
        val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

        epe_list = []

        for item in tqdm(val_loader):
            processor = inference_core.InferenceCore(model, config=cfg)
            # 1, T, C, H, W
            images, flows, extra_info = item
            images = images.cuda()

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            images = 2 * (images / 255.0) - 1.0
            flow_prev = None
            for ti in range(images.shape[1]-1):
                flow_low, flow_pre = processor.step(images[:, ti:ti+2], end=(ti == images.shape[1]-2),
                                                    add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
                flow_pre = padder.unpad(flow_pre[0]).cpu()
                epe = torch.sum((flow_pre - flows[0, ti]) ** 2, dim=0).sqrt()
                epe_list.append(epe.view(-1).numpy())

                if 'warm_start' in cfg and cfg.warm_start:
                    flow_prev = forward_interpolate(flow_low[0])[None].cuda()

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = epe

    return results


@torch.no_grad()
def validate_spring(model, cfg, rank=0, split='train'):
    """ Peform validation using the Spring (train) split """

    model.eval()
    results = {}

    val_dataset = datasets.SpringTrain(return_gt=True, split=split)
    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=1)

    epe_list = []
    epe_list_10 = []
    epe_list_10_40 = []
    epe_list_40 = []

    for item in tqdm(val_loader):
        processor = inference_core.InferenceCore(model, config=cfg)
        # 1, T, C, H, W
        images, flows, extra_info = item
        images = images.cuda()

        padder = InputPadder(images.shape)
        images = padder.pad(images)

        images = 2 * (images / 255.0) - 1.0

        valid = ~torch.isnan(torch.sum(flows, dim=2))
        valid_10 = valid & (torch.sum(flows ** 2, dim=2).sqrt() < 10)
        valid_10_40 = valid & (torch.sum(flows ** 2, dim=2).sqrt() >= 10) & (torch.sum(flows ** 2, dim=2).sqrt() < 40)
        valid_40 = valid & (torch.sum(flows ** 2, dim=2).sqrt() >= 40)

        flow_prev = None
        for ti in range(images.shape[1]-1):
            flow_low, flow_pre = processor.step(images[:, ti:ti+2], end=(ti == images.shape[1]-2),
                                                add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
            flow_pre = padder.unpad(flow_pre[0]).cpu()

            epe = torch.sum((flow_pre - flows[0, ti]) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1)[valid[0, ti].view(-1)].numpy())
            epe_list_10.append(epe.view(-1)[valid_10[0, ti].view(-1)].numpy())
            epe_list_10_40.append(epe.view(-1)[valid_10_40[0, ti].view(-1)].numpy())
            epe_list_40.append(epe.view(-1)[valid_40[0, ti].view(-1)].numpy())

            if 'warm_start' in cfg and cfg.warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all > 1)

    px1_10 = np.mean(np.concatenate(epe_list_10) > 1)
    px1_10_40 = np.mean(np.concatenate(epe_list_10_40) > 1)
    px1_40 = np.mean(np.concatenate(epe_list_40) > 1)

    print("Validation EPE: %f, 1px: %f, 1px(s0~10): %f, 1px(s10~40): %f, 1px(s40+): %f" % (epe, px1, px1_10, px1_10_40, px1_40))
    results['spring_epe'] = epe
    results['spring_px1'] = px1
    results['spring_px1_10'] = px1_10
    results['spring_px1_10_40'] = px1_10_40
    results['spring_px1_40'] = px1_40

    return results


@torch.no_grad()
def create_spring_submission(model, cfg, output_path='output', rank=0):
    """ Create submission for the Spring leaderboard """
    results = {}
    model.eval()

    test_dataset = datasets.Spring_submission(return_gt=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=cfg.world_size, rank=rank, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=1,
                                  sampler=test_sampler)

    for item in tqdm(test_loader):
        processor = inference_core.InferenceCore(model, config=cfg)
        # 1, T, C, H, W
        images, (frame, sequence, dir, cam) = item
        sequence = sequence[0]
        dir = dir[0]
        cam =cam[0]
        images = images.cuda()

        padder = InputPadder(images.shape)
        images = padder.pad(images)

        images = 2 * (images / 255.0) - 1.0
        flow_prev = None
        for ti in range(images.shape[1] - 1):
            flow_low, flow_pre = processor.step(images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                                                add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
            flow_pre = padder.unpad(flow_pre[0]).cpu()

            if 'warm_start' in cfg and cfg.warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            _flow = flow_pre.permute(1, 2, 0).numpy()

            output_dir = os.path.join('spring_submission', output_path, sequence, f'flow_{dir}_{cam}')
            output_file = os.path.join(output_dir, f'flow_{dir}_{cam}_%04d.flo5' % frame[ti])

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlo5File(_flow, output_file)

    return results


@torch.no_grad()
def validate_things(model, cfg, rank=0):
    """ Peform validation using the things (train) split """

    model.eval()
    results = {}

    for dstype in ['frames_finalpass', "frames_cleanpass"]:
        val_dataset = datasets.ThingsTEST(dstype=dstype, return_gt=True)
        val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

        epe_list = []

        for item in tqdm(val_loader):
            processor = inference_core.InferenceCore(model, config=cfg)
            # 1, T, C, H, W
            images, flows, extra_info = item
            images = images.cuda()

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            images = 2 * (images / 255.0) - 1.0
            valid = torch.sum(flows ** 2, dim=2).sqrt() < 400
            flow_prev = None
            for ti in range(images.shape[1]-1):
                flow_low, flow_pre = processor.step(images[:, ti:ti+2], end=(ti == images.shape[1]-2),
                                                    add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
                flow_pre = padder.unpad(flow_pre[0]).cpu()
                epe = torch.sum((flow_pre - flows[0, ti]) ** 2, dim=0).sqrt()
                epe_list.append(epe.view(-1)[valid[0, ti].view(-1)].numpy())
                if 'warm_start' in cfg and cfg.warm_start:
                    flow_prev = forward_interpolate(flow_low[0])[None].cuda()

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = epe
    return results


@torch.no_grad()
def validate_kitti(model, cfg, rank=0):
    """ Peform validation using the KITTI (train) split """

    model.eval()

    val_dataset = datasets.KITTITest(aug_params=None)
    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

    epe_list = []
    out_list = []

    for item in tqdm(val_loader):
        processor = inference_core.InferenceCore(model, config=cfg)
        # 1, T, C, H, W
        images, flows, valids = item
        images = images.cuda()

        padder = InputPadder(images.shape)
        images = padder.pad(images)

        images = 2 * (images / 255.0) - 1.0
        flow_prev = None
        for ti in range(images.shape[1] - 1):
            flow_low, flow_pre = processor.step(images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                                                add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
            if 'warm_start' in cfg and cfg.warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
        flow_pre = padder.unpad(flow_pre[0]).cpu()
        epe = torch.sum((flow_pre - flows[0, -1]) ** 2, dim=0).sqrt()

        mag = torch.sum(flows[0, -1] ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valids[0, -1].view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def create_sintel_submission(model, cfg, output_path='output', rank=0):
    """ Create submission for the Sintel leaderboard """
    results = {}
    model.eval()

    for dstype in ['final', 'clean']:
        test_dataset = datasets.MpiSintel_submission(dstype=dstype, return_gt=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=cfg.world_size, rank=rank, shuffle=False)
        test_loader = data.DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2,
                                      sampler=test_sampler)

        for item in tqdm(test_loader):
            processor = inference_core.InferenceCore(model, config=cfg)
            # 1, T, C, H, W
            images, (frame, sequence) = item
            sequence = sequence[0]
            images = images.cuda()

            padder = InputPadder(images.shape)
            images = padder.pad(images)

            images = 2 * (images / 255.0) - 1.0
            flow_prev = None
            for ti in range(images.shape[1] - 1):
                flow_low, flow_pre = processor.step(images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                                                    add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)
                flow_pre = padder.unpad(flow_pre[0]).cpu()

                if 'warm_start' in cfg and cfg.warm_start:
                    flow_prev = forward_interpolate(flow_low[0])[None].cuda()

                _flow = flow_pre.permute(1, 2, 0).numpy()

                flow_img = flow_viz.flow_to_image(_flow)
                image = Image.fromarray(flow_img)
                if not os.path.exists(f'vis_sintel/{output_path}/{dstype}/flow/{sequence}'):
                   os.makedirs(f'vis_sintel/{output_path}/{dstype}/flow/{sequence}')
                if not os.path.exists(f'vis_sintel/gt/{dstype}/image/{sequence}'):
                   os.makedirs(f'vis_sintel/gt/{dstype}/image/{sequence}')

                image.save(f'vis_sintel/{output_path}/{dstype}/flow/{sequence}/{ti}.png')
                imageio.imwrite(f'vis_sintel/gt/{dstype}/image/{sequence}/{ti}.png', ((images[0, ti].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8))

                output_dir = os.path.join('sintel_submission', output_path, dstype, sequence)
                output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame[ti] + 1))

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                frame_utils.writeFlow(output_file, _flow)

    return results


@torch.no_grad()
def create_kitti_submission(model, cfg, output_path='output', rank=0):
    model.eval()
    results = {}

    test_dataset = datasets.KITTISubmission(return_gt=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=cfg.world_size, rank=rank, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2,
                                  sampler=test_sampler)

    for item in tqdm(test_loader):
        processor = inference_core.InferenceCore(model, config=cfg)
        # 1, T, C, H, W
        images, frame_id = item
        images = images.cuda()

        padder = InputPadder(images.shape)
        images = padder.pad(images)

        images = 2 * (images / 255.0) - 1.0
        flow_prev = None
        for ti in range(images.shape[1] - 1):
            flow_low, flow_pre = processor.step(images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                                                add_pe=('rope' in cfg and cfg.rope), flow_init=flow_prev)

            if 'warm_start' in cfg and cfg.warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

        flow_pre = padder.unpad(flow_pre[0]).cpu()
        _flow = flow_pre.permute(1, 2, 0).numpy()

        flow_img = flow_viz.flow_to_image(_flow)
        image = Image.fromarray(flow_img)

        if not os.path.exists(f'vis_kitti/{output_path}'):
            os.makedirs(f'vis_kitti/{output_path}')

        image.save(f'vis_kitti/{output_path}/{frame_id[0]}')

        output_dir = os.path.join('kitti_submission', output_path)
        output_file = os.path.join(output_dir, frame_id[0])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frame_utils.writeFlowKITTI(output_file, _flow)

    return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
