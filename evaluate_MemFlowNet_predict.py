import sys
sys.path.append('core')
import numpy as np
import torch
import core.datasets_video as datasets
from utils.utils import InputPadder, forward_interpolate
import torch.utils.data as data
from inference import inference_core_predict as inference_core
from inference import inference_core_skflow as estimate_inference_core
from tqdm import tqdm
import torch.nn.functional as F


@torch.no_grad()
def validate_sintel(model, cfg, rank=0, estimate_model=None, estimate_cfg=None):
    """ Peform validation using the Sintel (train) split """

    model.eval()

    if estimate_model is not None:
        estimate_model.eval()
        use_gt = False
    else:
        use_gt = True

    results = {}

    for dstype in ['final', "clean"]:
        val_dataset = datasets.MpiSintelTrain(dstype=dstype, return_gt=True)
        val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

        epe_list = []
        epe_list_10 = []
        epe_list_10_40 = []
        epe_list_40 = []
        epe_list_forward = []
        epe_list_forward_10 = []
        epe_list_forward_10_40 = []
        epe_list_forward_40 = []

        for item in tqdm(val_loader):
            processor = inference_core.InferenceCore(model, config=cfg)

            if not use_gt:
                estimate_processor = estimate_inference_core.InferenceCore(estimate_model, config=estimate_cfg)

            # 1, T, C, H, W
            images, flows, extra_info = item
            images = images.cuda()

            padder = InputPadder(images.shape)
            images = padder.pad(images)
            padded_flow = padder.pad(flows).cuda()

            images = 2 * (images / 255.0) - 1.0

            valid_10 = torch.sum(flows ** 2, dim=2).sqrt() < 10
            valid_10_40 = ~valid_10 & (torch.sum(flows ** 2, dim=2).sqrt() < 40)
            valid_40 = torch.sum(flows ** 2, dim=2).sqrt() >= 40

            new_size = (padded_flow[:, 0].shape[2] // 8, padded_flow[:, 0].shape[3] // 8)
            flow_prev = torch.zeros(1, 2, new_size[0], new_size[1]).cuda()
            for ti in range(images.shape[1]-1):
                flow_low, flow_pre = processor.step(images[:, ti], add_pe=('rope' in cfg and cfg.rope), forward_warp_flow=flow_prev)
                flow_pre = padder.unpad(flow_pre[0]).cpu()
                if ti > 0:
                    epe = torch.sum((flow_pre - flows[0, ti]) ** 2, dim=0).sqrt()
                    epe_list.append(epe.view(-1).numpy())
                    epe_list_10.append(epe.view(-1)[valid_10[0, ti].view(-1)].numpy())
                    epe_list_10_40.append(epe.view(-1)[valid_10_40[0, ti].view(-1)].numpy())
                    epe_list_40.append(epe.view(-1)[valid_40[0, ti].view(-1)].numpy())

                    if not use_gt:
                        forward_epe = torch.sum((estimate_flow_pre - flows[0, ti]) ** 2, dim=0).sqrt()
                        epe_list_forward.append(forward_epe.view(-1).numpy())
                        epe_list_forward_10.append(forward_epe.view(-1)[valid_10[0, ti].view(-1)].numpy())
                        epe_list_forward_10_40.append(forward_epe.view(-1)[valid_10_40[0, ti].view(-1)].numpy())
                        epe_list_forward_40.append(forward_epe.view(-1)[valid_40[0, ti].view(-1)].numpy())

                if use_gt:
                    flow = padded_flow[:, ti]
                    flow = F.interpolate(flow, size=new_size, mode='bilinear', align_corners=True) / 8
                else:
                    flow, estimate_flow_pre = estimate_processor.step(images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                                                      add_pe=('rope' in estimate_cfg and estimate_cfg.rope),
                                                      flow_init=flow_prev)
                    estimate_flow_pre = padder.unpad(estimate_flow_pre)
                    estimate_flow_pre = forward_interpolate(estimate_flow_pre[0])
                processor.set_memory(images[:, ti:ti + 2], flow, end=(ti == images.shape[1] - 2))
                flow_prev = forward_interpolate(flow[0])[None].cuda()

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        epe_all_10 = np.concatenate(epe_list_10)
        epe_10 = np.mean(epe_all_10)
        epe_all_10_40 = np.concatenate(epe_list_10_40)
        epe_10_40 = np.mean(epe_all_10_40)
        epe_all_40 = np.concatenate(epe_list_40)
        epe_40 = np.mean(epe_all_40)

        print("Validation (%s) EPE: %f, EPE(s0-10): %f, EPE(s10-40): %f, EPE(s40+): %f, 1px: %f, 3px: %f, 5px: %f" %
              (dstype, epe, epe_10, epe_10_40, epe_40, px1, px3, px5))

        results[dstype] = epe

        if not use_gt:
            epe_all_forward = np.concatenate(epe_list_forward)
            epe_forward = np.mean(epe_all_forward)
            px1_forward = np.mean(epe_all_forward < 1)
            px3_forward = np.mean(epe_all_forward < 3)
            px5_forward = np.mean(epe_all_forward < 5)

            epe_all_forward_10 = np.concatenate(epe_list_forward_10)
            epe_forward_10 = np.mean(epe_all_forward_10)
            epe_all_forward_10_40 = np.concatenate(epe_list_forward_10_40)
            epe_forward_10_40 = np.mean(epe_all_forward_10_40)
            epe_all_forward_40 = np.concatenate(epe_list_forward_40)
            epe_forward_40 = np.mean(epe_all_forward_40)

            print(
                "Validation (%s) EPE of forward warped flow: %f, EPE(s0-10): %f, EPE(s10-40): %f, EPE(s40+): %f, 1px: %f, 3px: %f, 5px: %f" % (
                    dstype, epe_forward, epe_forward_10, epe_forward_10_40, epe_forward_40, px1_forward,
                    px3_forward, px5_forward))

    return results


@torch.no_grad()
def validate_things(model, cfg, rank=0, estimate_model=None, estimate_cfg=None):
    """ Peform validation using the things (train) split """

    model.eval()

    if estimate_model is not None:
        estimate_model.eval()
        use_gt = False
    else:
        use_gt = True

    results = {}

    for dstype in ['frames_finalpass', "frames_cleanpass"]:
        val_dataset = datasets.ThingsTEST(dstype=dstype, return_gt=True)
        val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

        epe_list = []
        epe_list_10 = []
        epe_list_10_40 = []
        epe_list_40 = []
        epe_list_forward = []
        epe_list_forward_10 = []
        epe_list_forward_10_40 = []
        epe_list_forward_40 = []

        for item in tqdm(val_loader):
            processor = inference_core.InferenceCore(model, config=cfg)

            if not use_gt:
                estimate_processor = estimate_inference_core.InferenceCore(estimate_model, config=estimate_cfg)
            # 1, T, C, H, W
            images, flows, extra_info = item
            images = images.cuda()

            padder = InputPadder(images.shape)
            images = padder.pad(images)
            padded_flow = padder.pad(flows).cuda()

            images = 2 * (images / 255.0) - 1.0

            valid = torch.sum(flows ** 2, dim=2).sqrt() < 400
            valid_10 = torch.sum(flows ** 2, dim=2).sqrt() < 10
            valid_10_40 = ~valid_10 & (torch.sum(flows ** 2, dim=2).sqrt() < 40)
            valid_40 = valid & (torch.sum(flows ** 2, dim=2).sqrt() >= 40)

            new_size = (padded_flow[:, 0].shape[2] // 8, padded_flow[:, 0].shape[3] // 8)
            flow_prev = torch.zeros(1, 2, new_size[0], new_size[1]).cuda()

            for ti in range(images.shape[1]-1):
                flow_low, flow_pre = processor.step(images[:, ti], add_pe=('rope' in cfg and cfg.rope), forward_warp_flow=flow_prev)
                flow_pre = padder.unpad(flow_pre[0]).cpu()
                if ti > 0:
                    epe = torch.sum((flow_pre - flows[0, ti]) ** 2, dim=0).sqrt()
                    epe_list.append(epe.view(-1)[valid[0, ti].view(-1)].numpy())
                    epe_list_10.append(epe.view(-1)[valid_10[0, ti].view(-1)].numpy())
                    epe_list_10_40.append(epe.view(-1)[valid_10_40[0, ti].view(-1)].numpy())
                    epe_list_40.append(epe.view(-1)[valid_40[0, ti].view(-1)].numpy())
                    if not use_gt:
                        forward_epe = torch.sum((estimate_flow_pre - flows[0, ti]) ** 2, dim=0).sqrt()
                        epe_list_forward.append(forward_epe.view(-1)[valid[0, ti].view(-1)].numpy())
                        epe_list_forward_10.append(forward_epe.view(-1)[valid_10[0, ti].view(-1)].numpy())
                        epe_list_forward_10_40.append(forward_epe.view(-1)[valid_10_40[0, ti].view(-1)].numpy())
                        epe_list_forward_40.append(forward_epe.view(-1)[valid_40[0, ti].view(-1)].numpy())

                if use_gt:
                    flow = padded_flow[:, ti]
                    flow = F.interpolate(flow, size=new_size, mode='bilinear', align_corners=True) / 8
                else:
                    flow, estimate_flow_pre = estimate_processor.step(images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                                                      add_pe=('rope' in estimate_cfg and estimate_cfg.rope),
                                                      flow_init=flow_prev)
                    estimate_flow_pre = padder.unpad(estimate_flow_pre)
                    estimate_flow_pre = forward_interpolate(estimate_flow_pre[0])
                processor.set_memory(images[:, ti:ti + 2], flow, end=(ti == images.shape[1] - 2))
                flow_prev = forward_interpolate(flow[0])[None].cuda()

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        epe_all_10 = np.concatenate(epe_list_10)
        epe_10 = np.mean(epe_all_10)
        epe_all_10_40 = np.concatenate(epe_list_10_40)
        epe_10_40 = np.mean(epe_all_10_40)
        epe_all_40 = np.concatenate(epe_list_40)
        epe_40 = np.mean(epe_all_40)

        print("Validation (%s) EPE: %f, EPE(s0-10): %f, EPE(s10-40): %f, EPE(s40+): %f, 1px: %f, 3px: %f, 5px: %f" %
              (dstype, epe, epe_10, epe_10_40, epe_40, px1, px3, px5))
        results[dstype] = epe

        if not use_gt:
            epe_all_forward = np.concatenate(epe_list_forward)
            epe_forward = np.mean(epe_all_forward)
            px1_forward = np.mean(epe_all_forward < 1)
            px3_forward = np.mean(epe_all_forward < 3)
            px5_forward = np.mean(epe_all_forward < 5)

            epe_all_forward_10 = np.concatenate(epe_list_forward_10)
            epe_forward_10 = np.mean(epe_all_forward_10)
            epe_all_forward_10_40 = np.concatenate(epe_list_forward_10_40)
            epe_forward_10_40 = np.mean(epe_all_forward_10_40)
            epe_all_forward_40 = np.concatenate(epe_list_forward_40)
            epe_forward_40 = np.mean(epe_all_forward_40)

            print("Validation (%s) EPE of forward warped flow: %f, EPE(s0-10): %f, EPE(s10-40): %f, EPE(s40+): %f, 1px: %f, 3px: %f, 5px: %f" % (
                dstype, epe_forward, epe_forward_10, epe_forward_10_40, epe_forward_40, px1_forward, px3_forward, px5_forward))

    return results


@torch.no_grad()
def validate_kitti(model, cfg, rank=0, estimate_model=None, estimate_cfg=None):
    """ Peform validation using the Sintel (train) split """

    model.eval()

    if estimate_model is not None:
        estimate_model.eval()
        use_gt = False
    else:
        use_gt = True

    results = {}

    val_dataset = datasets.KITTITest(aug_params=None)
    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=2)

    epe_list = []
    epe_list_10 = []
    epe_list_10_40 = []
    epe_list_40 = []
    out_list = []
    epe_list_forward = []
    epe_list_forward_10 = []
    epe_list_forward_10_40 = []
    epe_list_forward_40 = []
    out_list_forward = []

    for item in tqdm(val_loader):
        processor = inference_core.InferenceCore(model, config=cfg)

        if not use_gt:
            estimate_processor = estimate_inference_core.InferenceCore(estimate_model, config=estimate_cfg)

        # 1, T, C, H, W
        images, flows, valids = item
        images = images.cuda()

        padder = InputPadder(images.shape)
        images = padder.pad(images)
        padded_flow = padder.pad(flows).cuda()

        images = 2 * (images / 255.0) - 1.0
        valid_10 = (valids >= 0.5) & (torch.sum(flows ** 2, dim=2).sqrt() < 10)
        valid_10_40 = (valids >= 0.5) & (torch.sum(flows ** 2, dim=2).sqrt() >= 10) & (torch.sum(flows ** 2, dim=2).sqrt() < 40)
        valid_40 = (valids >= 0.5) & (torch.sum(flows ** 2, dim=2).sqrt() >= 40)

        new_size = (padded_flow[:, 0].shape[2] // 8, padded_flow[:, 0].shape[3] // 8)
        flow_prev = torch.zeros(1, 2, new_size[0], new_size[1]).cuda()

        for ti in range(images.shape[1] - 1):
            flow_low, flow_pre = processor.step(images[:, ti], add_pe=('rope' in cfg and cfg.rope), forward_warp_flow=flow_prev)

            if ti < images.shape[1] - 2:
                if use_gt:
                    flow = flow_low
                else:
                    flow, estimate_flow_pre = estimate_processor.step(images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                                                      add_pe=('rope' in estimate_cfg and estimate_cfg.rope),
                                                      flow_init=flow_prev)
                processor.set_memory(images[:, ti:ti + 2], flow, end=(ti == images.shape[1] - 2))
                flow_prev = forward_interpolate(flow[0])[None].cuda()

        flow_pre = padder.unpad(flow_pre[0]).cpu()
        epe = torch.sum((flow_pre - flows[0, -1]) ** 2, dim=0).sqrt()

        mag = torch.sum(flows[0, -1] ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valids[0, -1].view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        if torch.sum(valid_10[0, -1]) > 0:
            epe_list_10.append(epe[valid_10[0, -1].view(-1)].mean().item())
        if torch.sum(valid_10_40[0, -1]) > 0:
            epe_list_10_40.append(epe[valid_10_40[0, -1].view(-1)].mean().item())
        if torch.sum(valid_40[0, -1]) > 0:
            epe_list_40.append(epe[valid_40[0, -1].view(-1)].mean().item())
        out_list.append(out[val].cpu().numpy())

        # forward warped flow
        estimate_flow_pre = padder.unpad(estimate_flow_pre)
        estimate_flow_pre = forward_interpolate(estimate_flow_pre[0])
        epe = torch.sum((estimate_flow_pre - flows[0, -1]) ** 2, dim=0).sqrt()

        epe = epe.view(-1)

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list_forward.append(epe[val].mean().item())
        if torch.sum(valid_10[0, -1]) > 0:
            epe_list_forward_10.append(epe[valid_10[0, -1].view(-1)].mean().item())
        if torch.sum(valid_10_40[0, -1]) > 0:
            epe_list_forward_10_40.append(epe[valid_10_40[0, -1].view(-1)].mean().item())
        if torch.sum(valid_40[0, -1]) > 0:
            epe_list_forward_40.append(epe[valid_40[0, -1].view(-1)].mean().item())
        out_list_forward.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    epe_all_10 = np.array(epe_list_10)
    epe_10 = np.mean(epe_all_10)
    epe_all_10_40 = np.array(epe_list_10_40)
    epe_10_40 = np.mean(epe_all_10_40)
    epe_all_40 = np.array(epe_list_40)
    epe_40 = np.mean(epe_all_40)

    print("Validation KITTI: %f, %f, EPE(s0-10): %f, EPE(s10-40): %f, EPE(s40+): %f" % (epe, f1, epe_10, epe_10_40, epe_40))

    epe_list_forward = np.array(epe_list_forward)
    out_list_forward = np.concatenate(out_list_forward)

    epe_forward = np.mean(epe_list_forward)
    f1_forward = 100 * np.mean(out_list_forward)

    epe_all_forward_10 = np.array(epe_list_forward_10)
    epe_forward_10 = np.mean(epe_all_forward_10)
    epe_all_forward_10_40 = np.array(epe_list_forward_10_40)
    epe_forward_10_40 = np.mean(epe_all_forward_10_40)
    epe_all_forward_40 = np.array(epe_list_forward_40)
    epe_forward_40 = np.mean(epe_all_forward_40)

    print("Validation KITTI of forward warped flow: %f, %f, EPE(s0-10): %f, EPE(s10-40): %f, EPE(s40+): %f" %
          (epe_forward, f1_forward, epe_forward_10, epe_forward_10_40, epe_forward_40))
    return {'kitti-epe': epe, 'kitti-f1': f1}
