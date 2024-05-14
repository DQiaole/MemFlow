import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import random
from glob import glob
import os.path as osp
from torch.utils.data.distributed import DistributedSampler
from .utils import frame_utils
from .utils.augmentor_video import FlowAugmentor, SparseFlowAugmentor
from .utils.utils import forward_interpolate
import pickle


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, input_frames=5, forward_warp=False, subsample_groundtruth=False):
        self.augmentor = None
        self.sparse = sparse
        self.input_frames = input_frames
        print("[input frame number is {}]".format(self.input_frames))
        self.forward_warp = forward_warp
        print("[return forward warped flow {}]".format(self.forward_warp))
        self.subsample_groundtruth = subsample_groundtruth
        print("[whether subsample groundtruth: {}]".format(self.subsample_groundtruth))
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.has_gt_list = []

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        valids = None

        if not self.sparse:
            flows = [frame_utils.read_gen(path) for path in self.flow_list[index]]
        else:
            flows = []
            valids = []
            for idx in range(len(self.has_gt_list[index])):
                if self.has_gt_list[index][idx]:
                    flow, valid = frame_utils.readFlowKITTI(self.flow_list[index][idx])
                    flows.append(flow)
                    valids.append(valid)
                else:
                    flow, valid = frame_utils.readFlowKITTI(self.flow_list[index][idx])
                    flows.append(flow * 0.0)
                    valids.append(valid * 0.0)

        imgs = [frame_utils.read_gen(path) for path in self.image_list[index]]

        flows = [np.array(flow).astype(np.float32) for flow in flows]

        if self.subsample_groundtruth:
            flows = [flow[::2, ::2] for flow in flows]

        imgs = [np.array(img).astype(np.uint8) for img in imgs]

        # grayscale images
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]

        if self.augmentor is not None:
            if self.sparse:
                imgs, flows, valids = self.augmentor(imgs, flows, valids)
            else:
                imgs, flows = self.augmentor(imgs, flows)

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]

        if valids is None:
            valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]
            o_valids = False
        else:
            valids = [torch.from_numpy(valid).float() for valid in valids]
            o_valids = True
        if not self.forward_warp:
            return torch.stack(imgs), torch.stack(flows), torch.stack(valids)
        else:
            new_size = (flows[0].shape[1] // 8, flows[0].shape[2] // 8)
            if not o_valids:
                downsampled_flow = [F.interpolate(flow.unsqueeze(0), size=new_size, mode='bilinear', align_corners=True).squeeze(0) / 8 for flow in flows[:-1]]
                forward_warped_flow = [torch.zeros(2, new_size[0], new_size[1])] + [forward_interpolate(flow) for flow in downsampled_flow]
            else:
                forward_warped_flow = [torch.zeros(2, new_size[0], new_size[1])] * len(flows)
            return torch.stack(imgs), torch.stack(flows), torch.stack(valids), torch.stack(forward_warped_flow)

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.has_gt_list = v * self.has_gt_list
        return self

    def __len__(self):
        return len(self.image_list)


class FlowDatasetTest(data.Dataset):
    def __init__(self, return_gt=True, subsample_groundtruth=False):

        self.return_gt = return_gt
        self.init_seed = False
        self.subsample_groundtruth = subsample_groundtruth
        self.flow_list = []
        self.image_list = []
        self.extra_info_list = []

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        imgs = [frame_utils.read_gen(path) for path in self.image_list[index]]
        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        # grayscale images
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]

        if self.return_gt:
            flows = [frame_utils.read_gen(path) for path in self.flow_list[index]]
            flows = [np.array(flow).astype(np.float32) for flow in flows]
            if self.subsample_groundtruth:
                flows = [flow[::2, ::2] for flow in flows]
            flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]

            return torch.stack(imgs), torch.stack(flows), self.extra_info_list[index]
        else:
            return torch.stack(imgs), self.extra_info_list[index]

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintelTrain(FlowDatasetTest):
    def __init__(self, return_gt=True, dstype='clean'):
        super(MpiSintelTrain, self).__init__(return_gt=return_gt)

        root = 'datasets/'

        self.image_list = []
        self.flow_list = []
        self.extra_info_list = []

        with open("./flow_dataset_mf/sintel_training_" + dstype + "_png.pkl", "rb") as f:
            _image_list = pickle.load(f)
        with open("./flow_dataset_mf/sintel_training_" + dstype + "_flo.pkl", "rb") as f:
            _future_flow_list = pickle.load(f)
        with open("./flow_dataset_mf/sintel_training_scene.pkl", "rb") as f:
            extra_info_list = pickle.load(f)

        len_list = len(_image_list)

        for idx_list in range(len_list):
            image_num = 0
            flow_num = 0

            _images = _image_list[idx_list]
            _future_flows = _future_flow_list[idx_list]

            len_image = len(_images)

            for idx in range(len_image - 1):
                _images[idx] = root + _images[idx].strip()
                _future_flows[idx] = root + _future_flows[idx].strip()
                image_num += 1
                flow_num += 1
            _images[-1] = root + _images[-1].strip()
            image_num += 1

            self.image_list.append(_images)
            self.flow_list.append(_future_flows)
            self.extra_info_list.append(extra_info_list[idx_list])

            print(image_num, flow_num, extra_info_list[idx_list])

        print(self.image_list[0])
        print(self.flow_list[0])
        print(self.extra_info_list[0])


class SpringTrain(FlowDatasetTest):
    def __init__(self, return_gt=True, subsample_groundtruth=True, split='train'):
        super(SpringTrain, self).__init__(return_gt=return_gt, subsample_groundtruth=subsample_groundtruth)

        print('Note: use spring split:', split)

        root = 'datasets/spring/train'
        if not os.path.exists(root):
            raise ValueError(f"Spring train directory does not exist: {root}")

        self.image_list = []
        self.flow_list = []
        self.extra_info_list = []

        for scene in sorted(os.listdir(root)):
            if scene != '0041' and split == 'subset_val':
                continue
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(root, scene, f"frame_{cam}", '*.png')))
                len_image = len(images)
                # forward
                self.image_list.append(images)
                _future_flow_list = []
                for i in range(1, len_image):
                    _future_flow_list.append(os.path.join(root, scene, f"flow_FW_{cam}", f"flow_FW_{cam}_{i:04d}.flo5"))
                self.flow_list.append(_future_flow_list)
                self.extra_info_list.append((scene, cam, 'FW'))
                # backward
                self.image_list.append(images[::-1])
                _past_flow_list = []
                for i in range(len_image, 1, -1):
                    _past_flow_list.append(os.path.join(root, scene, f"flow_BW_{cam}", f"flow_BW_{cam}_{i:04d}.flo5"))
                self.flow_list.append(_past_flow_list)
                self.extra_info_list.append((scene, cam, 'BW'))

        print(self.image_list[:2])
        print(self.flow_list[:2])
        print(self.extra_info_list[:2])


class Spring_submission(FlowDatasetTest):
    def __init__(self, return_gt=False, root='datasets/spring'):
        super(Spring_submission, self).__init__(return_gt=return_gt)

        split = "test"
        image_root = osp.join(root, split)

        for scene in os.listdir(image_root):
            for cam in ["left", "right"]:
                image_list = sorted(glob(osp.join(image_root, scene, f"frame_{cam}", '*.png')))
                len_image = len(image_list)
                _images = image_list
                # forward
                self.image_list.append(_images)
                self.extra_info_list.append((list(range(1, len_image)), scene, 'FW', cam))
                # backward
                self.image_list.append(_images[::-1])
                self.extra_info_list.append((list(range(len_image, 1, -1)), scene, 'BW', cam))

        print("~~~~~~~~~~~~~~")
        print(self.image_list[:2])
        print(self.extra_info_list[:2])


class Spring(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5, forward_warp=False):
        super(Spring, self).__init__(aug_params=aug_params, input_frames=input_frames, sparse=False,
                                     forward_warp=forward_warp, subsample_groundtruth=True)

        root = 'datasets/spring/train'
        if not os.path.exists(root):
            raise ValueError(f"Spring train directory does not exist: {root}")

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []
        all_scenes = []

        for scene in sorted(os.listdir(root)):
            if scene == '0041':
                continue
            all_scenes.append(scene)
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(root, scene, f"frame_{cam}", '*.png')))
                len_image = len(images)
                # forward
                _future_flow_list = []
                for i in range(1, len_image):
                    _future_flow_list.append(os.path.join(root, scene, f"flow_FW_{cam}", f"flow_FW_{cam}_{i:04d}.flo5"))
                for idx_image in range(0, len_image - input_frames + 1):
                    self.image_list.append(images[idx_image:idx_image + input_frames])
                    self.flow_list.append(_future_flow_list[idx_image:idx_image + input_frames - 1])
                    self.has_gt_list.append([True] * (input_frames - 1))
                # backward
                images = images[::-1]
                _past_flow_list = []
                for i in range(len_image, 1, -1):
                    _past_flow_list.append(os.path.join(root, scene, f"flow_BW_{cam}", f"flow_BW_{cam}_{i:04d}.flo5"))
                for idx_image in range(0, len_image - input_frames + 1):
                    self.image_list.append(images[idx_image:idx_image + input_frames])
                    self.flow_list.append(_past_flow_list[idx_image:idx_image + input_frames - 1])
                    self.has_gt_list.append([True] * (input_frames - 1))

        print('forward:')
        print(self.image_list[0:3])
        print(self.flow_list[0:3])
        print(self.has_gt_list[0:3])
        print('backward:')
        print(self.image_list[-3:])
        print(self.flow_list[-3:])
        print(self.has_gt_list[-3:])
        print('trained scenes:', all_scenes)


class ThingsTEST(FlowDatasetTest):
    def __init__(self, return_gt=True, dstype='frames_cleanpass'):
        super(ThingsTEST, self).__init__(return_gt=return_gt)

        root = 'datasets/FlyingThings3D/'

        self.image_list = []
        self.flow_list = []
        self.extra_info_list = []

        len_image = 10

        for subset in ["A", "B", "C"]:

            for dir_index in range(50):

                if (subset, dir_index) in [("A", 4), ("B", 31), ("C", 18), ("C", 43)]:
                    continue

                _images = [root + dstype + "/TEST/" + subset + "/{:04}".format(
                    dir_index) + "/left/" + "{:04}.png".format(idx) for idx in range(7, 16)]
                _flows = [root + "optical_flow/TEST/" + subset + "/{:04}".format(
                    dir_index) + "/into_future/left/" + "OpticalFlowIntoFuture_{:04}_L.pfm".format(idx) for idx in
                          range(7, 15)]

                self.image_list.append(_images)
                self.flow_list.append(_flows)
                self.extra_info_list.append(["{}_{}".format(subset, dir_index)])

        print(self.image_list[0])
        print(self.flow_list[0])
        print(self.extra_info_list[0])


class MpiSintel_submission(FlowDatasetTest):
    def __init__(self, return_gt=False, root='datasets/Sintel', dstype='clean'):
        super(MpiSintel_submission, self).__init__(return_gt=return_gt)

        split = "test"
        image_root = osp.join(root, split, dstype)

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            len_image = len(image_list)
            _images = image_list
            self.image_list.append(_images)
            self.extra_info_list.append((list(range(0, len_image)), scene))

        print("~~~~~~~~~~~~~~")
        print(self.image_list[0])
        print(self.extra_info_list[0])


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5, root='datasets/FlyingThings3D/', dstype='frames_cleanpass', forward_warp=False):
        super(FlyingThings3D, self).__init__(aug_params=aug_params, input_frames=input_frames, sparse=False, forward_warp=forward_warp)

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []
        with open("./flow_dataset_mf/flyingthings_" + dstype + "_png.pkl", "rb") as f:
            _image_list = pickle.load(f)
        with open("./flow_dataset_mf/flyingthings_" + dstype + "_future_pfm.pkl", "rb") as f:
            _future_flow_list = pickle.load(f)
        with open("./flow_dataset_mf/flyingthings_" + dstype + "_past_pfm.pkl", "rb") as f:
            _past_flow_list = pickle.load(f)

        len_list = len(_image_list)
        print(len(_image_list), len(_future_flow_list), len(_past_flow_list))

        for idx_list in range(len_list):
            _images = _image_list[idx_list]
            _future_flows = _future_flow_list[idx_list]
            _past_flows = _past_flow_list[idx_list]

            len_image = len(_images)

            for idx in range(len_image):
                _images[idx] = root + _images[idx].strip()[10:]
                _future_flows[idx] = root + _future_flows[idx].strip()[10:]
                _past_flows[idx] = root + _past_flows[idx].strip()[10:]

            for idx_image in range(0, len_image - input_frames + 1):
                self.image_list.append(_images[idx_image:idx_image + input_frames])
                self.flow_list.append(_future_flows[idx_image:idx_image + input_frames - 1])
                self.has_gt_list.append([True] * (input_frames - 1))

                self.image_list.append(_images[idx_image:idx_image + input_frames][::-1])
                self.flow_list.append(_past_flows[idx_image + 1:idx_image + input_frames][::-1])
                self.has_gt_list.append([True] * (input_frames - 1))

        print(self.image_list[:2])
        print(self.flow_list[:2])
        print(self.has_gt_list[:2])


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5, dstype='clean', forward_warp=False):
        super(MpiSintel, self).__init__(aug_params=aug_params, input_frames=input_frames, sparse=False, forward_warp=forward_warp)

        root = 'datasets/'

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []

        with open("./flow_dataset_mf/sintel_training_" + dstype + "_png.pkl", "rb") as f:
            _image_list = pickle.load(f)
        with open("./flow_dataset_mf/sintel_training_" + dstype + "_flo.pkl", "rb") as f:
            _future_flow_list = pickle.load(f)

        len_list = len(_image_list)
        print(len(_image_list), len(_future_flow_list))

        for idx_list in range(len_list):
            _images = _image_list[idx_list]
            _future_flows = _future_flow_list[idx_list]

            len_image = len(_images)

            for idx in range(len_image - 1):
                _images[idx] = root + _images[idx].strip()
                _future_flows[idx] = root + _future_flows[idx].strip()
            _images[-1] = root + _images[-1].strip()

            for idx_image in range(0, len_image - input_frames + 1):
                self.image_list.append(_images[idx_image:idx_image + input_frames])
                self.flow_list.append(_future_flows[idx_image:idx_image + input_frames - 1])
                self.has_gt_list.append([True] * (input_frames - 1))
        print(self.image_list[0:3])
        print(self.flow_list[0:3])
        print(self.has_gt_list[0:3])


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5, forward_warp=False):
        super(HD1K, self).__init__(aug_params=aug_params, input_frames=input_frames, sparse=True, forward_warp=forward_warp)

        root = 'datasets/'

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []

        with open("./flow_dataset_mf/hd1k_png.pkl", "rb") as f:
            _image_list = pickle.load(f)
        with open("./flow_dataset_mf/hd1k_flo.pkl", "rb") as f:
            _future_flow_list = pickle.load(f)

        len_list = len(_image_list)
        print(len(_image_list), len(_future_flow_list))

        for idx_list in range(len_list):
            _images = _image_list[idx_list]
            _future_flows = _future_flow_list[idx_list]

            len_image = len(_images)

            for idx in range(len_image):
                _images[idx] = root + _images[idx].strip()
                _future_flows[idx] = root + _future_flows[idx].strip()

            for idx_image in range(0, len_image - input_frames + 1):
                self.image_list.append(_images[idx_image:idx_image + input_frames])
                self.flow_list.append(_future_flows[idx_image:idx_image + input_frames - 1])
                self.has_gt_list.append([True] * (input_frames - 1))
        print(self.image_list[0:3])
        print(self.flow_list[0:3])
        print(self.has_gt_list[0:3])


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5, forward_warp=False):
        super(KITTI, self).__init__(aug_params=aug_params, input_frames=input_frames, sparse=True, forward_warp=forward_warp)

        root = 'datasets/KITTI/'

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []

        for idx_list in range(200):
            for idx_image in range(0, input_frames - 1):
                self.image_list.append(
                    [(root + "KITTI-multiview/training/image_2/000{:03}_{:02}.png".format(idx_list, i - idx_image + 10)) for
                     i in range(input_frames)])
                self.flow_list.append(
                    [root + "training/flow_occ/000{:03}_10.png".format(idx_list)] * (input_frames - 1))
                self.has_gt_list.append(
                    [False] * idx_image + [True] + [False] * (input_frames - 2 - idx_image))
        print(self.image_list[0:3])
        print(self.flow_list[0:3])
        print(self.has_gt_list[0:3])


class KITTITest(FlowDataset):
    def __init__(self, aug_params=None):
        super(KITTITest, self).__init__(aug_params=aug_params, sparse=True)

        root = 'datasets/KITTI/'

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []

        for idx_list in range(200):
            self.image_list.append(
                [(root + "KITTI-multiview/training/image_2/000{:03}_{:02}.png".format(idx_list, i)) for i in range(0, 12)])  # 8
            self.flow_list.append(
                [root + "training/flow_occ/000{:03}_10.png".format(idx_list)] * 11)
            self.has_gt_list.append([False] * 10 + [True])

        print(self.image_list[0])
        print(self.flow_list[0])
        print(self.has_gt_list[0])


class KITTISubmission(FlowDatasetTest):
    def __init__(self, return_gt=False):
        super(KITTISubmission, self).__init__(return_gt=return_gt)

        root = 'datasets/KITTI/KITTI-multiview/testing/'

        self.image_list = []
        self.extra_info_list = []

        for idx_list in range(200):
            self.image_list.append([(root + "image_2/000{:03}_{:02}.png".format(idx_list, i)) for i in range(0, 12)])
            self.extra_info_list.append("000{:03}_10.png".format(idx_list))

        print("~~~~~~~~~~~~~~")
        print(self.image_list[:2])
        print(self.extra_info_list[:2])


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H', DDP=False, rank=0):
    forward_warp = 'concat_flow' in args[args.network] and args[args.network].concat_flow
    if args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, input_frames=args.input_frames, dstype='frames_cleanpass',
                                       forward_warp=forward_warp)
        final_dataset = FlyingThings3D(aug_params, input_frames=args.input_frames, dstype='frames_finalpass',
                                       forward_warp=forward_warp)
        train_dataset = clean_dataset + final_dataset
    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, input_frames=args.input_frames, dstype='frames_cleanpass',
                                       forward_warp=forward_warp)
        sintel_clean = MpiSintel(aug_params, dstype='clean', input_frames=args.input_frames,
                                       forward_warp=forward_warp)
        sintel_final = MpiSintel(aug_params, dstype='final', input_frames=args.input_frames,
                                       forward_warp=forward_warp)
        hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True},
                    input_frames=args.input_frames, forward_warp=forward_warp)
        kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True},
                      input_frames=args.input_frames, forward_warp=forward_warp)

        print("[dataset len: ]", len(things), len(sintel_clean), len(hd1k), len(kitti))

        train_dataset = 100 * sintel_clean + 100 * sintel_final + 50 * kitti + 5 * hd1k + things
    elif args.stage == 'spring_only':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        spring = Spring(aug_params, input_frames=args.input_frames, forward_warp=forward_warp)
        print("[dataset len: ]", len(spring))

        train_dataset = 10 * spring
    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        kitti = KITTI(aug_params, input_frames=args.input_frames, forward_warp=forward_warp)
        train_dataset = 100 * kitti

    print('Training with %d image pairs' % len(train_dataset))
    if DDP:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size,
                                           rank=rank, shuffle=True)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size // args.world_size,
                                       pin_memory=True, shuffle=False, num_workers=16, sampler=train_sampler)
        return train_sampler, train_loader
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

        return train_loader
