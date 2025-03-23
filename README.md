# MemFlow
### [Website](https://dqiaole.github.io/MemFlow/) | [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Dong_MemFlow_Optical_Flow_Estimation_and_Prediction_with_Memory_CVPR_2024_paper.pdf) | [Supplementary](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Dong_MemFlow_Optical_Flow_CVPR_2024_supplemental.pdf)
> [**MemFlow: Optical Flow Estimation and Prediction with Memory**](https://arxiv.org/abs/2404.04808)            
> Qiaole Dong, Yanwei Fu        
> **CVPR 2024**

![](imgs/overview.png)

## Requirements

```Shell
conda create --name memflow python=3.8
conda activate memflow
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install yacs loguru einops timm==0.4.12 imageio matplotlib tensorboard scipy opencv-python h5py tqdm
```

For faster training or inference, you should further install the [FlashAttention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).

This [FlashAttention wheel](https://github.com/Dao-AILab/flash-attention/releases/download/v2.3.2/flash_attn-2.3.2+cu116torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl) is compatible with our CUDA version. Refer to [issue](https://github.com/DQiaole/MemFlow/issues/14#issue-2938768407). 

## Models
We provide pretrained [models](https://github.com/DQiaole/MemFlow/releases/tag/v1.0.0). The default path of the models for evaluation is:
```Shell
├── ckpts
    ├── MemFlowNet_things.pth
    ├── MemFlowNet_sintel.pth
    ├── MemFlowNet_kitti.pth
    ├── MemFlowNet_spring.pth
    ├── MemFlowNet_T_things.pth
    ├── MemFlowNet_T_things_kitti.pth
    ├── MemFlowNet_T_sintel.pth
    ├── MemFlowNet_T_kitti.pth
    ├── MemFlowNet_P_things.pth
    ├── MemFlowNet_P_sintel.pth
```

## Demos
Download models and put them in the `ckpts` folder. Run the following command:
```shell
python -u inference.py --name MemFlowNet --stage sintel --restore_ckpt ckpts/MemFlowNet_sintel.pth --seq_dir demo_input_images --vis_dir demo_flow_vis
```
Note: you can change the `_CN.val_decoder_depth` of `configs/sintel_memflownet.py` from `15` to smaller numbers for better speed and performance trade-off as in Fig. 1.

## Required Data
To evaluate/train MatchFlow, you will need to download the required datasets.
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)
* [Spring](https://spring-benchmark.org/)


By default our codes will search for the datasets in these locations. You can create symbolic links to wherever 
the datasets were downloaded in the `datasets` folder

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
    ├── spring
        ├── test
        ├── training
        ├── flow_subsampling
```

## Evaluation
Please download the models to `ckpts` folder. Then you can evaluate the provided model using following script:
```Shell
bash evaluate.sh
```

## Training
We used the following training schedule in our paper (2 A100/A6000 GPUs). Training logs will be written to the `logs` which can be 
visualized using tensorboard.
```Shell
bash train.sh
```

## Reference
If you found our paper helpful, please consider citing:
```bibtex
@inproceedings{dong2024memflow,
  title={MemFlow: Optical Flow Estimation and Prediction with Memory},
  author={Dong, Qiaole and Fu, Yanwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Acknowledgement

Thanks to previous open-sourced repo: 
* [SKFlow](https://github.com/littlespray/SKFlow)
* [VideoFlow](https://github.com/XiaoyuShi97/VideoFlow)