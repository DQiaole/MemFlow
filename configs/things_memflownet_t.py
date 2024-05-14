from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix = 'things_memflownet_t'
_CN.gamma = 0.8
_CN.max_flow = 400
_CN.batch_size = 8
_CN.sum_freq = 100
_CN.val_freq = 10000
_CN.image_size = [432, 960]
_CN.add_noise = False
_CN.use_smoothl1 = False
_CN.filter_epe = False
_CN.critical_params = []

_CN.network = 'MemFlowNet_skflow'

_CN.restore_steps = 0
_CN.mixed_precision = True
_CN.val_decoder_depth = 15

###############################################
# Mem

_CN.input_frames = 3
_CN.num_ref_frames = 2
_CN.train_avg_length = (432 * 960 // 64) * 3 / 2
_CN.mem_every = 1
_CN.top_k = None
_CN.enable_long_term = False
_CN.enable_long_term_count_usage = True
_CN.max_mid_term_frames = 2
_CN.min_mid_term_frames = 1
_CN.num_prototypes = 128
_CN.max_long_term_elements = 10000

################################################
################################################
_CN.MemFlowNet_skflow = CN()
_CN.MemFlowNet_skflow.pretrain = True
_CN.MemFlowNet_skflow.cnet = 'twins'
_CN.MemFlowNet_skflow.fnet = 'twins'
_CN.MemFlowNet_skflow.gma = 'GMA-SK2'
_CN.MemFlowNet_skflow.down_ratio = 8
_CN.MemFlowNet_skflow.feat_dim = 256
_CN.MemFlowNet_skflow.corr_fn = 'default'
_CN.MemFlowNet_skflow.corr_levels = 4

_CN.MemFlowNet_skflow.decoder_depth = 12
_CN.MemFlowNet_skflow.critical_params = ["cnet", "fnet", "pretrain", "corr_levels", "decoder_depth", "train_avg_length"]

_CN.MemFlowNet_skflow.train_avg_length = _CN.train_avg_length

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'
_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 2.5e-4
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 600000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
