from inference.memory_manager_skflow import MemoryManager
from core.Networks.MemFlowNet.corr import CorrBlock
import torch


class InferenceCore:
    def __init__(self, network, config):
        self.config = config
        self.model = network
        self.mem_every = config['mem_every']
        self.enable_long_term = config['enable_long_term']

        self.clear_memory()

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = -self.mem_every
        self.memory = MemoryManager(config=self.config)

    def step(self, images, end=False, add_pe=False, flow_init=None):
        # image: 1*2*3*H*W
        self.curr_ti += 1

        is_mem_frame = (self.curr_ti - self.last_mem_ti >= self.mem_every) and (not end)

        # B, C, H, W
        query, key, net, inp = self.model.encode_context(images[:, 0, ...])
        # B, T, C, H, W
        coords0, coords1, fmaps = self.model.encode_features(images, flow_init=flow_init)

        # predict flow
        corr_fn = CorrBlock(fmaps[:, 0, ...], fmaps[:, 1, ...],
                            num_levels=4, radius=4)

        for itr in range(self.config.val_decoder_depth):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            motion_features, current_value = self.model.update_block.get_motion_and_value(flow, corr)
            # get global motion
            memory_readout = self.memory.match_memory(query, key, current_value, scale=self.model.att.scale)
            motion_features_global = motion_features + self.model.update_block.aggregator.gamma * memory_readout
            net, up_mask, delta_flow = self.model.update_block(net, inp, motion_features, motion_features_global)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
        # upsample predictions
        flow_up = self.model.upsample_flow(coords1 - coords0, up_mask)

        # save as memory if needed
        if is_mem_frame:
            self.memory.add_memory(key, current_value)
            self.last_mem_ti = self.curr_ti

        return coords1 - coords0, flow_up
