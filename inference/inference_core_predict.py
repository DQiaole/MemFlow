from inference.memory_manager_skflow import MemoryManager
from core.Networks.MemFlowNet.corr import CorrBlock
import torch


class InferenceCore:
    def __init__(self, network, config):
        self.config = config
        self.model = network
        self.mem_every = config['mem_every']
        self.enable_long_term = config['enable_long_term']
        self.current_key = None
        self.clear_memory()

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = -self.mem_every
        self.current_key = None
        self.memory = MemoryManager(config=self.config)

    def step(self, images, add_pe=False, forward_warp_flow=None):
        # image: 1*2*3*H*W
        B, _, H, W = images.shape

        is_mem_frame = (self.curr_ti - self.last_mem_ti >= self.mem_every) and (not end)

        # B, C, H, W
        query, key, inp = self.model.encode_context(images)

        self.current_key = key
        # get global motion
        memory_readout = self.memory.match_memory(query, None, None, scale=self.model.att.scale)
        motion_features_global = self.model.motion_prompt.repeat(B, 1, H // 8, W // 8) + self.model.update_block.aggregator.gamma * memory_readout

        if 'concat_flow' in self.config[self.config.network] and self.config[self.config.network].concat_flow:
            motion_features_global = torch.cat([motion_features_global, forward_warp_flow], dim=1)

        _, up_mask, delta_flow = self.model.update_block(inp, motion_features_global)
        if 'concat_flow' in self.config[self.config.network] and self.config[self.config.network].concat_flow:
            delta_flow = delta_flow + forward_warp_flow
        # upsample predictions
        flow_up = self.model.upsample_flow(delta_flow, up_mask)

        return delta_flow, flow_up

    def set_memory(self, images, flow, end=False):
        # image: 1*2*3*H*W
        self.curr_ti += 1

        is_mem_frame = (self.curr_ti - self.last_mem_ti >= self.mem_every) and (not end)

        # save as memory if needed
        if is_mem_frame:
            # B, T, C, H, W
            coords0, coords1, fmaps = self.model.encode_features(images)

            # predict flow
            corr_fn = CorrBlock(fmaps[:, 0, ...], fmaps[:, 1, ...],
                                num_levels=4, radius=4)

            corr = corr_fn(coords1 + flow)  # index correlation volume
            _, current_value = self.model.update_block.get_motion_and_value(flow, corr)

            self.memory.add_memory(self.current_key, current_value)
            self.last_mem_ti = self.curr_ti
        return None