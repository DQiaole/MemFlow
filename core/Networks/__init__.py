def build_network(cfg):
    name = cfg.network
    print(name)
    if name == 'MemFlowNet_skflow':
        from .MemFlowNet.MemFlow import MemFlowNet as network
    elif name == 'MemFlowNet_predict':
        from .MemFlowNet.MemFlow_P import MemFlowNet as network
    else:
        raise ValueError(f"Network = {name} is not a valid name!")

    return network(cfg[name])
