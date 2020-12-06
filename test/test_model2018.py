from modelling.meta_arch.bmshl2018 import Compressor2018
import sys
# sys.path.append("....")
import torch
def test_forward():
    class CFG:
        pass
    cfg = CFG()
    
    cfg.DATA = CFG()
    cfg.DATA.IN_CHANNELS = 3
    
    cfg.MODEL = CFG()
    cfg.MODEL.STRIDES = [2,2,2,2]
    cfg.MODEL.CONV_KERNEL = 5
    cfg.MODEL.INTER_CHANNELS = 192
    cfg.MODEL.LATENT_CHANNELS = 192
    
    cfg.MODEL.HYPER_PRIOR = CFG()
    cfg.MODEL.HYPER_PRIOR.STRIDES = [1,2,2]
    cfg.MODEL.HYPER_PRIOR.KERNELS = [3,5,5]
    
    cfg.MODEL.ENTROPY_MODEL = CFG()
    cfg.MODEL.ENTROPY_MODEL.DIMS = [3,3,3]
    cfg.MODEL.ENTROPY_MODEL.INIT_SCALE = 10.
    cfg.MODEL.ENTROPY_MODEL.BIN = 1.
    cfg.MODEL.ENTROPY_MODEL.PROB_EPS = 1e-10
    cfg.MODEL.ENTROPY_MODEL.CONDITIONAL_MODEL = "LaplacianConditionalModel"

    cfg.MODEL.LOSS = CFG()
    cfg.MODEL.LOSS.DISTORTION_LOSS_WEIGHT = 1.
    model = Compressor2018(cfg)
    print(model)
    model.train()
    print("done init")
    x = torch.rand(2,3,256,256)
    out = model(x)
    for k, value in out.items():
        if len(value.size()):
            print(k, value.shape)
        else:
            print(k, value)

if __name__ == "__main__":
    test_forward()