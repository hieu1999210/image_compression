from entropy_model import (
    CDFEstimator, 
    EntropyModel, 
    GaussianConditionalModel,
    LaplacianConditionalModel,
)
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def test_CDF_estimator_shape():
    estimator = CDFEstimator(4)
    x = torch.randn(2,4)
    assert tuple(estimator(x).size()) == (2,4)


def test_CDF_estimator_fit_norm():
    """
    test by visualization
    """
    estimator = CDFEstimator(1).cuda()
    optmizer = torch.optim.AdamW(estimator.parameters(), 0.001)
    eps = 0.002
    xs = []
    bs = 128
    # laplace = torch.distributions.laplace.Laplace(0., 1.)
    for _ in tqdm(range(10000)):
        u = 5. if torch.rand(1) > .5 else -5.
        x = torch.randn((bs,1)) + u 
        # x = laplace.sample((bs,1))
        xs.append(x.numpy())
        x = x.cuda()
        lower = estimator(x-eps)
        upper = estimator(x+eps)
        sign = -torch.sign(lower+upper).detach()
        prob = (sign*(torch.sigmoid(sign*upper)-torch.sigmoid(sign*lower)))
        total_bits = (torch.clamp(-1.0 * torch.log(prob + 1e-10), 0, 50)).sum()/bs
        total_bits.backward()
        # likelihood = estimator(x)
        # entropy = -likelihood.sigmoid().log().mean()
        # entropy.backward()
        torch.nn.utils.clip_grad_value_(estimator.parameters(), 5.)
        optmizer.step()
        optmizer.zero_grad()
    torch.save(estimator.state_dict(), "./state.pth")
    xs = np.concatenate(xs).squeeze()
    plt.hist(xs, bins=100)
    plt.show()
    with torch.no_grad():
        sample = torch.linspace(-10,10,10001).cuda().view(-1,1)
        lower = estimator(sample-eps)
        upper = estimator(sample+eps)
        sign = -torch.sign(lower+upper)
        prob = sign*(torch.sigmoid_(sign*upper) - torch.sigmoid_(sign*lower))
        plt.plot(sample.cpu().numpy().squeeze(1), prob.cpu().numpy(), color="r")
        plt.show()
        cdf = torch.sigmoid_(estimator(sample))
        plt.plot(sample.cpu().numpy().squeeze(1), cdf.cpu().numpy(), color="b")
        plt.show()


def test_CDF_estimator_fit_beta(p=.75, q=.75):
    """
    test by visualization
    """
    m = torch.distributions.beta.Beta(torch.tensor([p]), torch.tensor([q]))
    estimator = CDFEstimator(1, dims=[10,10,10,10])
    estimator.load_state_dict(torch.load("./state.pth"))
    estimator = estimator.cuda()
    optmizer = torch.optim.AdamW(estimator.parameters(), 0.001)
    eps = 0.001
    xs = []
    bs = 256
    for _ in tqdm(range(15000)):
        x = m.sample((bs,1))-0.5
        xs.append(x.numpy())
        x = x.cuda()
        lower = estimator(x-eps)
        upper = estimator(x+eps)
        sign = -torch.sign(lower+upper).detach()
        prob = (sign*(torch.sigmoid(sign*upper)-torch.sigmoid(sign*lower)))
        total_bits = (torch.clamp(-1.0 * torch.log(prob + 1e-10), 0, 50)).sum()/bs
        total_bits.backward()
        optmizer.step()
        optmizer.zero_grad()
    torch.save(estimator.state_dict(), "./state.pth")
    xs = np.concatenate(xs).squeeze()
    plt.hist(xs, bins=100)
    plt.show()
    with torch.no_grad():
        sample = torch.linspace(-.5,.5,20001).cuda().view(-1,1)
        lower = estimator(sample-eps)
        upper = estimator(sample+eps)
        sign = -torch.sign(lower+upper)
        prob = sign*(torch.sigmoid_(sign*upper) - torch.sigmoid_(sign*lower))
        plt.plot(sample.cpu().numpy().squeeze(1), prob.cpu().numpy(), color="r")
        plt.show()
        cdf = torch.sigmoid(estimator(sample))
        plt.plot(sample.cpu().numpy().squeeze(1), cdf.cpu().numpy(), color="b")
        plt.show()


def test_shape_entropy_model():
    """
    currently only test train mode
    """
    class CFG:
        pass
    cfg = CFG()
    cfg.MODEL = CFG()
    cfg.MODEL.ENTROPY_MODEL = CFG()
    cfg.MODEL.ENTROPY_MODEL.DIMS = [3,3,3]
    cfg.MODEL.ENTROPY_MODEL.SCALE = 10.
    cfg.MODEL.ENTROPY_MODEL.BIN = 1.
    entropy_model = EntropyModel(in_channels=2, cfg=cfg)
    entropy_model.train()
    x = torch.randn(2,2,1)
    quantized_x, probs, loss = entropy_model(x)
    assert quantized_x.size() == x.size()
    assert probs.size() == x.size()

def test_entropy_model_learn_normal():
    """
    currently only test train mode
    """
    xs = []
    bs = 128
    bin_ = 0.1
    class CFG:
        pass
    cfg = CFG()
    cfg.MODEL = CFG()
    cfg.MODEL.ENTROPY_MODEL = CFG()
    cfg.MODEL.ENTROPY_MODEL.DIMS = [3,3,3]
    cfg.MODEL.ENTROPY_MODEL.SCALE = 10.
    cfg.MODEL.ENTROPY_MODEL.BIN = bin_
    entropy_model = EntropyModel(in_channels=2, cfg=cfg).cuda()
    entropy_model.train()
    optmizer = torch.optim.AdamW(entropy_model.parameters(), 0.001)
    for _ in tqdm(range(1000)):
        x = torch.cat([torch.randn((bs,1)), torch.randn((bs,1))*5], dim=1)
        xs.append(x.numpy())
        x = x.cuda()
        quantized_x, _, loss = entropy_model(x)
        loss.backward()
        torch.nn.utils.clip_grad_value_(entropy_model.parameters(), 5.)
        optmizer.step()
        optmizer.zero_grad()
    
    xs = np.concatenate(xs).squeeze()
    plt.hist(xs[:,0], bins=100)
    plt.show()
    
    plt.hist(xs[:,1], bins=100)
    plt.show()
    
    with torch.no_grad():
        sample = torch.stack([torch.linspace(-10,10,10001)]*2,dim=1).cuda()
        _, prob, _ = entropy_model(sample)
        plt.plot(sample.cpu().numpy()[:,0], prob.cpu().numpy()[:,0], color="r")
        plt.show()
        plt.plot(sample.cpu().numpy()[:,1], prob.cpu().numpy()[:,1], color="g")
        plt.show()


def test_laplacian_conditional_model():
    class CFG:
        pass
    cfg = CFG()
    cfg.MODEL = CFG()
    cfg.MODEL.ENTROPY_MODEL = CFG()
    cfg.MODEL.ENTROPY_MODEL.BIN = 1.
    
    model = LaplacianConditionalModel(cfg)
    sample = torch.linspace(-10,10,10001)
    mean = torch.ones_like(sample)*3
    scale = torch.ones_like(sample)*2
    model.mean = mean
    model.scale = scale
    prob = model._prob_mass(sample)
    plt.plot(sample.cpu().numpy(), prob.cpu().numpy(), color="r")
    plt.show()


def test_gaussian_conditional_model():
    class CFG:
        pass
    cfg = CFG()
    cfg.MODEL = CFG()
    cfg.MODEL.ENTROPY_MODEL = CFG()
    cfg.MODEL.ENTROPY_MODEL.BIN = 1.
    
    model = GaussianConditionalModel(cfg)
    sample = torch.linspace(-10,10,10001)
    mean = torch.ones_like(sample)*3
    scale = torch.ones_like(sample)*2
    model.mean = mean
    model.scale = scale
    prob = model._prob_mass(sample)
    plt.plot(sample.cpu().numpy(), prob.cpu().numpy(), color="r")
    plt.show()
if __name__ == "__main__":
    # test_CDF_estimator_shape()
    # test_CDF_estimator_fit_norm()
    # test_CDF_estimator_fit_beta()
    # test_shape_entropy_model()
    # test_entropy_model_learn_normal()
    test_laplacian_conditional_model()
    test_gaussian_conditional_model()