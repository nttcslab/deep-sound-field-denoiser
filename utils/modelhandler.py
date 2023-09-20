import torch
import torch.nn as nn

from models.DnCNN import DnCNN
from models.LRDUNet import LRDUNet
from models.NAFNet import NAFNet


def createmodel(model_name, lr=0.001):
    """initialize model, lossfun, optimizer

    Args:
        model_name: Specify model {"DnCNN", "LRDUNet", "NAFNet"}
        lr: learning rate
    Raises:
        ValueError: Invalid model name

    Returns:
        _type_: net, lossfun, optimizer
    """
    if model_name == "DnCNN":
        net = DnCNN()
        lossfun = nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif model_name == "LRDUNet":
        net = LRDUNet()
        lossfun = nn.L1Loss(reduction="mean")
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif model_name == "NAFNet":
        img_channel = 2
        width = 32
        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]
        net = NAFNet(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blks,
            dec_blk_nums=dec_blks,
        )
        lossfun = nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        raise ValueError("Invalid model name")

    return net, lossfun, optimizer


def loadtrainedmodel(model_name, weights_file):
    """Load trained network from  name and weights

    Args:
        model_name (_type_): Model name
        weights_file (_type_): Loading weight file (.pth)

    Returns:
        _type_: trained network
    """
    net = createmodel(model_name)[0]
    net.load_state_dict(torch.load(weights_file))
    return net
