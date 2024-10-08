import os

import matplotlib
import torch

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory:
    def __init__(self, log_dir):
        self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

    def append_loss(self, epoch, loc, **kwargs):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, [])
            getattr(self, key).append(value)
            with open(os.path.join(self.log_dir, key + ".txt"), 'a') as f:
                f.write(str(value))
                f.write("\n")
            self.writer.add_scalar(key, value, epoch)

        self.loss_plot(loc,**kwargs)

    def loss_plot(self,loc, **kwargs):
        plt.figure()

        for key, value in kwargs.items():
            losses = getattr(self, key)
            if losses and isinstance(losses[0], torch.Tensor):
                losses = [loss.cpu().numpy() for loss in losses]
            plt.plot(range(len(losses)), losses, linewidth=2, label=key)
        if loc == 'loss':

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

            plt.cla()
            plt.close("all")
        elif loc == 'auc':
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Auc')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_auc.png"))

            plt.cla()
            plt.close("all")