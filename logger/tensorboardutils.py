from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    def __init__(self, log_dir, logger, enabled):
        log_dir = str(log_dir)
        self.writer = SummaryWriter(log_dir)
