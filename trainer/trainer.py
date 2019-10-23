import time

# import imageio
import numpy as np
import torch
# from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop
import datetime


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            if self.writer is not None:
                self.writer.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        # total_metrics = np.zeros(len(self.metrics))
        start_time = time.time()
        for batch_idx, sample_batched in enumerate(self.data_loader):
            data, target = sample_batched['image'], sample_batched['heat_map_stack']

            # Debug to check heatmap
            # lm_no = 26
            # name_hm_maxima = self.config.temp_dir / ('hm_maxima' + str(batch_idx) + '_LM_' + str(lm_no) + '.png')
            # hm_debug = target[0, 0, :, :, lm_no]
            # hm_debug = hm_debug.numpy()
            # imageio.imwrite(name_hm_maxima, hm_debug)

            # TODO: This transform should probably not be done here
            data = data.permute(0, 3, 1, 2)  # from NHWC to NCHW

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            # TODO: Not sure these permutations should be done here
            # output: from (S, B, NL, H, W) -> (B, S, NL, H, W)
            # target: from (B, S, H, W, NL) -> (B, S, Nl, H, W)  (NL is equal to number of channels (C))
            output = output.permute(1, 0, 2, 3, 4)
            target = target.permute(0, 1, 4, 2, 3)

            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            if self.writer is not None:
                self.writer.writer.add_scalar('train/loss', loss.item())
            total_loss += loss.item()

            # TODO: Compute custom metrics (Landmark distances etc)
            # total_metrics += self._eval_metrics(output, target)

            time_per_test = (time.time() - start_time) / (batch_idx + 1)
            time_left = (self.len_epoch - batch_idx) * time_per_test

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Time per batch: {:.5} Time left in epoch: {}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    time_per_test,
                    str(datetime.timedelta(seconds=time_left))))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            # 'metrics': (total_metrics / self.len_epoch).tolist()
        }
        print('Debug saving checkpoint')  # TODO only for debug purposes
        self._save_checkpoint(epoch, save_best=False)
        
        print('Doing validation')
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        n_validation = len(self.valid_data_loader)
        start_time = time.time()
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(self.valid_data_loader):
                data, target = sample_batched['image'], sample_batched['heat_map_stack']
                # TODO: This transform should probably not be done here
                data = data.permute(0, 3, 1, 2)  # from NHWC to NCHW

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                # TODO: Not sure these permutations should be done here
                # output: from (S, B, NL, H, W) -> (B, S, NL, H, W)
                # target: from (B, S, H, W, NL) -> (B, S, Nl, H, W)  (NL is equal to number of channels (C))
                output = output.permute(1, 0, 2, 3, 4)
                target = target.permute(0, 1, 4, 2, 3)

                loss = self.loss(output, target)

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # if self.writer is not None:
                #    self.writer.writer.add_scalar('validation/loss', loss.item())
                total_val_loss += loss.item()

                time_per_test = (time.time() - start_time) / (batch_idx + 1)
                time_left = (n_validation - batch_idx) * time_per_test

                if batch_idx % self.log_step == 0:
                    self.logger.debug(
                        'Validation: {}/{} Loss: {:.6f} Time per batch: {:.5} Time left in validation: {}'.format(
                            batch_idx,
                            n_validation,
                            loss,
                            time_per_test,
                            str(datetime.timedelta(seconds=time_left))))

                # total_val_metrics += self._eval_metrics(output, target)  # TODO: Add custom metrics
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')

        if self.writer is not None:
            avg_val_loss = total_val_loss / len(self.valid_data_loader)
            self.writer.writer.add_scalar('validation/loss', avg_val_loss, epoch)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
