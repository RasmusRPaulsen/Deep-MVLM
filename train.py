import argparse
import collections
import math

import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import matplotlib.pyplot as plt
import numpy as np
import random


# Helper function to show a batch
# from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
def show_batch(sample_batched, config):
    """Show image with landmarks for a batch of samples."""
    images_batch, heat_map_batch = sample_batched['image'], sample_batched['heat_map_stack']

    heat_map_batch = heat_map_batch.numpy()
    im_size = images_batch.size(2)
    hm_size = heat_map_batch.shape[2]

    channels = config['data_loader']['args']['image_channels']
    # Super hacky way to convert gray to RGB
    # show first image in batch
    i = np.zeros((im_size, im_size, 3))
    if channels == "geometry" or channels == "depth":
        i[:, :, 0] = images_batch[0][:, :, 0]
        i[:, :, 1] = images_batch[0][:, :, 0]
        i[:, :, 2] = images_batch[0][:, :, 0]
    elif channels == "RGB":
        i[:, :, :] = images_batch[0][:, :, :]

    # Generate combined heatmap image in red channel.
    # This must be possible to do smarter - Alas! My Python skillz are lacking
    hm = np.zeros((hm_size, hm_size, 3))
    n_lm = heat_map_batch.shape[4]
    for lm in range(n_lm):
        r = random.random()  # generate random colour placed on the unit sphere in RGB space
        g = random.random()
        b = random.random()
        v_len = math.sqrt(r * r + g * g + b * b)
        r = r / v_len
        g = g / v_len
        b = b / v_len
        hm[:, :, 0] = hm[:, :, 0] + heat_map_batch[0, 0, :, :, lm] * r
        hm[:, :, 1] = hm[:, :, 1] + heat_map_batch[0, 0, :, :, lm] * g
        hm[:, :, 2] = hm[:, :, 2] + heat_map_batch[0, 0, :, :, lm] * b
        # for x in range(hm_size):
        #   for y in range(hm_size):
        #       hm[x, y, 0] = hm[x, y, 0] + heat_map_batch[0, 0, x, y, lm] * r
        #       hm[x, y, 1] = hm[x, y, 1] + heat_map_batch[0, 0, x, y, lm] * g
        #       hm[x, y, 2] = hm[x, y, 2] + heat_map_batch[0, 0, x, y, lm] * b

    plt.figure()
    plt.imshow(i)
    plt.figure()
    plt.imshow(hm)
    plt.axis('off')
    plt.ioff()
    plt.show()


def test_dataloader(config):
    # logger = config.get_logger('train')
    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)

    for batch_idx, sample_batched in enumerate(data_loader):
        print('Batch id: ', batch_idx)
        show_batch(sample_batched, config)
        break


def test_model_mvlm(config):
    logger = config.get_logger('train')

    model = config.initialize('arch', module_arch)
    logger.info(model)


def get_cuda_info():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    print('Selected cuda device: ', torch.cuda.current_device())

    print('Number of GPUs available: ', torch.cuda.device_count())

    # Additional Info when using cuda
    if device.type == 'cuda':
        print('Cuda device name: ', torch.cuda.get_device_name(0))
        print('Cuda capabilities: ', torch.cuda.get_device_capability(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
        print('Max allocated:   ', round(torch.cuda.max_memory_allocated(0) / 1024 ** 3, 1), 'GB')


def main(config):
    # logger = config.get_logger('train')

    # setup data_loader instances
    print('Initialising data loader')
    data_loader = config.initialize('data_loader', module_data)
    print('Initialising validation data')
    valid_data_loader = data_loader.split_validation()

    print('Initialising model')
    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    # logger.info(model)

    print('Initialising loss')
    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    print('Initialising optimizer')
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    print('Initialising scheduler')
    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    print('Initialising trainer')
    trainer = Trainer(model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    print('starting to train')
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    global_config = ConfigParser(args, options)
    main(global_config)
    # test_dataloader(global_config)
    # test_model_mvlm(config)
    # get_cuda_info(config)
