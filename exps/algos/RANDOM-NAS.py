##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import argparse
import glob
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
import pdb

from config_utils import configure2str, dict2config, load_config
from datasets import get_datasets, get_nas_search_loaders
from log_utils import AverageMeter, convert_secs2time, time_string
from models import (get_cell_based_tiny_net, get_search_spaces,
                    get_sub_search_spaces)
from nas_201_api import NASBench201API as API
from procedures import (copy_checkpoint, get_optim_scheduler, prepare_logger4,
                        prepare_seed, save_checkpoint)
from utils import get_model_infos, obtain_accuracy


def search_func(xloader, network, criterion, scheduler, w_optimizer, epoch_str,
                print_freq, logger):
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(
    ), AverageMeter()
    network.train()
    end = time.time()
    for step, (base_inputs, base_targets, arch_inputs,
               arch_targets) in enumerate(xloader):
        scheduler.update(None, 1.0 * step / len(xloader))
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        # update the weights
        network.module.random_genotype(True)
        w_optimizer.zero_grad()
        _, logits = network(base_inputs)
        base_loss = criterion(logits, base_targets)
        base_loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), 5)
        w_optimizer.step()
        # record
        base_prec1, base_prec5 = obtain_accuracy(logits.data,
                                                 base_targets.data,
                                                 topk=(1, 5))
        base_losses.update(base_loss.item(), base_inputs.size(0))
        base_top1.update(base_prec1.item(), base_inputs.size(0))
        base_top5.update(base_prec5.item(), base_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = '*SEARCH* ' + time_string(
            ) + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
            Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(
                batch_time=batch_time, data_time=data_time)
            Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(
                loss=base_losses, top1=base_top1, top5=base_top5)
            logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
    return base_losses.avg, base_top1.avg, base_top5.avg


def valid_func(xloader, network, criterion):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(
    ), AverageMeter()
    network.eval()
    end = time.time()
    with torch.no_grad():
        for step, (arch_inputs, arch_targets) in enumerate(xloader):
            arch_targets = arch_targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction

            network.module.random_genotype(True)
            _, logits = network(arch_inputs)
            arch_loss = criterion(logits, arch_targets)
            # record
            arch_prec1, arch_prec5 = obtain_accuracy(logits.data,
                                                     arch_targets.data,
                                                     topk=(1, 5))
            arch_losses.update(arch_loss.item(), arch_inputs.size(0))
            arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
            arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_losses.avg, arch_top1.avg, arch_top5.avg


def search_find_best(xloader, network, n_samples):
    with torch.no_grad():
        network.eval()
        archs, valid_accs = [], []
        #print ('obtain the top-{:} architectures'.format(n_samples))
        loader_iter = iter(xloader)
        for i in range(n_samples):
            arch = network.module.random_genotype(True)
            try:
                inputs, targets = next(loader_iter)
            except:
                loader_iter = iter(xloader)
                inputs, targets = next(loader_iter)

            _, logits = network(inputs)
            val_top1, val_top5 = obtain_accuracy(logits.cpu().data,
                                                 targets.data,
                                                 topk=(1, 5))

            archs.append(arch)
            valid_accs.append(val_top1.item())

        best_idx = np.argmax(valid_accs)
        best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
        return best_arch, best_valid_acc


def main(xargs):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger4(args)

    train_data, valid_data, xshape, class_num = get_datasets(
        xargs.dataset, xargs.data_path, -1)
    config = load_config(xargs.config_path, {
        'class_num': class_num,
        'xshape': xshape
    }, logger)
    search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', \
                                          (config.batch_size, config.test_batch_size), xargs.workers)
    logger.log(
        '||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'
        .format(xargs.dataset, len(search_loader), len(valid_loader),
                config.batch_size))
    logger.log('||||||| {:10s} ||||||| Config={:}'.format(
        xargs.dataset, config))

    # search_space = get_search_spaces('cell', xargs.search_space_name)
    search_space = get_sub_search_spaces('cell', xargs.search_space_name)
    logger.log('search_space={}'.format(search_space))
    model_config = dict2config(
        {
            'name': 'RANDOM',
            'C': xargs.channel,
            'N': xargs.num_cells,
            'max_nodes': xargs.max_nodes,
            'num_classes': class_num,
            'space': search_space,
            'affine': False,
            'track_running_stats': bool(xargs.track_running_stats)
        }, None)
    search_model = get_cell_based_tiny_net(model_config)

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(
        search_model.parameters(), config)
    logger.log('w-optimizer : {:}'.format(w_optimizer))
    logger.log('w-scheduler : {:}'.format(w_scheduler))
    logger.log('criterion   : {:}'.format(criterion))
    if xargs.arch_nas_dataset is None: api = None
    else: api = API(xargs.arch_nas_dataset)
    logger.log('{:} create API = {:} done'.format(time_string(), api))

    last_info, model_base_path, model_best_path = logger.path(
        'info'), logger.path('model'), logger.path('best')
    network, criterion = torch.nn.DataParallel(
        search_model).cuda(), criterion.cuda()

    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log("=> loading checkpoint of the last-info '{:}' start".format(
            last_info))
        last_info = torch.load(last_info)
        start_epoch = last_info['epoch']
        checkpoint = torch.load(last_info['last_checkpoint'])
        genotypes = checkpoint['genotypes']
        valid_accuracies = checkpoint['valid_accuracies']
        search_model.load_state_dict(checkpoint['search_model'])
        w_scheduler.load_state_dict(checkpoint['w_scheduler'])
        w_optimizer.load_state_dict(checkpoint['w_optimizer'])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch."
            .format(last_info, start_epoch))
    else:
        logger.log('=> do not find the last-info file : {:}'.format(last_info))
        start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {}

    # start training
    start_time, search_time, epoch_time, total_epoch = time.time(
    ), AverageMeter(), AverageMeter(), config.epochs + config.warmup
    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = 'Time Left: {:}'.format(
            convert_secs2time(epoch_time.val * (total_epoch - epoch), True))
        epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
        logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(
            epoch_str, need_time, min(w_scheduler.get_lr())))

        # selected_arch = search_find_best(valid_loader, network, criterion, xargs.select_num)
        search_w_loss, search_w_top1, search_w_top5 = search_func(
            search_loader, network, criterion, w_scheduler, w_optimizer,
            epoch_str, xargs.print_freq, logger)
        search_time.update(time.time() - start_time)
        logger.log(
            '[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'
            .format(epoch_str, search_w_loss, search_w_top1, search_w_top5,
                    search_time.sum))
        valid_a_loss, valid_a_top1, valid_a_top5 = valid_func(
            valid_loader, network, criterion)
        logger.log(
            '[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'
            .format(epoch_str, valid_a_loss, valid_a_top1, valid_a_top5))
        cur_arch, cur_valid_acc = search_find_best(valid_loader, network,
                                                   xargs.select_num)
        logger.log('[{:}] find-the-best : {:}, accuracy@1={:.2f}%'.format(
            epoch_str, cur_arch, cur_valid_acc))
        genotypes[epoch] = cur_arch
        # check the best accuracy
        valid_accuracies[epoch] = valid_a_top1
        if valid_a_top1 > valid_accuracies['best']:
            valid_accuracies['best'] = valid_a_top1
            find_best = True
        else:
            find_best = False

        # save checkpoint
        save_path = save_checkpoint(
            {
                'epoch': epoch + 1,
                'args': deepcopy(xargs),
                'search_model': search_model.state_dict(),
                'w_optimizer': w_optimizer.state_dict(),
                'w_scheduler': w_scheduler.state_dict(),
                'genotypes': genotypes,
                'valid_accuracies': valid_accuracies
            }, model_base_path, logger)
        last_info = save_checkpoint(
            {
                'epoch': epoch + 1,
                'args': deepcopy(args),
                'last_checkpoint': save_path,
            }, logger.path('info'), logger)
        if find_best:
            logger.log(
                '<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'
                .format(epoch_str, valid_a_top1))
            pdb.set_trace()
            copy_checkpoint(model_base_path, model_best_path, logger)
        if api is not None:
            logger.log('{:}'.format(api.query_by_arch(genotypes[epoch])))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log('\n' + '-' * 200)
    logger.log('Pre-searching costs {:.1f} s'.format(search_time.sum))
    start_time = time.time()
    best_arch, best_acc = search_find_best(valid_loader, network,
                                           xargs.select_num)
    search_time.update(time.time() - start_time)
    logger.log(
        'RANDOM-NAS finds the best one : {:} with accuracy={:.2f}%, with {:.1f} s.'
        .format(best_arch, best_acc, search_time.sum))
    if api is not None: logger.log('{:}'.format(api.query_by_arch(best_arch)))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Random search for NAS.')
    parser.add_argument('--data_path', type=str, help='Path to dataset')
    parser.add_argument('--dataset',
                        type=str,
                        choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                        help='Choose between Cifar10/100 and ImageNet-16.')
    # channels and number-of-cells
    parser.add_argument('--search_space_name',
                        type=str,
                        help='The search space name.')
    parser.add_argument('--config_path',
                        type=str,
                        help='The path to the configuration.')
    parser.add_argument('--max_nodes',
                        type=int,
                        help='The maximum number of nodes.')
    parser.add_argument('--channel', type=int, help='The number of channels.')
    parser.add_argument('--num_cells',
                        type=int,
                        help='The number of cells in one stage.')
    parser.add_argument(
        '--select_num',
        type=int,
        help='The number of selected architectures to evaluate.')
    parser.add_argument(
        '--track_running_stats',
        type=int,
        choices=[0, 1],
        help='Whether use track_running_stats or not in the BN layer.')
    # log
    parser.add_argument('--workers',
                        type=int,
                        default=2,
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--save_dir',
                        type=str,
                        help='Folder to save checkpoints and log.')
    parser.add_argument(
        '--arch_nas_dataset',
        type=str,
        help='The path to load the architecture dataset (tiny-nas-benchmark).')
    parser.add_argument('--print_freq',
                        type=int,
                        help='print frequency (default: 200)')
    parser.add_argument('--rand_seed', type=int, help='manual seed')
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    main(args)
