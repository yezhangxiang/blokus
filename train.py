import argparse
import os
import sys
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from blokus_dataset import BlokusDataset
from utility import index_split
import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('train_data', metavar='DIR',
                    help='path to training dataset')
parser.add_argument('val_data', metavar='DIR',
                    help='path to validate dataset')
parser.add_argument('--channel-index', default='0:7', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', default=[30, 60], type=int, nargs='+',
                    help='Decrease learning rate')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--retrain', default='', type=str, metavar='PATH',
                    help='path to model to retrain with (default: none)')
parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--gpu_id', default='', type=str, help='id for CUDA_VISIBLE_DEVICES')


def main(args):
    best_prec1 = 0

    title = 'blokus-' + args.arch
    checkpoint_path = os.path.join(args.checkpoint, title + time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime()))

    is_use_cuda = False
    if args.gpu_id and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        is_use_cuda = True

    channel_index = index_split(args.channel_index)
    channel_num = len(channel_index)
    class_num = 20 * 20 * 91

    start_epoch = args.start_epoch

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    file_handler = logging.FileHandler(os.path.join(checkpoint_path, 'blokus.log'))
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    argument_str = 'argument is:\n'
    logger.info(sys.argv)
    for k, v in args._get_kwargs():
        argument_str += ('\t' + str(k) + ' : ' + str(v) + '\n')
    logger.info(argument_str)

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](channel_num, num_classes=class_num)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    if is_use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if args.retrain:
        logger.info('==> loading model from file')
        assert os.path.isfile(args.retrain), 'Error: no retrain file found!'
        checkpoint = torch.load(args.retrain)
        model.load_state_dict(checkpoint['state_dict'])

    if is_use_cuda:
        cudnn.benchmark = True

    logger.info('\tTotal params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Data loading code
    train_dataset = BlokusDataset(args.train_data, channel_index=channel_index)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_dataset = BlokusDataset(args.val_data, channel_index=channel_index)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    logger.info('Train data number: ' + str(len(train_dataset)))
    logger.info('Validate data number: ' + str(len(val_dataset)))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.schedule, args.gamma)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq, is_use_cuda)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, args.print_freq, is_use_cuda)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_path)


def train(train_loader, model, criterion, optimizer, epoch, args_print_freq, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logger = logging.getLogger(__name__)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, targets, inputs_all) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            targets = targets.cuda(async=True)
        inputs = inputs.float()
        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(targets)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args_print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, args_print_freq, use_cuda):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    logger = logging.getLogger(__name__)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, targets, inputs_all) in enumerate(val_loader):
        if use_cuda:
            targets = targets.cuda(async=True)
        inputs = inputs.float()
        input_var = torch.autograd.Variable(inputs, volatile=True)
        target_var = torch.autograd.Variable(targets, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args_print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    file_path = os.path.join(checkpoint, filename)

    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(checkpoint, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args_schedule, args_gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args_schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args_gamma


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
