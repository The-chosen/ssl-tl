from model import ResNet
from utils import autoaugment as auto

import math
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import os
from torchvision import transforms
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.NIHloader import NIHDataset
from torchsampler import ImbalancedDatasetSampler

MODEL_DICT = {
    'resnet50': ResNet.resnet50,
    'resnet18': ResNet.resnet18,
}


def train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args, LOSS_FUNC, device):
    model.train()
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    for index, (images, target) in enumerate(train_loader):
        images, target = images.to(device), target.to(device)
        output = model(images)
        optimizer.zero_grad()
        loss = LOSS_FUNC(output, target)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images[0].size(0))
        acc1 = accuracy(output, target, topk=(1,))
        top1.update(acc1[0], images[0].size(0))

        if (index + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d]\tIter [%d/%d]\tAvg Loss: %.4f\tLoss: %.4f\tAvg Acc1: %.4f\tAcc1: %.4f'
                       % (
                           epoch + 1, args.epoch, index + 1, len(train_loader), losses.avg, loss.item(), top1.avg,
                           acc1[0]))
    return losses.avg


def test(model, test_loader, nb_classes, LOSS_FUNC, device):
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    model.eval()
    y_scores = []
    y_true = []
    with torch.no_grad():
        for index, (images, target) in enumerate(test_loader):
            images, target = images.to(device), target.to(device)
            output = model(images)
            _, preds = torch.max(output, 1)
            loss = LOSS_FUNC(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images[0].size(0))
            top1.update(acc1[0], images[0].size(0))
            top5.update(acc5[0], images[0].size(0))
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            y_score = F.softmax(output, dim=1)
            y_scores.append(y_score.cpu().numpy())
            y_true.append(target.long().cpu().numpy())

    y_scores = np.concatenate(y_scores, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    aucs = auc(y_scores, y_true, nb_classes)
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Val Loss {losses.avg:.3f}'
          .format(top1=top1, top5=top5, losses=losses))
    return top1.avg, top5.avg, confusion_matrix, losses.avg, aucs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def auc(y_scores, y_true, nb_class):
    '''Return a list of AUC for each class'''
    y_true = np.eye(nb_class)[y_true]
    aucs = []
    for c in range(nb_class):
        AUC = roc_auc_score(y_true[:, c], y_scores[:, c])
        aucs.append(AUC)
    return aucs


def main():
    parser = argparse.ArgumentParser(description='Image Classification.')
    parser.add_argument('--model-name', type=str, default='resnet50')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/nih_resnet18/',
                        help='Path to save checkpoint, only the model with highest top1 acc will be saved,'
                             'And the records will also be writen in the folder')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='Maximum training epoch')
    parser.add_argument('--start-epoch', type=int, default=0, help='Start training epoch')

    parser.add_argument('--root-dir', type=str, default='../../../../datasets/nih/images/',
                        help='path to the image folder')
    parser.add_argument('--train_ann_file', default='../../../../datasets/nih/NIH_train.csv', type=str,
                        help='path to csvfile')
    parser.add_argument('--val_ann_file', default='../../../../datasets/nih/NIH_val.csv', type=str,
                        help='path to csvfile')
    # parser.add_argument('--test-dir', type=str, default='xxx/test',
    #                     help='path to the train folder, each class has a single folder')
    parser.add_argument('--cos', type=bool, default=False, help='Use cos learning rate sheduler')
    parser.add_argument('--schedule', default=[50, 100, 150], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')

    parser.add_argument('--pretrained', type=str, default='None',
                        help='Load which pretrained model, '
                             'None : Do not load any weight, random initialize'
                             'Imagenet : official Imagenet pretrained model,'
                             'MoCo : Transfer model from Moco, path in $transfer-resume$'
                             'Transfer : Transfer model from Supervised pretrained, path in $transfer-resume$'
                             'Resume : Load checkpoint for corrupted training process, path in $resume$')
    parser.add_argument('--transfer-resume', type=str, default='',
                        help='Path to load transfering pretrained model')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to resume a checkpoint')
    parser.add_argument('--num-class', type=int, default=15, help='Number of class for the classification')
    parser.add_argument('--PRINT-INTERVAL', type=int, default=3, help='Number of batch to print the loss')
    parser.add_argument('--save-epoch', default=10, type=int)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device))
    # Create checkpoint file

    if os.path.exists(args.checkpoint_path) == False:
        os.makedirs(args.checkpoint_path)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_trans = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]
    )
    train_trans = transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         auto.ImageNetPolicy(),
         transforms.ToTensor(),
         normalize
         ]
    )

    trainset = NIHDataset(root=args.root_dir, ann_file=args.train_ann_file, transforms=train_trans)
    categoriesid = trainset.get_categoriesid()
    valset = NIHDataset(root=args.root_dir, ann_file=args.val_ann_file, transforms=test_trans,
                        categoriesid=categoriesid)

    def call_back_label(dataset, idx):
        return dataset.get_label(idx)

    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size, shuffle=False,
                              sampler=ImbalancedDatasetSampler(trainset, callback_get_label=call_back_label),
                              num_workers=16)
    val_loader = DataLoader(valset, batch_size=args.batch_size, num_workers=16)
    # test_loader = DataLoader(testset,batch_size=args.batch_size,num_workers=8,pin_memory=True)

    print('dataset created\ttotal {}\ttrain {}\tval {}'.format(len(trainset) + len(valset), len(trainset), len(valset)))
    # Define Loss Function
    LOSS_FUNC = nn.CrossEntropyLoss().to(device)

    print(args.model_name)

    if args.pretrained == 'Imagenet':
        # ImageNet supervised pretrained model
        print('ImageNet supervised pretrained model')
        model = MODEL_DICT[args.model_name](num_classes=args.num_class, pretrained=True)
        for name, param in model.named_parameters():
            param.requires_grad = True
    elif args.pretrained == 'MoCo':
        # load weight from transfering model from moco
        print('Load weight from transfering model from moco')
        model = MODEL_DICT[args.model_name](num_classes=args.num_class, pretrained=False)
        for name, param in model.named_parameters():
            param.requires_grad = True
        if args.transfer_resume:
            if os.path.isfile(args.transfer_resume):
                print("=> loading checkpoint '{}'".format(args.transfer_resume))
                checkpoint = torch.load(args.transfer_resume, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

                print("=> loaded pre-trained model '{}'".format(args.transfer_resume))
            else:
                print("=> no checkpoint found at '{}'".format(args.transfer_resume))

        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()

    elif args.pretrained == 'Transfer':
        # load weight from transfering model from supervised pretraining

        model = MODEL_DICT[args.model_name](num_classes=args.num_class, pretrained=False)
        print('Load weight from transfering model from supervised pretraining')
        for name, param in model.named_parameters():
            param.requires_grad = True
        if args.transfer_resume:
            if os.path.isfile(args.transfer_resume):
                print("=> loading checkpoint '{}'".format(args.transfer_resume))

                checkpoint = torch.load(args.transfer_resume)
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer

                    if k.startswith("module.fc."):
                        del state_dict[k]
                    elif k.startswith('module.'):
                        # remove prefix
                        state_dict[k[len("module."):]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k]
                msg = model.load_state_dict(state_dict, strict=False)

                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.transfer_resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.transfer_resume))

        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
    else:
        # Random Initialize
        print('Random Initialize')
        model = MODEL_DICT[args.model_name](num_classes=args.num_class, pretrained=False)
        for name, param in model.named_parameters():
            param.requires_grad = True

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model = model.to(device)
    # Optimizer and learning rate scheduler

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.pretrained == 'Resume':
        # load weight from checkpoint
        print('Load weight from checkpoint {}'.format(args.resume))
        load_resume(args, model, optimizer, args.resume)

    metric = []
    for epoch in range(args.start_epoch, args.epoch):
        print('epoch: {}'.format(epoch))
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(model, train_loader, optimizer, args.PRINT_INTERVAL, epoch, args, LOSS_FUNC, device)
        acc1, acc5, confusion_matrix, val_loss, aucs = test(model, val_loader, args.num_class, LOSS_FUNC, device)
        metric.append(acc1)

        # Save train/val loss, acc1, acc5, confusion matrix(F1, recall, precision), AUCs
        record = {
            'epoch': epoch + 1,
            'train loss': train_loss,
            'val loss': val_loss,
            'acc1': acc1,
            'acc5': acc5,
#             'state_dict': model.state_dict(),
            'confusion matrix': confusion_matrix,
            'AUCs': aucs
        }
        torch.save(record, os.path.join(args.checkpoint_path, 'recordEpoch{}.pth.tar'.format(epoch)))
        # Only save the model with highest top1 acc
        if np.max(metric) == acc1:
            checkpoint = {
                'epoch': epoch + 1,
                'arch': args.model_name,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, 'best.pth.tar'))
            print("Model Saved")

        if epoch != 0 and epoch % args.save_epoch == 0:

            filename = os.path.join(args.checkpoint_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))

            checkpoint={
                'epoch': epoch + 1,
                'arch': args.model_name,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, filename)
            print("Model Saved")


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epoch))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    print("Learning rate adjust to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_resume(args, model, optimizer, load_path):
    if load_path:
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))

            checkpoint = torch.load(load_path)
            # args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))


if __name__ == '__main__':
    print("Start training")
    main()