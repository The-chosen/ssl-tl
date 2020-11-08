# from model import Densenet, Inceptionv3, ResNet, VGG, SimpleCNN, Efficientnet, ResNeSt, Ensemble,SeResNet, Deeplabv3
from model import ResNet
from utils import autoaugment as auto

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
from torch.utils.tensorboard import SummaryWriter   

# from imbalanced_dataset_sampler.torchsampler import ImbalancedDatasetSampler
from rgrDataset import RgrDataset
from petCTDataset import PetCTDataset


MODEL_DICT = {
    'resnet18': ResNet.resnet18,
    'resnet50': ResNet.resnet50,
    'resnet101': ResNet.resnet101,
    'resnet152': ResNet.resnet152,
}

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

# Util function: load resume
def load_resume(args, model, optimizer, load_path):
    if load_path:
        # Default load best.pth.tar
        load_path = os.path.join(load_path, 'best.pth.tar')
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))
    else:
        print('[ERROR] No load path provided ...') 
    
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

        AUC = roc_auc_score(y_true[:,c], y_scores[:,c])
        aucs.append(AUC)
    return aucs

# Actual train function
def train(model, train_loader, optimizer, PRINT_INTERVAL, epoch, args,LOSS_FUNC, device):
    model.train()
    losses = AverageMeter('Loss', ':6.3f')
    for index, (images, target) in enumerate(tqdm(train_loader)):
        images, target = images.to(device), target.to(device)
        output = model(images)
        optimizer.zero_grad()
        loss = LOSS_FUNC(output, target)
        losses.update(loss.item(), images[0].size(0))
        loss.backward()
        optimizer.step()
        if (index + 1) % PRINT_INTERVAL == 0:
            tqdm.write('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                       % (epoch + 1, args.epoch, index + 1, len(train_loader), losses.avg))
    print(str(epoch) + ' epoch train end~')
    return losses.avg

# For validation and test
def test(model, test_loader,nb_classes, LOSS_FUNC, device):
    print('Enter test!')
    losses = AverageMeter('Loss', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    confusion_matrix = None
    aucs = None
    model.eval()
    y_scores = []
    y_true = []
    with torch.no_grad():
        for index, (images, target) in enumerate(test_loader):

            if index % 100 == 0:
                print("Test iteration: " + str(index))
                
            images, target = images.to(device), target.to(device)
            
            output = model(images)

            loss = LOSS_FUNC(output, target)
            
            losses.update(loss.item(), images[0].size(0))

    return top1.avg, confusion_matrix, losses.avg, aucs

def main():
    # Hyper parameters
    parser = argparse.ArgumentParser(description='PET & CT Regression train ...')
    
    # Model name
    parser.add_argument('--model-name',  type=str, default='resnet18')
    
    # Save pth for checkpoints & logs
    parser.add_argument('--checkpoint-path',type = str, default='./checkpoints/checkpoint_pet_ct/',
                        help= 'Path to save checkpoint, only the model with highest top1 acc will be saved,'
                              'And the records will also be writen in the folder')
    parser.add_argument('--logpath', type=str, default='./logs/log_pet_ct/', help='log path')
    
    # Path for data
    parser.add_argument('--train-dir', type=str, default='../../../../datasets/final_mr_ct/trainB/',
                        help='path to the train folder, each class has a single folder')
    parser.add_argument('--val-dir', type=str, default='../../../../datasets/final_mr_ct/valB/',
                        help='path to the validation folder, each class has a single folder')
    parser.add_argument('--test-dir', type=str, default='../../../../datasets/final_mr_ct/testB/',
                        help='path to the test folder, each class has a single folder')
    parser.add_argument('--auge-dir', type=str, default='../../../../datasets/final_mr_ct/pet_fake/',
                        help='path to the train folder, each class has a single folder')

    # optimizer hyper parameters
    parser.add_argument('--batch-size', type = int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--epoch',type = int ,default=100, help='Maximum training epoch')
    
    # Parameters related to pretrain
    parser.add_argument('--pretrained',type=str, default="None",
                        help='Load which pretrained model, '
                             'None : Do not load any weight, random initialize'
                             'Imagenet : official Imagenet pretrained model,'
                             'MoCo : Transfer model from Moco, path in $transfer-resume$'
                             'Transfer : Transfer model from Supervised pretrained, path in $transfer-resume$'
                             'Resume : Load checkpoint for corrupted training process, path in $resume$')
    parser.add_argument('--transfer-resume', type=str, 
                        default="../../../checkpoints/checkpoints_baseline_tl/trained_path/90.pth.tar/",
                        help='Path to load transfering pretrained model (TL)')
    parser.add_argument('--resume', type = str, default='',
                        help='Path to resume a checkpoint')
    
    # Other hyper parameters
    parser.add_argument('--num-class', type = int, default=1, help='Number of class for the classification')
    parser.add_argument('--PRINT-INTERVAL', type=int, default=20, help='Number of batch to print the loss')

    parser.add_argument('--save-model', type=int, default=10, help='frequent to save model')
    parser.add_argument('--augement', default='pure', type=str, help='choose pure or toge or expe')
    
    
    # Get all parameters via parser
    args = parser.parse_args()
    
    # Tensorboard writer
    writer = SummaryWriter(args.logpath)

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device {}".format(device))
    
    # Create checkpoint file if it doesn't exist
    if os.path.exists(args.checkpoint_path) == False:
        os.makedirs(args.checkpoint_path)
        print("[WARNING] Checkpoint path doesn't exist. New one is created!")

    # Set normalize parameters
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Train data augmentations
    train_trans = transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            auto.ImageNetPolicy(),
                            transforms.ToTensor(),
                            normalize
                        ]
                    )
    
    # Validation & test data augmentations
    val_test_trans = transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize
                        ]
                    )
    
    # Datasets (train, validation and test)
    trainset = PetCTDataset(pet_csv='./pet.csv', ct_csv='./ct.csv', 
                            pet_root_dir=args.auge_dir, ct_root_dir=args.train_dir, transform=train_trans)
    
    valset = RgrDataset('ct.csv', args.val_dir, args.augement, transform=val_test_trans)
    
    testset = RgrDataset('ct.csv', args.test_dir, args.augement, transform=val_test_trans)
 
    # Dataloaders (train, validation and test)
    train_loader = DataLoader(
                        trainset,
                        batch_size=args.batch_size,
                        num_workers=8,
                        sampler=None,
                        shuffle=True
                    )
    
    val_loader = DataLoader(valset, batch_size=args.batch_size)
    
    test_loader = DataLoader(testset,batch_size=args.batch_size)


    # Define loss function
    # LOSS_FUNC = nn.CrossEntropyLoss()
    LOSS_FUNC = nn.MSELoss()

    print("[INFO] Model Name: " + args.model_name)

    # Choose different pretrain methods
    
    # ImageNet supervised pretrained model
    if args.pretrained == 'Imagenet':
        print('ImageNet supervised pretrained model')
        model = MODEL_DICT[args.model_name](num_classes=args.num_class, pretrained=True)
    # MoCo pretrain: load weight from transfering model from moco
    elif args.pretrained == 'MoCo':
        print('Load weight from transfering model from moco')
        model = MODEL_DICT[args.model_name](num_classes=args.num_class, pretrained=False)
        if args.transfer_resume:
            if os.path.isfile(args.transfer_resume):
                print("=> loading checkpoint '{}'".format(args.transfer_resume))
                checkpoint = torch.load(args.pretrained, map_location="cpu")

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

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()

    # Transfer learning pretrain: load weight from transfering model from supervised pretraining
    elif args.pretrained == 'Transfer':
        print('Load weight from transfering model from supervised pretraining (TL) ...')
        
        # Get model without pretraining by imagenet
        model = MODEL_DICT[args.model_name](num_classes=args.num_class, pretrained=False)
        if args.transfer_resume:
            if os.path.isfile(args.transfer_resume):
                print("=> loading checkpoint '{}'".format(args.transfer_resume))
                checkpoint = torch.load(args.transfer_resume)
                msg = model.load_state_dict(checkpoint, strict=False)
                print("msg: " + str(msg))
                print("set(msg.missing_keys): " + str(set(msg.missing_keys)))
#                 assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} # No missing keys
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.transfer_resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.transfer_resume))
        else:
            print('[ERROR] No transfer model path provided ...')
        
        # Freeze all parameters except for fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
                
        # Initialize the fc layer for finetuning
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
    
    # Load weight from checkpoint
    elif args.pretrained == 'Resume':
        load_resume(args, model, optimizer, args.resume)
        
    # Random Initialize (None)
    else:
        print('[INFO] Random Initialize ...')
        model = MODEL_DICT[args.model_name](num_classes=args.num_class, pretrained=False)

    # Data parallel for multiple GPU usage
    if torch.cuda.device_count() > 1:
        print("[INFO] Let's use ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Start train
    print('...........Training..........')
    losses = [] # Losses list initialization
    for epoch in range(args.epoch):
        print("Epoch(train): ", epoch)
        
        # Train model
        train_loss = train(model, train_loader, optimizer, args.PRINT_INTERVAL, epoch, args, LOSS_FUNC, device)
        
        print("Epoch(val): ", epoch)
        
        # Validate model
        acc1, confusion_matrix, val_loss, aucs = test(model, val_loader, args.num_class, LOSS_FUNC, device)

        # Save loss of validation to losses list
        losses.append(val_loss)
        
        # Tensorboard writer
        writer.add_scalars('Loss', {
            'Train': train_loss,
            'Val': val_loss
        }, epoch)
        writer.flush() 
        
        # Save model every args.save_model times
        if epoch % args.save_model == 0:
            record = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'state_dict': model.state_dict(),
            }
            torch.save(record, os.path.join(args.checkpoint_path, 'record_epoch{}.pth.tar'.format(epoch)))

        # Save best model
        if np.min(losses) == val_loss:
            checkpoint = {
                    'epoch': epoch + 1,
                    'arch': args.model_name,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
            torch.save(checkpoint, os.path.join(args.checkpoint_path, 'best.pth.tar'))
            print("Best Model Saved")
            
        # Schedule optimizer    
        sheduler.step(epoch)
        
    # Start test
    print('...........Testing..........')
    print('Loading best checkpoint ...')
    
    # Load best checkpont
    load_resume(args, model, optimizer, args.checkpoint_path)
    
    # Test the best checkpoint
    acc1, confusion_matrix, test_loss, aucs = test(model, test_loader, args.num_class, LOSS_FUNC, device)
    
    # Record the test loss
    record = {
        'test_loss': test_loss,
    }
    torch.save(record, os.path.join(args.checkpoint_path, 'test_info.pth.tar'))
    
    print('[INFO] TEST LOSS: ' + str(test_loss))


if __name__ == '__main__':
    print("Start training")
    main()

    
# def evaluate():
#     parser = argparse.ArgumentParser(description='Image Classification.')
#     parser.add_argument('--model-name',  type=str, default='resnet50')

#     parser.add_argument('--batch-size', type = int, default=64, help='Batch size')
#     parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
#     parser.add_argument('--epoch',type = int ,default=200, help='Maximum training epoch')
# #     parser.add_argument('--train-dir', type=str, default='xxx/train',
# #                         help='path to the train folder, each class has a single folder')
# #     parser.add_argument('--val-dir', type=str, default='xxx/val',
# #                         help='path to the validation folder, each class has a single folder'
# #                         )
#     parser.add_argument('--train-dir', type=str, default='../../../../datasets/mr_ct_raw_dataset/luna_new_tl/train',
#                         help='path to the train folder, each class has a single folder')
#     parser.add_argument('--val-dir', type=str, default='../../../../datasets/mr_ct_raw_dataset/luna_new_tl/val',
#                         help='path to the validation folder, each class has a single folder'
#                         )
#     parser.add_argument('--test-dir', type=str, default='xxx/test',
#                         help='path to the train folder, each class has a single folder')
#     parser.add_argument('--resume', type=str, default='',
#                         help='Path to resume a checkpoint')
#     parser.add_argument('--num-class', type=int, default=2, help='Number of class for the classification')

#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device {}".format(device))
#     # Create checkpoint file
#     save_path = os.path.join(args.checkpoint_path, args.model_name)
#     if os.path.exists(save_path) == False:
#         os.makedirs(save_path)

#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
    
#     test_trans = transforms.Compose(
#                                  [
#                                   transforms.Resize((224,224)),
#                                   transforms.ToTensor(),
#                                   normalize
#                                  ]
#                              )

#     testset = datasets.ImageFolder(root=args.test_dir,
#                                transform=test_trans
#                                )


#     test_loader = DataLoader(testset,batch_size=args.batch_size)



#     print(args.model_name)
#     # LOSS_FUNC = LabelSmoothSoftmaxCE()
# #     LOSS_FUNC = nn.CrossEntropyLoss()
#     LOSS_FUNC = nn.MSELoss()
#     model = MODEL_DICT[args.model_name](num_classes=args.num_class)

#     if torch.cuda.device_count() > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
#         model = nn.DataParallel(model).to(device)


#     if args.resume:
#         if os.path.isfile(args.resume):
#             print("=> loading checkpoint '{}'".format(args.resume))

#             checkpoint = torch.load(args.resume)
#             model.load_state_dict(checkpoint['state_dict'])
#             print("=> loaded checkpoint '{}' (epoch {})"
#                   .format(args.resume, checkpoint['epoch']))
#         else:
#             print("=> no checkpoint found at '{}'".format(args.resumeh))

#     print('...........Testing..........')
# #     acc1, acc5, confusion_matrix, val_loss, aucs = test(model, test_loader, args.num_class, LOSS_FUNC, device)
#     acc1, confusion_matrix, val_loss, aucs = test(model, val_loader,args.num_class, LOSS_FUNC, device)
