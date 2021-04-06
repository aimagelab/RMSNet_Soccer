from data.SoccerNetv2_dataset import SoccerNet
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import shutil
import argparse
from models.resnet import resnet50
from data.load_data_utils import LABELS, REVERSE_LABELS, FPS
from utils import nms, standard_nms
import math
from SoccerNet.Evaluation.ActionSpotting import evaluate
import sys
import json
import os
import errno

np.random.seed(123)
torch.manual_seed(123)
random.seed(123)

parser = argparse.ArgumentParser(description='SoccerNet training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lr', '--learning-rate', default=1.6, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('-t', '--frames_per_clip', default=41, type=int,
                    help='Duration (in frames) of each clip')
parser.add_argument('-hw', '--resize_to', nargs='+', default=[224,398], type=int,
                    help='frames spatial shape')
parser.add_argument('-l', '--lam', default=10, type=int,
                    help='MSE loss weight')
parser.add_argument('-c', '--num_classes', default=18, type=int,
                    help='Number of classes')
parser.add_argument('--nms_mode', default="standard", type=str,
                    help='"new" or "standard"')
parser.add_argument('--nms', default=2000, type=int,
                    help='ms for NMS (remove duplicate predictions with distance < nms ms)')
parser.add_argument('--test_overlap', default=0, type=float,
                    help='percentage in [0,1] of overlap between consecutive clips during inference')
parser.add_argument('--class_samples_per_epoch', default=1000, type=int,
                    help='number of random samples for each class in a training epoch')

parser.add_argument('--frames_path', default='./Frames_HQ', type=str,
                    help='path to Frames dir')
parser.add_argument('--labels_path', default='./annotations_v2', type=str,
                    help='path to Labels annotations')
parser.add_argument('--listgame_path', default='./Labels', type=str,
                    help='path to dir where splits are stored')
parser.add_argument('--out_dir', default='.', type=str,
                    help='path to dir where to store checkpoints')

parser.add_argument("--mixed_precision", action="store_true")
parser.add_argument("--opt_level", default='O0', type=str,
                    help='Opt level for apex mixed precision')

parser.add_argument("--testing_split", default='val', type=str,
                    help='Which split to be used for testing')
parser.add_argument("--training_split", default='train', type=str,
                    help='Which split to be used for training: specify "train+val" to train on both training and validation sets')


def main():
    args = parser.parse_args()
    main_worker(0, args)

def main_worker(gpu, args):
    args.gpu = gpu
    print(args)
    try:
        os.makedirs(args.out_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if args.mixed_precision:
        from apex import amp

    print("=> creating model")
    model = resnet50(pretrained=True).cuda()
    num_ftrs = model.fc.in_features
    model.temporal_conv1 = torch.nn.Conv1d(in_channels=num_ftrs, out_channels=512, kernel_size=9, stride=1, padding=4).cuda()
    model.temporal_conv2 = torch.nn.Conv1d(in_channels=512, out_channels=256, kernel_size=9, stride=1, padding=4).cuda()
    model.fc2 = torch.nn.Linear(256, 128).cuda()
    model.fc_class = torch.nn.Linear(128, args.num_classes).cuda()
    model.fc_t_shift = torch.nn.Linear(128, 1).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print("=> Initializing datasets...")

    dataset_val = SoccerNet(frames_per_clip=args.frames_per_clip, resize_to=tuple(args.resize_to), split=args.testing_split, frames_path=args.frames_path, labels_path=args.labels_path, listgame_path=args.listgame_path, class_samples_per_epoch=args.class_samples_per_epoch, test_overlap=args.test_overlap)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    best_map = 0

    if not args.evaluate:
        dataset_train = SoccerNet(frames_per_clip=args.frames_per_clip, resize_to=tuple(args.resize_to), split=args.training_split, frames_path=args.frames_path, labels_path=args.labels_path, listgame_path=args.listgame_path, class_samples_per_epoch=args.class_samples_per_epoch, test_overlap=args.test_overlap)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        if args.mixed_precision:
            model, optim = amp.initialize(model, optim, opt_level=args.opt_level)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, cosine_lr(args.lr, it_max=args.epochs * len(dataloader_train), warmup_iterations=len(dataloader_train)))

    model = torch.nn.DataParallel(model)

    if args.resume != '':
        #ToDo: add amp.load_state_dict()
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint["state_dict"])
        best_map = checkpoint["best_map"]
        optim.load_state_dict(checkpoint["optimizer"])
        if args.mixed_precision:
            amp.load_state_dict(checkpoint['amp'])
        if args.start_epoch == 0:
            args.start_epoch = checkpoint["epoch"]

    if args.evaluate:
        print("=> Validation begins...")
        a_mAP = validate(dataloader_val, model, criterion, args.start_epoch, args)
        sys.exit()

    best_epoch = 0
    for e in range(args.start_epoch, args.epochs):
        if e != 0:
            dataloader_train.dataset.update_background_samples()
        print("=> Training begins...")
        train(dataloader_train, model, criterion, optim, scheduler, e, args)

        print("=> Validation begins...")
        map = validate(dataloader_val, model, criterion, e, args)

        is_best = map > best_map
        if is_best:
            best_epoch = e
        best_map = max(best_map, map)

        print("Best mAP so far: " + str(best_map) + " at epoch: " + str(best_epoch))
        #ToDo: add amp.state_dict()
        if args.mixed_precision:
            save_checkpoint({
                'epoch': e + 1,
                'state_dict': model.state_dict(),
                'best_map': best_map,
                'optimizer': optim.state_dict(),
                'amp': amp.state_dict()
            }, is_best, args.out_dir)
        else:
            save_checkpoint({
                'epoch': e + 1,
                'state_dict': model.state_dict(),
                'best_map': best_map,
                'optimizer': optim.state_dict(),
            }, is_best, args.out_dir)


def train(dataloader_train, model, criterion, optim, scheduler, epoch, args):
    model.train()
    for m in model.modules(): #freeze BatchNorm layers
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    losses = AverageMeter('Loss', ':.4e')
    class_loss = AverageMeter('XE Loss', ':.4e')
    regr_loss = AverageMeter('MSE Loss', ':.4e')
    progress = ProgressMeter("Training Epoch: [{}]".format(epoch), len(dataloader_train), losses, class_loss, regr_loss)

    it_counter = epoch * len(dataloader_train)
    for it, (video, label, rel_offset, match, half, start_frame) in enumerate(dataloader_train):
        it_counter += 1

        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            rel_offset = rel_offset.cuda(args.gpu, non_blocking=True)

        out, pred_rel_offset = model(video)
        pred_rel_offset = pred_rel_offset.squeeze(1)

        non_background_indexes_gt = (label != LABELS["background"])

        time_shift_loss = torch.nn.functional.mse_loss(pred_rel_offset[non_background_indexes_gt], rel_offset[non_background_indexes_gt].float()) #let's compute the time-shift loss only for not background events

        loss = criterion(out, label) if math.isnan(time_shift_loss) else criterion(out, label) + args.lam * time_shift_loss

        losses.update(loss.item(), video.shape[0])
        class_loss.update(criterion(out, label), video.shape[0])
        regr_loss.update(0 if math.isnan(time_shift_loss) else time_shift_loss.item(), video.shape[0])

        optim.zero_grad()
        if args.mixed_precision:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optim.step()

        scheduler.step(it_counter)

        if it % 10 == 0:
            progress.printt(it)


def validate(dataloader_val, model, criterion, epoch, args):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    class_loss = AverageMeter('XE Loss', ':.4e')
    regr_loss = AverageMeter('MSE Loss', ':.4e')
    jh = JsonHandler(args.out_dir)

    progress = ProgressMeter("Validation Epoch: [{}]".format(epoch), len(dataloader_val), losses, class_loss, regr_loss)

    with torch.no_grad():
        for it, (video, label, rel_offset, match, half, start_frame) in enumerate(dataloader_val):
            if args.gpu is not None:
                video = video.cuda(args.gpu, non_blocking=True)
                label = label.cuda(args.gpu, non_blocking=True)
                rel_offset = rel_offset.cuda(args.gpu, non_blocking=True)
                start_frame = start_frame.cuda(args.gpu, non_blocking=True)

            out, pred_rel_offset = model(video)

            pred_rel_offset = pred_rel_offset.squeeze(1)
            score, cl = torch.max(torch.nn.functional.softmax(out, dim=1), dim=1)

            non_background_indexes_predicted = (cl != LABELS["background"])
            score = score[non_background_indexes_predicted]
            half = [hm for i, hm in enumerate(half) if non_background_indexes_predicted[i]]
            start_frame = start_frame[non_background_indexes_predicted]
            cl = cl[non_background_indexes_predicted]
            match = [m for i, m in enumerate(match) if non_background_indexes_predicted[i]]

            if len(score) != 0:
                jh.update_preds(match, half, cl, start_frame, score, pred_rel_offset[non_background_indexes_predicted], args.frames_per_clip)

            non_background_indexes_gt = (label != LABELS["background"])
            time_shift_loss = torch.nn.functional.mse_loss(pred_rel_offset[non_background_indexes_gt], rel_offset[non_background_indexes_gt].float())
            loss = criterion(out, label) if math.isnan(time_shift_loss) else criterion(out, label) + args.lam * time_shift_loss

            losses.update(loss.item(), video.shape[0])
            class_loss.update(criterion(out, label), video.shape[0])
            regr_loss.update(0 if math.isnan(time_shift_loss) else time_shift_loss.item(), video.shape[0])

            if it % 10 == 0:
                progress.printt(it)

        predictions_path = jh.save_json(epoch, args.testing_split, args.nms_mode, args.nms)
        results = evaluate(SoccerNet_path=args.labels_path, Predictions_path=predictions_path, split=(args.testing_split if args.testing_split!="val" else "valid"))
        print("Average mAP: ", results["a_mAP"])
        print("Average mAP per class: ", results["a_mAP_per_class"])
        print("Average mAP visible: ", results["a_mAP_visible"])
        print("Average mAP visible per class: ", results["a_mAP_per_class_visible"])
        print("Average mAP unshown: ", results["a_mAP_unshown"])
        print("Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])

    return float(results["a_mAP"])


def save_checkpoint(state, is_best, out_dir='.'):
    filename = os.path.join(out_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(out_dir, 'model_best.pth.tar'))


def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint


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
    def __init__(self, prefix, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def printt(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def cosine_lr(base_lr, it_max, warmup_iterations):
    def fn(it):
        if it < warmup_iterations:
            return it / warmup_iterations
        return .5 * (np.cos(it / it_max * np.pi) + 1)
    return fn


class JsonHandler(object):
    def __init__(self, out_dir):
        self.preds = {}
        self.out_dir = out_dir

    def update_preds(self, match, half,  classes, start_frame, scores, t_shift, frames_per_clip):
        for m, h, c, f, s, t in zip(match, half, classes, start_frame, scores, t_shift):
            spot = f+t*frames_per_clip
            if m not in self.preds.keys():
                self.preds[m] = {"UrlLocal":m, "predictions":[]}
            self.preds[m]["predictions"].append({"gameTime":str(h.item())+" - " + str(int(spot/120)).zfill(2) + ":" + str(int((spot%120)/2)).zfill(2),
                                                 "label":REVERSE_LABELS[int(c)],
                                                 "position":str(int(spot/2*1000)),
                                                 "half":str(h.item()),
                                                 "confidence":str(s.item())})
    def reset(self):
        self.preds = {}

    def save_json(self, epoch, test_split, nms_mode, nms_thresh):
        predictions_path = os.path.join(self.out_dir, test_split, str(epoch))
        try:
            os.makedirs(predictions_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if nms_mode=="new":
            preds_after_nms = nms(self.preds, nms_thresh)
        elif nms_mode=="standard":
            preds_after_nms = standard_nms(self.preds, nms_thresh)
        else:
            print("Error: invalid NMS mode specified, must be 'standard' or 'new'")
            sys.exit()

        print("Saving prediction json at epoch: " + str(epoch) + "...")
        for f, pred in preds_after_nms.items():
            try:
                os.makedirs(os.path.join(predictions_path, f))
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            with open(os.path.join(predictions_path, f, "results_spotting.json"), "w") as outfile:
                json.dump(pred, outfile)
        return predictions_path


if __name__ == '__main__':
    main()
