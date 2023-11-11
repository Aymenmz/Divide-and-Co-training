import os
import sys
import time
import random
import shutil
import argparse
import warnings
import setproctitle

import torch
import torch.cuda.amp as amp
from torch import nn, distributed
from torch.backends import cudnn
from tensorboardX import SummaryWriter

import parser_params
from utils import rmsprop_tf
from utils import metric, lr_scheduler, label_smoothing, norm, prefetch, summary
from model import splitnet
from dataset import factory
from utils.thop import profile, clever_format

# global best accuracy
best_acc1 = 0

def main(args):
    # Scale learning rate based on global batch size
    if args.is_linear_lr:
        args = lr_scheduler.scale_lr_and_momentum(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Set GPU to None for CPU training
    args.gpu = None

    # If we train the model separately, all the number of loops will be one.
    # It is similar to split_factor = 1
    args.loop_factor = 1 if args.is_train_sep else args.split_factor

    # Use distributed training or not
    args.is_distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function.
        # spawn will produce the process index for the first arg of main_worker
        torch.multiprocessing.spawn(main_worker, nprocs=1, args=(args,))
    else:
        # Simply call main_worker function
        main_worker(args)

    # Clean up processes
    # torch.distributed.destroy_process_group()


def main_worker(args):
    global best_acc1

    # Set the name of the process
    setproctitle.setproctitle(args.proc_name + '_rank{}'.format(args.rank))

    # Define tensorboard summary
    val_writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'val'))

    # Define loss function (criterion) and optimizer
    if args.is_label_smoothing:
        criterion = label_smoothing.label_smoothing_CE(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss()

    # Create model
    if args.pretrained:
        model_info = "INFO:PyTorch: using pre-trained model '{}'".format(args.arch)
    else:
        model_info = "INFO:PyTorch: creating model '{}'".format(args.arch)

    print(model_info)
    model = splitnet.SplitNet(args,
                              norm_layer=norm.norm(args.norm_mode),
                              criterion=criterion)

    # Print the number of parameters in the model
    print("INFO:PyTorch: The number of parameters in the model is {}".format(metric.get_the_number_of_params(model)))

    if args.is_summary:
        summary_choice = 0
        if summary_choice == 0:
            summary.summary(model,
                            torch.rand((1, 3, args.crop_size, args.crop_size)),
                            target=torch.ones(1, dtype=torch.long))
        else:
            flops, params = profile(model,
                                    inputs=(torch.rand((1, 3, args.crop_size, args.crop_size)),
                                    torch.ones(1, dtype=torch.long),
                                    'summary'))
            print(clever_format([flops, params], "%.4f"))
        return None

    # Optimizer
    optimizer = None  # Initialize the optimizer, but it won't be used for CPU training

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("INFO:PyTorch: => loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location='cpu')  # Load on CPU

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            model.load_state_dict(checkpoint['state_dict'])
            print("INFO:PyTorch: Loading state_dict of optimizer")

            if "scaler" in checkpoint:
                print("INFO:PyTorch: Loading state_dict of AMP loss scaler")
                # Note: If you're not using GPUs, you can skip loading the scaler

            print("INFO:PyTorch: => loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("INFO:PyTorch: => no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    data_split_factor = args.loop_factor if args.is_diff_data_train else 1
    print("INFO:PyTorch: => The number of views of train data is '{}'".format(data_split_factor))

    train_loader, train_sampler = factory.get_data_loader(args.data,
                                                          split_factor=data_split_factor,
                                                          batch_size=args.batch_size,
                                                          crop_size=args.crop_size,
                                                          dataset=args.dataset,
                                                          split="train",
                                                          is_distributed=False,  # Not using distributed training
                                                          is_autoaugment=args.is_autoaugment,
                                                          randaa=args.randaa,
                                                          is_cutout=args.is_cutout,
                                                          erase_p=args.erase_p,
                                                          num_workers=args.workers)

    val_loader = factory.get_data_loader(args.data,
                                        batch_size=args.eval_batch_size,
                                        crop_size=args.crop_size,
                                        dataset=args.dataset,
                                        split="val",
                                        num_workers=args.workers)

    # Learning rate scheduler
    scheduler = lr_scheduler.lr_scheduler(mode=args.lr_mode,
                                          init_lr=args.lr,
                                          num_epochs=args.epochs,
                                          iters_per_epoch=len(train_loader),
                                          lr_milestones=args.lr_milestones,
                                          lr_step_multiplier=args.lr_step_multiplier,
                                          slow_start_epochs=args.slow_start_epochs,
                                          slow_start_lr=args.slow_start_lr,
                                          end_lr=args.end_lr,
                                          multiplier=args.lr_multiplier,
                                          decay_factor=args.decay_factor,
                                          decay_epochs=args.decay_epochs,
                                          staircase=True)

    if args.evaluate:
        validate(val_loader, model, args)
        return None

    saved_ckpt_filenames = []

    streams = None

    for epoch in range(args.start_epoch, args.epochs + 1):

        # Train for one epoch
        train(train_loader, model, epoch, args, optimizer, scheduler, streams)

        if (epoch + 1) % args.eval_per_epoch == 0:
            # Evaluate on validation set
            acc_all = validate(val_loader, model, args)

            # Remember best acc@1 and save checkpoint
            is_best = acc_all[0] > best_acc1
            best_acc1 = max(acc_all[0], best_acc1)

            # Save checkpoint
            # Summary per epoch
            val_writer.add_scalar('avg_acc1', acc_all[0], global_step=epoch)
            if args.dataset == 'imagenet':
                val_writer.add_scalar('avg_acc5', acc_all[1], global_step=epoch)

            for i in range(2, args.loop_factor + 2):
                val_writer.add_scalar('{}_acc1'.format(i - 1), acc_all[i], global_step=epoch)

            val_writer.add_scalar('learning_rate', 0, global_step=epoch)  # Placeholder for learning rate
            val_writer.add_scalar('best_acc1', best_acc1, global_step=epoch)

            # Save checkpoints
            filename = "checkpoint_{0}.pth.tar".format(epoch)
            saved_ckpt_filenames.append(filename)

            # Remove the oldest file if the number of saved ckpts is greater than args.max_ckpt_nums
            if len(saved_ckpt_filenames) > args.max_ckpt_nums:
                os.remove(os.path.join(args.model_dir, saved_ckpt_filenames.pop(0)))

            ckpt_dict = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': None,  # Not saving the optimizer for CPU training
            }

            if args.is_amp:
                ckpt_dict['scaler'] = None  # Not saving the scaler for CPU training

            metric.save_checkpoint(ckpt_dict, is_best, args.model_dir, filename=filename)

    sys.exit(0)

            
            
            
            
# def train(train_loader, model, epoch, args, optimizer=None, scheduler=None, streams=None):
#     batch_time = metric.AverageMeter('Time', ':6.3f')
#     data_time = metric.AverageMeter('Data', ':6.3f')
#     avg_ce_loss = metric.AverageMeter('ce_loss', ':.4e')
#     avg_cot_loss = metric.AverageMeter('cot_loss', ':.4e')

#     top1_all = [metric.AverageMeter(f'{i}_Acc@1', ':6.2f') for i in range(args.loop_factor)]
#     avg_top1 = metric.AverageMeter('Avg_Acc@1', ':6.2f')

#     total_iters = len(train_loader)
#     progress = metric.ProgressMeter(total_iters, batch_time, data_time, avg_ce_loss, avg_cot_loss, *top1_all, avg_top1, prefix=f"Epoch: [{epoch}]")

#     model.train()
#     end = time.time()

#     for i, (images, target) in enumerate(train_loader):
#         data_time.update(time.time() - end)

#         # compute outputs and losses
#         ensemble_output, outputs, ce_loss, cot_loss = model(images, target=target, mode='train', epoch=epoch)

#         batch_size_now = images.size(0)
#         for j in range(args.loop_factor):
#             acc1 = metric.accuracy(outputs[j], target, topk=(1,))
#             top1_all[j].update(acc1[0].item(), batch_size_now)

#         avg_acc1 = metric.accuracy(ensemble_output, target, topk=(1,))
#         avg_top1.update(avg_acc1[0].item(), batch_size_now)
#         avg_ce_loss.update(ce_loss.mean().item(), batch_size_now)
#         avg_cot_loss.update(cot_loss.mean().item(), batch_size_now)

#         total_loss = (ce_loss + cot_loss) / args.iters_to_accumulate

#         # Compute gradient and do SGD step if optimizer is provided
#         if optimizer is not None:
#             total_loss.backward()
#             if (i + 1) % args.iters_to_accumulate == 0 or (i + 1) == total_iters:
#                 optimizer.step()
#                 optimizer.zero_grad()

#         batch_time.update(time.time() - end)
#         end = time.time()

#         if (i + 1) % (args.print_freq * args.iters_to_accumulate) == 0:
#             progress.print(i)

#     # No need to empty cache since we're not using GPU



def train(train_loader, model, epoch, args, optimizer=None, scheduler=None, streams=None):
    batch_time = metric.AverageMeter('Time', ':6.3f')
    data_time = metric.AverageMeter('Data', ':6.3f')
    avg_ce_loss = metric.AverageMeter('ce_loss', ':.4e')
    avg_cot_loss = metric.AverageMeter('cot_loss', ':.4e')

    top1_all = [metric.AverageMeter(f'{i}_Acc@1', ':6.2f') for i in range(args.loop_factor)]
    avg_top1 = metric.AverageMeter('Avg_Acc@1', ':6.2f')

    total_iters = len(train_loader)
    progress = metric.ProgressMeter(total_iters, batch_time, data_time, avg_ce_loss, avg_cot_loss, *top1_all, avg_top1, prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    log_dir = "/log"
    # Create a SummaryWriter for logging
    writer = SummaryWriter(log_dir=log_dir)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # compute outputs and losses
        ensemble_output, outputs, ce_loss, cot_loss = model(images, target=target, mode='train', epoch=epoch)

        batch_size_now = images.size(0)
        for j in range(args.loop_factor):
            acc1 = metric.accuracy(outputs[j], target, topk=(1,))
            top1_all[j].update(acc1[0].item(), batch_size_now)

        avg_acc1 = metric.accuracy(ensemble_output, target, topk=(1,))
        avg_top1.update(avg_acc1[0].item(), batch_size_now)
        avg_ce_loss.update(ce_loss.mean().item(), batch_size_now)
        avg_cot_loss.update(cot_loss.mean().item(), batch_size_now)

        total_loss = (ce_loss + cot_loss) / args.iters_to_accumulate

        # Compute gradient and do SGD step if optimizer is provided
        if optimizer is not None:
            total_loss.backward()
            if (i + 1) % args.iters_to_accumulate == 0 or (i + 1) == total_iters:
                optimizer.step()
                optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % (args.print_freq * args.iters_to_accumulate) == 0:
            progress.print(i)

        # Log loss and accuracy to TensorBoard
        global_step = epoch * total_iters + i
        writer.add_scalar('Train/Loss', total_loss.item(), global_step)
        writer.add_scalar('Train/Accuracy', avg_top1.avg, global_step)

    # Close the SummaryWriter
    writer.close()

    # No need to empty cache since we're not using GPU


def validate(val_loader, model, args):
    batch_time = metric.AverageMeter('Time', ':6.3f')
    avg_ce_loss = metric.AverageMeter('ce_loss', ':.4e')

    top1_all = [metric.AverageMeter(f'{i}_Acc@1', ':6.2f') for i in range(args.loop_factor)]
    avg_top1 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
    avg_top5 = metric.AverageMeter('Avg_Acc@5', ':6.2f')

    progress = metric.ProgressMeter(len(val_loader), batch_time, avg_ce_loss, *top1_all, avg_top1, avg_top5, prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            # compute outputs and losses
            ensemble_output, outputs, ce_loss = model(images, target=target, mode='val')

            batch_size_now = images.size(0)
            for j in range(args.loop_factor):
                acc1, acc5 = metric.accuracy(outputs[j], target, topk=(1, 5))
                top1_all[j].update(acc1[0].item(), batch_size_now)

            avg_acc1, avg_acc5 = metric.accuracy(ensemble_output, target, topk=(1, 5))
            avg_top1.update(avg_acc1[0].item(), batch_size_now)
            avg_top5.update(avg_acc5[0].item(), batch_size_now)
            avg_ce_loss.update(ce_loss.mean().item(), batch_size_now)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # Return the computed metrics as a tuple
        return avg_top1.avg, avg_top5.avg, [top1.avg for top1 in top1_all]



if __name__ == '__main__':
    torch.manual_seed(0)  # Set a seed for reproducibility

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    args = parser_params.add_parser_params(parser)

    os.makedirs(args.model_dir, exist_ok=True)
    print(args)
    main_worker(args)