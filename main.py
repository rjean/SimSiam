import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
import shutil
from tqdm import tqdm
from configs import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, PlotLogger, knn_monitor
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import cProfile

import glob
import os

def main(device, args):
    dataset_kwargs = {
        'dataset':args.dataset,
        'data_dir': args.data_dir,
        'download':args.download,
        'debug_subset_size':args.batch_size if args.debug else None
    }
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    
    train_dataset=get_dataset(
            transform=get_aug(args.aug, args.image_size, True, horizontal_flip=not args.nohflip), 
            train=True, 
            **dataset_kwargs)

    memory_dataset=get_dataset(
            transform=get_aug(args.aug, args.image_size, False, train_classifier=False, horizontal_flip=not args.nohflip), 
            train=True,
            only_train=True, #Required for STL10
            **dataset_kwargs)

    test_dataset = get_dataset( 
            transform=get_aug(args.aug, args.image_size, False, train_classifier=False, horizontal_flip=not args.nohflip), 
            train=False,
            **dataset_kwargs)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=memory_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )


    # define model
    model_path = None
    model = get_model(args.model, args.backbone).to(device)

    best_accuracy = 0

    # define optimizer
    optimizer = get_optimizer(
        args.optimizer, model, 
        lr=args.base_lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay)


    lr_scheduler = LR_Scheduler(
        optimizer,
        args.warmup_epochs, args.warmup_lr, 
        args.num_epochs, args.base_lr, args.final_lr, 
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    ) 


    #model_folder = os.path.join(args.output_dir, f'{args.model}-{args.dataset}-epoch{epoch+1}.pth')
    load_epoch = 0 
    if args.resume_from_last:
        #https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder/39327156#39327156
        list_of_files = glob.glob(f'{args.output_dir}/*.pth') # * means all if need specific format then *.csv
        if len(list_of_files)>0:
            model_path = max(list_of_files, key=os.path.getctime)
            print(f"Loading model parameters from {model_path}")
            saved = torch.load(model_path)
            model.load_state_dict(saved['state_dict'])
            lr_scheduler = saved['lr_scheduler']
            #Parse the epoch number from the filename.
            load_epoch = saved["epoch"]
            model.eval()
        else:
            print("No checkpoint found for this experiment. Starting from scratch")
            #No save found!
            pass

    if args.model == 'simsiam' and args.proj_layers is not None: model.projector.set_layers(args.proj_layers)
    model = torch.nn.DataParallel(model)
    if torch.cuda.device_count() > 1: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    



    loss_meter = AverageMeter(name='Loss')
    plot_logger = PlotLogger(params=['lr', 'loss', 'accuracy'])
    writer = SummaryWriter()

    # Start training
    global_progress = tqdm(range(load_epoch, args.stop_at_epoch), desc=f'Training')

    skip_training = False
    if len(global_progress)==0:
        print(f"Training already of {load_epoch} epochs complete. Proceding to evaluation.")
        global_progress=tqdm([load_epoch])
        skip_training=True
    #pr = cProfile.Profile()
    
    for epoch in global_progress:
        #pr.enable()
        loss_meter.reset()
        model.train()
        
        # plot_logger.update({'epoch':epoch, 'accuracy':accuracy})
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.num_epochs}', disable=args.hide_progress)
        lr=0
        for idx, ((images1, images2), labels) in enumerate(local_progress):
            if skip_training:
                break #Training is completed. Skip to evaluation.
            if idx==0:
                grid1 = torchvision.utils.make_grid(images1, normalize=True)
                grid2 = torchvision.utils.make_grid(images2, normalize=True)
                both = torch.cat((grid1,grid2),dim=2)
                writer.add_image('image_pairs', both, epoch)
            with autocast():
                model.zero_grad()
                loss = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
            
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            lr = lr_scheduler.step()
            
            data_dict = {'lr':lr, "loss":loss_meter.val}
            local_progress.set_postfix(data_dict)
            plot_logger.update(data_dict)
            if args.dry_run:
                print("Warning: Dry run mode enable. No iteration will be performed.")
                break #Do not iterrate for dry-running.
        accuracy = knn_monitor(model.module.backbone, memory_loader, test_loader, epoch, k=200, hide_progress=args.hide_progress, writer=writer, output_dir= args.output_dir)
        global_progress.set_postfix({"epoch":epoch, "loss_avg":loss_meter.avg, "accuracy":accuracy})
        plot_logger.update({'accuracy':accuracy})
        plot_logger.save(os.path.join(args.output_dir, 'logger.svg'))
        writer.add_scalar("Loss/train", loss_meter.avg, epoch)
        writer.add_scalar("Learning rate", lr, epoch)
        writer.add_scalar("NN Accuracy/valid", accuracy, epoch)

        # Save checkpoint at the end of each epoch 
        
        model_path = os.path.join(args.output_dir, f'{args.model}-{args.dataset}-last.pth')
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.module.state_dict(),
            # 'optimizer':optimizer.state_dict(), # will double the checkpoint file size
            'lr_scheduler':lr_scheduler,
            'args':args,
            'loss_meter':loss_meter,
            'plot_logger':plot_logger
        }, model_path)
        
        print(f"Model saved to {model_path}")
        if accuracy>best_accuracy:
            best_accuracy = accuracy
            best_model_path = model_path.replace("-last.pth", "-best.pth")
            print(f"Best model so far, saved to {best_model_path}")
            shutil.copy(model_path, best_model_path)
        #pr.disable()
        #pr.print_stats("cumulative")
        #break


        

    if args.eval_after_train is not None:
        args.eval_from = model_path
        arg_list = [x.strip().lstrip('--').split() for x in args.eval_after_train.split('\n')]
        args.__dict__.update({x[0]:eval(x[1]) for x in arg_list})
        if args.debug: 
            args.batch_size = 2
            args.num_epochs = 3

        linear_eval(args)

if __name__ == "__main__":
    args = get_args()
    # print(args.device)
    main(device=args.device, args=args)
















