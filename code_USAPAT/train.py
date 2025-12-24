"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
import cv2
import PIL
import numpy as np

# 报错修改
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from PIL import PngImagePlugin
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

import torch

def count_parameters(model):
    """统计 CycleGAN 模型的参数数量"""
    param_counts = {}
    
    # 检查是否包含生成器（G）和判别器（D）
    if hasattr(model, 'netG_A') and isinstance(model.netG_A, torch.nn.Module):
        print('netG_A')
        total_params = sum(p.numel() for p in model.netG_A.parameters())
        trainable_params = sum(p.numel() for p in model.netG_A.parameters() if p.requires_grad)
        param_counts['netG_A'] = (total_params, trainable_params)
    
    if hasattr(model, 'netG_B') and isinstance(model.netG_B, torch.nn.Module):
        total_params = sum(p.numel() for p in model.netG_B.parameters())
        trainable_params = sum(p.numel() for p in model.netG_B.parameters() if p.requires_grad)
        param_counts['netG_B'] = (total_params, trainable_params)
    
    if hasattr(model, 'netD_A') and isinstance(model.netD_A, torch.nn.Module):
        total_params = sum(p.numel() for p in model.netD_A.parameters())
        trainable_params = sum(p.numel() for p in model.netD_A.parameters() if p.requires_grad)
        param_counts['netD_A'] = (total_params, trainable_params)
    
    if hasattr(model, 'netD_B') and isinstance(model.netD_B, torch.nn.Module):
        total_params = sum(p.numel() for p in model.netD_B.parameters())
        trainable_params = sum(p.numel() for p in model.netD_B.parameters() if p.requires_grad)
        param_counts['netD_B'] = (total_params, trainable_params)

    
    
    return param_counts
def print_total_parameters(param_counts):
  total_params = 0
  total_trainable = 0
  
  print("\n===== Detailed Model Parameters =====")
  for name, (total, trainable) in param_counts.items():
      print(f"{name}:")
      print(f"  Total parameters: {total:,}")
      print(f"  Trainable parameters: {trainable:,}")
      total_params += total
      total_trainable += trainable
  
  print("\n===== Overall Summary =====")
  print(f"Total model parameters: {total_params:,}")
  print(f"Trainable parameters: {total_trainable:,}")
  print("===========================\n")


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        time1 = time.time()
        print("Current epoch: ", epoch)
        print("Final epoch: ", opt.n_epochs + opt.n_epochs_decay)
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            # if True:
                print("total_iters % opt.display_freq == 0-------------")
                print("Current number of iterations - total_iters: ", total_iters)
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()

                # print(type(model.get_current_visuals()))
                # print(model.get_current_visuals().keys())
                res = model.get_current_visuals()
                for one_key in res.keys():
                    array = res[one_key].squeeze(0).cpu().detach().numpy()
                    if len(array.shape) == 3:
                        array = np.expand_dims(array, axis=0)
                    # for i in range(10):
                    for i in range(opt.batch_size):  # Display the first few pictures in each batch
                        try:
                            norm_image = cv2.normalize(array[i], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                        except:
                            print("The displayed image exceeds the batch range！")
                            continue
                        im = PIL.Image.fromarray(np.uint8(norm_image.transpose(1, 2, 0)))
                        im.save(opt.name + '/' + str(epoch) + '_' + str(total_iters) + '_' + str(i) + '_' + one_key + '.jpeg')

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                print("total_iters % opt.print_freq == 0--------------------------")
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                # visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        print("1 epoch time consumption: ", time.time() - time1)