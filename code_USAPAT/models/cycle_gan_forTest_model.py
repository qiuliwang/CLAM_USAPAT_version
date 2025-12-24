'''
Mask guided cyclegan
'''

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import cv2
import numpy as np
import PIL
from datetime import datetime

def xor(img1, img2):
    img1 = np.where(img1 > 127, 1, 0)
    img2 = np.where(img2 > 127, 1, 0)
    img_xor = img1 ^ img2
    img_xor = np.where(img_xor > 0, 255.0, 0.0)

    return img_xor

def get_and(img1, img2):
    img1 = np.where(img1 > 127, 1, 0)
    img2 = np.where(img2 > 127, 1, 0)
    img_xor = img1 + img2
    img_xor = np.where(img_xor > 0.5, 255, 0)
    return img_xor


# def FillHole(im_in):
#     im_floodfill = im_in.copy()
#     im_floodfill = np.uint8(im_floodfill)
#     h, w = im_in.shape[:2]
#     mask = np.zeros((h + 2, w + 2), np.uint8)
#
#     isbreak = False
#     # for batch_i in range(im_floodfill.shape[0]):
#     for i in range(im_floodfill.shape[0]):
#         for j in range(im_floodfill.shape[1]):
#             # print("i, j:  ", i, j)
#             # print("im_floodfill_shape： ", im_floodfill.shape)
#             # print("im_floodfill： ", im_floodfill)
#             # print("im_floodfill[i][j]:  ", im_floodfill[i][j])
#             if (im_floodfill[i][j] == 0):
#                 seedPoint = (i, j)
#                 isbreak = True
#                 break
#         if (isbreak):
#             break
#
#     cv2.floodFill(im_floodfill, mask, seedPoint, 255)
#
#     im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#
#     im_out = np.uint8(im_in) | im_floodfill_inv
#     return im_out
def FillHole(im_in):
    im_out = np.empty(im_in.shape, dtype=np.uint8)  # 创建与输入相同大小的输出数组
    # print("im_inshape： ", im_in.shape)

    for i in range(im_in.shape[0]):  # 遍历批处理维度
        im_floodfill = im_in[i, 0, :, :]  # 获取当前批处理中的图像
        im_floodfill = np.uint8(im_floodfill)
        h, w = im_floodfill.shape[:2]
        # print("im_floodfill_nshape： ", im_floodfill.shape)
        # print("im_floodfill_h,w: ： ", h, w)
        mask = np.zeros((h + 2, w + 2), np.uint8)

        isbreak = False
        seedPoint = None
        for x in range(im_floodfill.shape[0]):  # 寻找种子点
            for y in range(im_floodfill.shape[1]):
                if (im_floodfill[x, y] == 0):
                    seedPoint = (x, y)
                    isbreak = True
                    break
            if (isbreak):
                break

        if seedPoint is not None:
            cv2.floodFill(im_floodfill, mask, seedPoint, 255)  # 填充洞
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)  # 反转填充后的图像
            im_out[i, 0, :, :] = np.uint8(im_in[i, 0, :, :]) | im_floodfill_inv  # 按位或操作得到最终结果

    return im_out



class cycleganforTestmodel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseModel.__init__(self, opt)
        # print("=====
        # ==========load_path: ", load_path)
        # breakpoint()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # print(self.netG_A)
        # print(self.netG_B)


        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            # self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMask = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.count = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_mask': A_mask, 'B_mask:': B_mask, 'A_mask_path': A_mask_path, 'B_mask_path': B_mask_path}

        """

        # weight_path = '/home/liuyongxu/code/Generative_Model_WSI/checkpoints/lyx_classic_bad/latest_net_G.pth'
        # aaa = torch.load(weight_path)
        # # 修改参数名称
        # new_state_dict = {}
        # for key, value in aaa.items():
        #     # if 'model.26.bias' in key:
        #     #     new_key = key.replace('model.26.bias', 'module.model.26.bias')
        #     # else:
        #     new_key = "module." + key
        #     new_state_dict[new_key] = value
        # # print(aaa)
        # # breakpoint()
        # self.netG_A = self.netG_A.load_state_dict(new_state_dict)
        # self.netG_A = self.netG_A.load_state_dict(torch.load(weight_path))

        # AtoB = self.opt.direction == 'AtoB'
        # inputa = input['A']
        # inputa_mask_cell = input['A_mask_cell']
        # inputa_mask_blood = input['A_mask_blood']
        #
        # inputb = input['B']
        # inputb_mask_cell = input['B_mask_cell']
        # inputb_mask_stain = input['B_mask_stain']
        # self.real_B_mask_cell = inputb_mask_cell.to(self.device)
        # self.real_B_mask_stain = inputb_mask_stain.to(self.device)

        # self.real_A = torch.cat((inputa, inputa_mask), 1).to(self.device)
        # self.real_B = torch.cat((inputb, inputb_mask), 1).to(self.device)

        self.real_A = input['A'].to(self.device)
        # from PIL import Image
        # aaaaaa = self.real_A.cpu()
        # for ii in range(aaaaaa.size(0)):
        #     image_data = aaaaaa[ii].permute(1, 2, 0).mul(255).byte().numpy()  # 调整维度顺序，并转换为numpy数组
        #     image = Image.fromarray(image_data)
        #     image.save(f'ls/ls/real_{ii}.png')  # 保存为png格式，可以根据需要修改为其他格式
        self.real_B = input['B'].to(self.device)
        # self.real_A_mask_cell = inputa_mask_cell.to(self.device)
        # self.real_A_mask_blood = inputa_mask_blood.to(self.device)

        self.image_paths = input['A_paths']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # print(self.real_A.shape)
        # from PIL import Image
        # aaaaaa = self.real_A.cpu()
        # for ii in range(aaaaaa.size(0)):
        #     image_data = aaaaaa[ii].permute(1, 2, 0).mul(255).byte().numpy()  # 调整维度顺序，并转换为numpy数组
        #     image = Image.fromarray(image_data)
        #     image.save(f'ls/ls/real_{ii}.png')  # 保存为png格式，可以根据需要修改为其他格式
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # print(self.netG_A)
        # bbbbbb = self.fake_B.cpu()
        # for ii in range(bbbbbb.size(0)):
        #     image_data = bbbbbb[ii].permute(1, 2, 0).mul(255).byte().numpy()  # 调整维度顺序，并转换为numpy数组
        #     image = Image.fromarray(image_data)
        #     image.save(f'ls/ls/fake_{ii}.png')  # 保存为png格式，可以根据需要修改为其他格式
        # breakpoint()
        # print("---------------: ", self.fake_B.shape)
        # return self.fake_B

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)

        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # Mask loss
        self.loss_A_mask_cell = self.criterionMask(self.A_cell_mask, self.real_A_mask_cell)
        # print("损失loss_A_mask_cell: ", self.loss_A_mask_cell)
        self.loss_A_mask_blood = self.criterionMask(self.A_blood_mask, self.real_A_mask_blood)
        # print("损失loss_A_mask_blood: ", self.loss_A_mask_blood)

        self.loss_B_mask_cell = self.criterionMask(self.B_cell_mask, self.real_B_mask_cell)
        # print("损失loss_B_mask_cell: ", self.loss_B_mask_cell)
        self.loss_B_mask_stain = self.criterionMask(self.B_stain_mask, self.real_B_mask_stain)      # CD20添加
        # print("损失loss_B_mask_stain: ", self.loss_B_mask_stain.item())

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_A_mask_cell + self.loss_A_mask_blood + self.loss_B_mask_cell + self.loss_B_mask_stain       # 改动
        # print("总损失loss_G: ", self.loss_G)

        # 临时
        # 保存各个mask的损失
        # content = [
        #     "损失loss_A_mask_cell，损失loss_A_mask_blood，损失loss_B_mask_cell，损失loss_B_mask_stain，总损失： ",
        #     str(self.loss_A_mask_cell.item()) + '   ' + str(self.loss_A_mask_blood.item()) + '  ' + str(self.loss_B_mask_cell.item()) + '   ' + str(self.loss_B_mask_stain.item()) + '  ' + str(self.loss_G.item())
        # ]
        # # 打开文件并写入内容
        # with open("/home/liuyongxu/code/Generative_Model_WSI/lyx_loss_log.txt", "a", encoding="utf-8") as f:
        #     for line in content:
        #         f.write(line + "\n")

        # # print("mask的数据类型", self.B_cell_mask.dtype)
        # current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        # _A_cell_mask = self.A_cell_mask.cpu().detach().numpy()
        # _real_A_mask_cell = self.real_A_mask_cell.cpu().detach().numpy()
        # _A_cell_mask = np.where(_A_cell_mask > 0, 255, 0)
        # _real_A_mask_cell = np.where(_real_A_mask_cell > 0, 255, 0)
        # cv2.imwrite(f"/usr/Data/liuyongxu_data/code/Generative_Model_WSI/log_mask/{current_time}_A_cell_mask.png", _A_cell_mask[0].squeeze(0))
        # cv2.imwrite(f"/usr/Data/liuyongxu_data/code/Generative_Model_WSI/log_mask/{current_time}_real_A_mask_cell.png", _real_A_mask_cell[0].squeeze(0))

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        return self.fake_B, self.image_paths
        # # G_A and G_B
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        # self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        # self.backward_G()             # calculate gradients for G_A and G_B
        # self.optimizer_G.step()       # update G_A and G_B's weights
        # # D_A and D_B
        # self.set_requires_grad([self.netD_A, self.netD_B], True)
        # self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        # self.backward_D_A()      # calculate gradients for D_A
        # self.backward_D_B()      # calculate graidents for D_B
        # self.optimizer_D.step()  # update D_A and D_B's weights
