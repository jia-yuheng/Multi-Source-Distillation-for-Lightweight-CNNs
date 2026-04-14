import argparse
import os
import warnings
warnings.filterwarnings('ignore')
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import math
from pathlib import Path
from timm.data import create_dataset, create_loader, RealLabelsImagenet, Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import models.vanillanet
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, align_layer='layer3'):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        feat2 = self.layer2(x)  # 捕获 layer2 特征
        feat3 = self.layer3(feat2)  # 捕获 layer3 特征
        x = self.layer4(feat3)
        x = self.pool(x).view(x.size(0), -1)
        logits = self.fc(x)

        # 根据参数选择对齐层
        teacher_features = {}
        if align_layer == 'layer2':
            teacher_features['layer2'] = feat2
        elif align_layer == 'layer3':
            teacher_features['layer3'] = feat3
        elif align_layer == 'both':
            teacher_features = {'layer2': feat2, 'layer3': feat3}
        else:
            # 如果传入其他值，则不返回中间特征
            teacher_features = {}
        return logits, teacher_features

def ResNet18(num_classes=100, align_layer='layer3'):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    # 可以将 align_layer 作为属性保存，这样在调用 forward 时可直接读取
    model.align_layer = align_layer
    return model


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Vanillanet script', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--early_stop_epochs', default=None, type=int)
    parser.add_argument('--decay_epochs', default=100, type=int,
                        help='for deep training strategy')
    parser.add_argument('--decay_linear', type=str2bool, default=True,
                        help='cos/linear for decay manner')
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='vanillanet_6', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--switch_to_deploy', default=None, type=str)
    parser.add_argument('--deploy', type=str2bool, default=False)
    parser.add_argument('--drop', type=float, default=0, metavar='PCT',
                        help='Drop rate (default: 0.0)')
    parser.add_argument('--input_size', default=32, type=int,
                        help='image input size')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.999, nargs='+')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',     # cifar100 1e-4
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--layer_decay_num_layers', default=4, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.2, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='', metavar='',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--bce_loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce_target_thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.1, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.2,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/cifar100', type=str,
                        help='CIFAR-100 dataset path (will auto-download)')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=100, type=int,     # --nb_classes默认值从10改为100 CIFAR-100
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=False)
    parser.add_argument('--data_set', default='CIFAR100', choices=['CIFAR', 'CIFAR100','IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_new_sched', action='store_true', help='resume with new schedule')
    parser.set_defaults(resume_new_sched=False)
    parser.add_argument('--auto_resume', type=str2bool, default=False)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--test_freq', default=20, type=int)
    parser.add_argument('--test_epoch', default=260, type=int)
    parser.add_argument('--save_ckpt_num', default=10, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")
    
    parser.add_argument('--act_num', default=3, type=int)
    parser.add_argument('--real_labels', default='', type=str, metavar='FILENAME',
                        help='Real labels JSON file for imagenet evaluation')

    parser.add_argument('--kd_alpha', type=float, default=0.8,
                        help='知识蒸馏损失权重 (默认: 0.4)')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature scaling for distillation (default: 3.0)')
    # 新增对齐损失权重参数
    parser.add_argument('--lambda_align', type=float, default=1,
                        help='权重对齐损失（默认: 0.1）')

    parser.add_argument('--hsm_blocks', type=int, nargs='*', default=[3],
                        help='指定在哪些 Block （索引 0~3） 中插入 HSM-SSD')

    parser.add_argument('--hsm_state_dim_ratio', type=float, default= 0.03125,
                        help = 'HSM-SSD 内部 ratio =  state_dim / C_out=512 (默认 0.25)')
    # 4 / 512 = 0.0078125
    # 8 / 512 = 0.015625
    # 16 / 512 = 0.03125
    # 32 / 512 = 0.0625
    # 64 / 512 = 0.125

    return parser

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # ========== 新增强制覆盖归一化参数 ==========
    if args.data_set == 'CIFAR100':
        args.imagenet_default_mean_and_std = False  # 禁用 ImageNet 默认参数
    # ==========================================

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 在数据加载前强制覆盖归一化参数
    if args.data_set == 'CIFAR100':
        '''
        # 将 CIFAR-10 的统计参数写入 args
        args.mean = (0.4914, 0.4822, 0.4465)
        args.std = (0.2470, 0.2435, 0.2616)
        '''

        args.mean = (0.5071, 0.4867, 0.4408)  # CIFAR-100 均值
        args.std = (0.2675, 0.2565, 0.2761)  # CIFAR-100 标准差

        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        dataset_val, _ = build_dataset(is_train=False, args=args)
    else:
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        if args.disable_eval:
            args.dist_eval = False
            dataset_val = None
        else:
            dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)



    model = create_model(
        args.model, 
        pretrained=False,
        num_classes=args.nb_classes, 
        act_num=args.act_num,
        drop_rate=args.drop,
        deploy=args.deploy,
        dw_blocks=[0, 1,2,3],  # ✅ 加上这行
        hsm_blocks=args.hsm_blocks,  # 将 hsm 参数传入 VanillaNet
        hsm_state_dim_ratio=args.hsm_state_dim_ratio,
        )




    # ======================= Block 结构验证 =======================
    if 'vanilla' in args.model.lower():
        print("\n====== 模型结构检查（Block 类型） ======")
        for i, stage in enumerate(model.stages):
            print(f"Block {i}: {stage.__class__.__name__}")
        print("=====================================================\n")

    # ========== 修改教师模型加载 ==========
    # 使用自训练的 ResNet18 作为教师模型，权重文件位于当前目录下
    #teacher = ResNet18(num_classes=args.nb_classes).to(device)
    #可根据实验选择适合自己的对齐层（例如默认使用 layer3）
    teacher = ResNet18(num_classes=args.nb_classes, align_layer='layer3').to(device)

    teacher.load_state_dict(torch.load('./resnet18_cifar100_best.pth', map_location=device))
    # 冻结教师模型的所有参数（禁止梯度更新）
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()  # 固定BatchNorm等层
    print("Custom Resnet18 Teacher Model loaded.")

    # 在开始训练前，先验证教师模型在验证集上的准确率
    if dataset_val is not None:
        print("Evaluating teacher model on validation set...")
        teacher_eval_stats = evaluate(data_loader_val, teacher, device, use_amp=args.use_amp)
        print(
            f"Teacher Model Accuracy on validation: Top-1: {teacher_eval_stats['acc1']:.2f}%, Top-5: {teacher_eval_stats['acc5']:.2f}%")




    # ================== 修改开始 ==================
    # 修正模型输入层（根据 vanillanet.py 的实际结构）
    if args.input_size == 32 and 'vanilla' in args.model.lower():
        # 正确修改位置：替换 stem1 的第一个卷积层
        original_conv = model.stem1[0]  # stem1是nn.Sequential中的第一个卷积层
        new_conv = nn.Conv2d(
            in_channels=original_conv.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=3,  # 原代码是4x4
            stride=1,  # 原代码是stride=4
            padding=1,  # 保持32x32输出
            bias=original_conv.bias is not None
        )
        # 合理的权重初始化（替代原代码的均值填充）
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        if new_conv.bias is not None:
            nn.init.constant_(new_conv.bias, 0)
        model.stem1[0] = new_conv
    # ================== 修改结束 ==================

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    
    model.to(device)

    # === dummy 测试 HSMSSD 插件 ===
    print(">>>>> 验证 HSM-SSD 插件的前向流程 <<<<<")
    dummy = torch.randn(2, 3, args.input_size, args.input_size).to(device)
    # 只跑到 mid-stage: 调用到 DepthwiseSeparableBlock.forward 时，会打印 shape
    _ = model(dummy)
    print(">>>>> 如果没有报错，shape 打印符合预期，就表示集成成功 <<<<<")


    # ================== 新增验证代码 ==================
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    try:
        # 现在模型返回 (final_output, features)
        output, features = model(dummy_input)
        print("[验证成功] 模型最终输出尺寸:", output.shape)  # 应输出 torch.Size([2, 100]) 对于 CIFAR100
        # 可以打印部分中间特征信息以确认 adapter 模块工作，例如：
        for key, feat in features.items():
            print(f"[验证成功] {key} 特征尺寸: {feat.shape}")
    except Exception as e:
        print("[验证失败] 模型结构错误:", e)
        exit(1)

    # ================================================

    if args.switch_to_deploy:
      model.switch_to_deploy()
      model_ckpt = dict()
      model_ckpt['model'] = model.state_dict()
      torch.save(model_ckpt, args.switch_to_deploy)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = []
        for ema_decay in args.model_ema_decay:
            model_ema.append(
                ModelEma(model, decay=ema_decay, device='cpu' if args.model_ema_force_cpu else '', resume='')
            )
        print("Using EMA with decay = %s" % args.model_ema_decay)

    # 确保以下代码存在
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model  # 单卡训练时直接引用原始模型


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    
    if not args.eval:
        input_size = [2, 3, args.input_size, args.input_size]
        input = torch.randn(input_size).cuda()
        from torchprofile import profile_macs
        macs = profile_macs(model, input)
        print('model flops (G):', macs / 2 / 1.e9, 'input_size:', input_size)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = args.layer_decay_num_layers
        assigner = LayerDecayValueAssigner(num_max_layer=num_layers, values=list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    
    if mixup_fn is not None:
        if args.bce_loss:
            criterion = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    
    if args.eval:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        if args.real_labels:
            dataset = create_dataset(root=args.data_path, name='', split='validation', class_map='')
            real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
        else:
            real_labels = None
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp, real_labels=real_labels)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    max_accuracy_epoch = 0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0
        max_accuracy_ema_epoch = 0
        best_ema_decay = args.model_ema_decay[0]

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
        if 'VanillaNet' == model_without_ddp.__class__.__name__ and epoch <= args.decay_epochs:
            if args.decay_linear:
                act_learn = epoch / args.decay_epochs * 1.0
            else:
                act_learn = 0.5 * (1 - math.cos(math.pi * epoch / args.decay_epochs)) * 1.0
            print(f"VanillaNet decay_linear: {args.decay_linear}, act_learn weight: {act_learn:.3f}")
            model_without_ddp.change_act(act_learn)  # ✅ 直接操作原始模型


        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp,
            # ========== 新增参数传递 ==========
            teacher=teacher,  # 传递教师模型
            kd_alpha = args.kd_alpha,  # 传递损失权重
            temperature=args.temperature,  # 新增参数
            lambda_align = args.lambda_align  # 传递对齐损失权重
        )
        
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, epoch_name=str(epoch), model_ema=model_ema[0])
        
        if (data_loader_val is not None) and (epoch > 0) and (epoch % args.test_freq == 0 or epoch > args.test_epoch):
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                max_accuracy_epoch = epoch
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, epoch_name="best", model_ema=model_ema[0])

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                for idx, iter_model_ema in enumerate(model_ema):
                    test_stats_ema = evaluate(data_loader_val, iter_model_ema.ema, device, use_amp=args.use_amp)
                    print(f"Accuracy of the {args.model_ema_decay[idx]} EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                    if max_accuracy_ema < test_stats_ema["acc1"]:
                        max_accuracy_ema = test_stats_ema["acc1"]
                        max_accuracy_ema_epoch = epoch
                        best_ema_decay = args.model_ema_decay[idx]
                        if args.output_dir and args.save_ckpt:
                            utils.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, epoch_name="best-ema", model_ema=iter_model_ema)
                
                print(f'Max Acc: {max_accuracy:.3f}% @{max_accuracy_epoch}, {best_ema_decay} EMA: {max_accuracy_ema:.3f}% @{max_accuracy_ema_epoch}')
                if log_writer is not None:
                    log_writer.update(ema_test_acc1=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             **{f'ema_test_{k}': v for k, v in test_stats_ema.items()},
                             'epoch': epoch, 'n_parameters': n_parameters}
            else:
                print(f'Max Acc.: {max_accuracy:.3f}% @{max_accuracy_epoch}')
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch, 'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)
            
        if args.early_stop_epochs and epoch == args.early_stop_epochs:
            break

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    time.sleep(10)
    if args.real_labels and args.model_ema_eval:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        dataset = create_dataset(root=args.data_path, name='', split='validation', class_map='')
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
        print('Start eval on REAL.')
        ckpt = torch.load(os.path.join(args.output_dir, 'checkpoint-best-ema.pth'), map_location='cpu')
        msg = model_without_ddp.load_state_dict(ckpt['model_ema'])
        print(msg)
        test_stats = evaluate(data_loader_val, model_without_ddp, device, use_amp=args.use_amp, real_labels=real_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vanillanet script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
