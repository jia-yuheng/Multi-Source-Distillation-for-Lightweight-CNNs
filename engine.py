# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import logging

import utils

from utils import gradient_flow_repair


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False,teacher=None, kd_alpha=0.7,temperature=3.0,lambda_align=0.2):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('kd_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('ce_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('alignment_loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    print(f"当前参数: kd_alpha={kd_alpha}, temperature={temperature}")


    optimizer.zero_grad()

    
    
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ▼▼▼ 移动至Mixup处理之后 ▼▼▼
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

            # ▼▼▼ 将教师前向移至此处 ▼▼▼
            with torch.no_grad():
                teacher_output = teacher(samples.to(device), align_layer=teacher.align_layer)  # 确保教师模型返回的 teacher_features 是你期望的（例如 layer3 输出）


        else:
            # ▼▼▼ 非Mixup情况的教师前向 ▼▼▼
            with torch.no_grad():
                teacher_output =  teacher(samples.to(device))

        # ====================== 蒸馏及对齐损失计算 ======================
        if use_amp:
            with torch.cuda.amp.autocast():
                # 学生模型前向，解包：输出 logits 和中间特征字典 student_features
                student_output, student_features = model(samples)
                ce_loss = criterion(student_output, targets)
                # 教师模型 logits 已存放在 teacher_output[0]，teacher_features 存放在 teacher_output[1]
                soft_teacher = torch.nn.functional.softmax(teacher_output[0] / temperature, dim=-1)
                soft_student = torch.nn.functional.log_softmax(student_output / temperature, dim=-1)
                kd_loss = torch.nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean') * (
                            temperature ** 2)
                # 对齐损失计算：选用学生模型的 'block_1' 适配器输出对齐教师模型的 'layer3' 特征
                alignment_loss = 0.0
                selected_student_key = 'block_1'
                if isinstance(teacher_output, tuple):
                    _, teacher_features = teacher_output
                    if selected_student_key in student_features and 'layer3' in teacher_features:
                        student_feat = student_features[selected_student_key]
                        teacher_feat = teacher_features['layer3']
                        # 如果空间尺寸不同，则使用 adaptive 池化调整学生特征到教师特征相同的空间尺寸
                        if student_feat.shape[2:] != teacher_feat.shape[2:]:
                            student_feat = torch.nn.functional.adaptive_avg_pool2d(student_feat, teacher_feat.shape[2:])
                        alignment_loss = torch.nn.functional.mse_loss(student_feat, teacher_feat)
                loss = kd_alpha * kd_loss + (1 - kd_alpha) * ce_loss + lambda_align * alignment_loss
        else:
            # full precision 分支：与 AMP 分支类似
            student_output, student_features = model(samples)
            ce_loss = criterion(student_output, targets)
            soft_teacher = torch.nn.functional.softmax(teacher_output[0] / temperature, dim=-1)
            soft_student = torch.nn.functional.log_softmax(student_output / temperature, dim=-1)
            kd_loss = torch.nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
           
            # AT Loss 的辅助函数
            alignment_loss = 0.0
            selected_student_key = 'block_1'
            selected_teacher_key = teacher.align_layer  # 这里应该为 'layer3'
            if isinstance(teacher_output, tuple):
                _, teacher_features = teacher_output
                if selected_student_key in student_features and selected_teacher_key in teacher_features:
                    student_feat = student_features[selected_student_key]
                    teacher_feat = teacher_features[selected_teacher_key]
                    # 如果学生和教师特征的空间尺寸不同，则对学生特征使用 AdaptiveAvgPool2d 调整到教师特征相同的尺寸
                    if student_feat.shape[2:] != teacher_feat.shape[2:]:
                        student_feat = torch.nn.functional.adaptive_avg_pool2d(student_feat, teacher_feat.shape[2:])
                    # 使用官方 AT Loss 辅助函数计算对齐损失
                    alignment_loss = utils.at_loss(student_feat, teacher_feat)
                    
            loss = kd_alpha * kd_loss + (1 - kd_alpha) * ce_loss + lambda_align * alignment_loss
        # ================================================================

        loss_value = loss.item()

        # ▶▶▶ 新增两行 ▶▶▶
        metric_logger.update(kd_loss=kd_loss.item())
        metric_logger.update(ce_loss=ce_loss.item())
        metric_logger.update(
            alignment_loss=alignment_loss if isinstance(alignment_loss, float) else alignment_loss.item())

        # ◀◀◀ 修改结束 ◀◀◀

        if not math.isfinite(loss_value): # this could trigger if using AMP
            logging.error("Logging: Loss is {}, stopping training".format(loss_value))
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    for iter_model_ema in model_ema:
                        iter_model_ema.update(model)
                        for i in range(len(iter_model_ema.ema.stages)):
                            if hasattr(iter_model_ema.ema.stages[i], 'act_learn'):
                                iter_model_ema.ema.stages[i].act_learn = model.module.stages[i].act_learn
                            if hasattr(iter_model_ema.ema, 'act_learn'):
                                iter_model_ema.ema.act_learn = model.module.act_learn
        else: # full precision
            loss /= update_freq
            loss.backward()
            
            
            # >>> 在这里调用梯度流修复函数 <<<
            # 注意：model 为当前模型对象（如 model 或 model_without_ddp，根据实际情况）
            gradient_flow_repair(model, low_threshold=1e-3, high_threshold=1.0)


            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()

                if model_ema is not None:
                    for iter_model_ema in model_ema:
                        iter_model_ema.update(model)
                        for i in range(len(iter_model_ema.ema.stages)):
                            if hasattr(iter_model_ema.ema.stages[i], 'act_learn'):
                                iter_model_ema.ema.stages[i].act_learn = model.module.stages[i].act_learn
                            if hasattr(iter_model_ema.ema, 'act_learn'):
                                iter_model_ema.ema.act_learn = model.module.act_learn

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_avg=avg_grad, grad_min=min_grad, grad_max=max_grad, head="grad", step=it)
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, real_labels=None):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # 切换为评估模式
    model.eval()
    for batch in metric_logger.log_every(data_loader, 200, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                # 如果模型返回 tuple，则取第一个元素（final logits）
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(output, target)
        else:
            output = model(images)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)

        if real_labels is not None:
            real_labels.add_result(output)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* val Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        acc1, acc5 = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
        print('* REAL Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc1, acc5))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

