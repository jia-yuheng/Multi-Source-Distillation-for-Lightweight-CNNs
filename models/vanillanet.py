import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import weight_init, DropPath
from timm.models.registry import register_model

from .hsmssd import HSMSSD


class GatedActivation(nn.Module):
    def __init__(self, act_learn=3.0):
        super().__init__()
        self.act_learn = act_learn

    def forward(self, x):
        return x * torch.sigmoid(self.act_learn * x)


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-6),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.adapter(x)



class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False, ada_pool=None):
        super().__init__()
        self.act_learn = 1
        self.deploy = deploy
        #self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim, bias=False)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=5, stride=stride, padding=2, groups=dim, bias=False)

        self.pointwise = nn.Conv2d(dim, dim_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dim_out, eps=1e-6)
        self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((ada_pool, ada_pool)) if ada_pool else nn.MaxPool2d(stride)
        self.act = activation(dim_out, act_num, deploy=self.deploy)

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_out, dim_out // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out // 8, dim_out, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        #x = torch.nn.functional.leaky_relu(x, self.act_learn)
        self.gate = GatedActivation(self.act_learn)
        x = self.gate(x)

        x = self.pool(x)
        x = self.act(x)

        if hasattr(self, 'use_hsm') and self.use_hsm:
            B, C, H, W = x.shape
            L = H * W
            # 打印进入 HSM 前的形状
            #print(f"[HSMSSD] before: x.shape = {x.shape}")  # 调试用

            # 1) 展平 (B, C, L)
            seq = x.view(B, C, L)

            # 2) LayerNorm：转为 (B, L, C)
            seq = seq.permute(0, 2, 1)
            seq = self.ln_hsm(seq)  # ln_hsm = nn.LayerNorm(C)
            seq = seq.permute(0, 2, 1)

            # 打印送入 HSM 的 seq 形状
            #print(f"[HSMSSD] seq for HSM: {seq.shape}")  # 应该是 (B, C, L)

            # 3) 调用 HSM-SSD，输出 y: (B, C, H, H)
            y, _ = self.hsm(seq)

            # 打印 HSM 输出形状
            #print(f"[HSMSSD] after: y.shape = {y.shape}")  # 调试用

            # 用 y 替换 x
            x = y


        x = x * self.attn(x)   

        if hasattr(self, 'adapter') and self.adapter is not None:
            adapter_feat = self.adapter(x)
            return x, adapter_feat
        else:
            return x

    def switch_to_deploy(self):
        self.act.switch_to_deploy()


# Series informed activation function. Implemented by conv.
class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.act_num = act_num
        self.deploy = deploy
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.dilation = 2
        self.padding = self.act_num * self.dilation 


        if deploy:
            self.bias = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None
            self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, self.bias, padding=self.act_num, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=self.padding, dilation=self.dilation, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class Block(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False, ada_pool=None):
        super().__init__()
        self.act_learn = 1
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((ada_pool, ada_pool))

        self.act = activation(dim_out, act_num, deploy=self.deploy)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            #x = torch.nn.functional.leaky_relu(x, self.act_learn)
            self.gate = GatedActivation(self.act_learn)
            x = self.gate(x)
            x = self.conv2(x)
        x = self.pool(x)
        x = self.act(x)

        # 如果当前 Block 已挂载 adapter，则计算适配器输出
        if hasattr(self, 'adapter') and self.adapter is not None:
            adapter_feat = self.adapter(x)
            return x, adapter_feat
        else:
            return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1, 3),
                                             self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1, 3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True


class VanillaNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, dims=[96, 192, 384, 768],
                 drop_rate=0, act_num=3, strides=[2, 2, 2, 1], deploy=False, ada_pool=None, **kwargs):

     
        self.hsm_state_dim_ratio = kwargs.pop('hsm_state_dim_ratio', 0.25)

        super().__init__()

        self.hsm_blocks = kwargs.pop('hsm_blocks', [])
   
        self.deploy = deploy
        stride, padding = (4, 0) if not ada_pool else (3, 1)
        if self.deploy:
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=stride, padding=padding),
                activation(dims[0], act_num, deploy=self.deploy)
            )
        else:
            self.stem1 = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=stride, padding=padding),
                nn.BatchNorm2d(dims[0], eps=1e-6),
            )
            self.stem2 = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(dims[0], eps=1e-6),
                activation(dims[0], act_num)
            )

        self.act_learn = 1

        self.stages = nn.ModuleList()

        self.dw_blocks = kwargs.pop("dw_blocks", [])  
        for i in range(len(strides)):
            BlockClass = DepthwiseSeparableBlock if i in self.dw_blocks else Block
            if not ada_pool:
                stage = BlockClass(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy)
            else:
                stage = BlockClass(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy, ada_pool=ada_pool[i])
            self.stages.append(stage)


        for idx in self.hsm_blocks:
            blk = self.stages[idx]
       
            blk.use_hsm = True

            C_out = blk.pointwise.out_channels if isinstance(blk, DepthwiseSeparableBlock) else blk.act.dim
            blk.ln_hsm = nn.LayerNorm(C_out)

            #blk.hsm = HSMSSD(d_model=C_out, state_dim=C_out // 4)

            state_dim = max(1, int(C_out * self.hsm_state_dim_ratio))
            blk.hsm = HSMSSD(d_model=C_out, state_dim=state_dim)




        self.depth = len(strides)

        if self.deploy:
            self.cls = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(dims[-1], num_classes, 1),
            )
        else:
            self.cls1 = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(dims[-1], num_classes, 1),
                nn.BatchNorm2d(num_classes, eps=1e-6),
            )
            self.cls2 = nn.Sequential(
                nn.Conv2d(num_classes, num_classes, 1)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight_init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def change_act(self, m):
        for i in range(self.depth):
            self.stages[i].act_learn = m
        self.act_learn = m



    def forward(self, x):
        features = {} 
        if self.deploy:
            x = self.stem(x)
        else:
            x = self.stem1(x)

            #x = torch.nn.functional.leaky_relu(x, self.act_learn)
            self.gate = GatedActivation(self.act_learn)
            x = self.gate(x)

            x = self.stem2(x)


        for i in range(self.depth):
            stage_output = self.stages[i](x)
            if isinstance(stage_output, tuple):
                x, adapter_feat = stage_output
                features[f'block_{i}'] = adapter_feat
            else:
                x = stage_output
                features[f'block_{i}'] = x 

        if self.deploy:
            x = self.cls(x)
        else:
            x = self.cls1(x)
            #x = torch.nn.functional.leaky_relu(x, self.act_learn)
            self.gate = GatedActivation(self.act_learn)
            x = self.gate(x)

            x = self.cls2(x)

        return x.view(x.size(0), -1), features




    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        self.stem2[2].switch_to_deploy()
        kernel, bias = self._fuse_bn_tensor(self.stem1[0], self.stem1[1])
        self.stem1[0].weight.data = kernel
        self.stem1[0].bias.data = bias
        kernel, bias = self._fuse_bn_tensor(self.stem2[0], self.stem2[1])
        self.stem1[0].weight.data = torch.einsum('oi,icjk->ocjk', kernel.squeeze(3).squeeze(2),
                                                 self.stem1[0].weight.data)
        self.stem1[0].bias.data = bias + (self.stem1[0].bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        self.stem = torch.nn.Sequential(*[self.stem1[0], self.stem2[2]])
        self.__delattr__('stem1')
        self.__delattr__('stem2')

        for i in range(self.depth):
            self.stages[i].switch_to_deploy()

        kernel, bias = self._fuse_bn_tensor(self.cls1[2], self.cls1[3])
        self.cls1[2].weight.data = kernel
        self.cls1[2].bias.data = bias
        kernel, bias = self.cls2[0].weight.data, self.cls2[0].bias.data
        self.cls1[2].weight.data = torch.matmul(kernel.transpose(1, 3),
                                                self.cls1[2].weight.data.squeeze(3).squeeze(2)).transpose(1, 3)
        self.cls1[2].bias.data = bias + (self.cls1[2].bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        self.cls = torch.nn.Sequential(*self.cls1[0:3])
        self.__delattr__('cls1')
        self.__delattr__('cls2')
        self.deploy = True


@register_model
def vanillanet_5(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128 * 4, 256 * 4, 512 * 4, 1024 * 4], strides=[2, 2, 2], **kwargs)
    return model


# 修改后的 VanillaNet_6 定义
@register_model
def vanillanet_6(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[64, 128, 256, 512, 512],
        strides=[1, 1, 2, 1],   # 原始 [2, 2, 2, 1]，后改为 [1,1,2,1]

        #dw_blocks=[0, 1],  # ✅ 新增，控制哪些 Block 使用 DW 分离卷积

        **kwargs
    )
    # 定义 Adapter 的设置，选择对齐 Block 1（索引 1）输出与教师的 layer3
    adapter_settings = {0: 128, 1: 256}
    for idx, stage in enumerate(model.stages):
        if idx in adapter_settings:
            target_channels = adapter_settings[idx]
            # 这里建议直接利用当前 Block 输出的通道数（一般为 dims[idx+1]）；
            # 你也可改为 stage.act.dim，如果保证该值正确代表输出通道数
            current_channels = stage.act.dim
            stage.adapter = FeatureAdapter(in_channels=current_channels, out_channels=target_channels)
    return model





@register_model
def vanillanet_7(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 1024 * 4, 1024 * 4], strides=[1, 2, 2, 2, 1], **kwargs)
    return model


@register_model
def vanillanet_8(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
                       strides=[1, 2, 2, 1, 2, 1], **kwargs)
    return model


@register_model
def vanillanet_9(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
                       strides=[1, 2, 2, 1, 1, 2, 1], **kwargs)
    return model


@register_model
def vanillanet_10(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
        strides=[1, 2, 2, 1, 1, 1, 2, 1],
        **kwargs)
    return model


@register_model
def vanillanet_11(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
        strides=[1, 2, 2, 1, 1, 1, 1, 2, 1],
        **kwargs)
    return model


@register_model
def vanillanet_12(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4, 1024 * 4],
        strides=[1, 2, 2, 1, 1, 1, 1, 1, 2, 1],
        **kwargs)
    return model


@register_model
def vanillanet_13(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 4, 128 * 4, 256 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 512 * 4, 1024 * 4,
              1024 * 4],
        strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
        **kwargs)
    return model


@register_model
def vanillanet_13_x1_5(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 6, 128 * 6, 256 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 1024 * 6,
              1024 * 6],
        strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
        **kwargs)
    return model


@register_model
def vanillanet_13_x1_5_ada_pool(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128 * 6, 128 * 6, 256 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 512 * 6, 1024 * 6,
              1024 * 6],
        strides=[1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1],
        ada_pool=[0, 38, 19, 0, 0, 0, 0, 0, 0, 10, 0],
        **kwargs)
    return model
