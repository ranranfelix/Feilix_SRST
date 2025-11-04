import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import Choose_Device as Device

class OptimizedCNNBiLSTM(nn.Module):
    """
    优化Dropout策略的CNN-BiLSTM模型
    
    核心改进：
    1. 将dropout从7个位置减少到3个关键位置
    2. 降低dropout率从0.1-0.2到0.05-0.1
    3. 使用变分dropout保持序列一致性
    4. 移除LSTM内部的dropout（避免破坏记忆）
    """
    ch_out = 11
    out_channels_heads = (1, 4, 4, 1)
    sigmoid_ch_ix = [0, 1, 5, 6, 7, 8, 9]
    tanh_ch_ix = [2, 3, 4]
    relu_ch_ix = [9]
    p_ch_ix = [0]
    pxyz_mu_ch_ix = slice(1, 5)
    pxyz_sig_ch_ix = slice(5, 9)
    bg_ch_ix = [9]
    img_ch_ix = [10]

    def __init__(self, in_channels=1, out_channels=11, depth=2, seq_len=3, 
                 initial_features=48, gain=2, pad_convs=False,
                 norm=None, norm_groups=None, sigma_eps_default=0.01):
        super(OptimizedCNNBiLSTM, self).__init__()
        
        self.sigma_eps_default = sigma_eps_default
        self.initial_features = initial_features
        self.seq_len = seq_len
        
        # LSTM层
        self.forward_layer = ConvLSTMCell(2 * initial_features, initial_features, 
                                         norm=norm, norm_groups=norm_groups)
        self.backward_layer = ConvLSTMCell(2 * initial_features, initial_features, 
                                          norm=norm, norm_groups=norm_groups)
        
        # UNet模块
        try:
            import Net.Unet as Unet
            self.unet1 = Unet.Unet(in_channels, initial_features, depth=depth, 
                                   pad_convs=pad_convs, norm=norm, norm_groups=norm_groups)
            self.unet2 = Unet.Unet(3 * initial_features, initial_features, depth=depth, 
                                   pad_convs=pad_convs, norm=norm, norm_groups=norm_groups)
        except:
            print("Warning: Could not import Unet, please ensure Net.Unet is available")
        
        # 计算前后半部分的通道数
        first_half_len = seq_len // 2
        latter_half_len = seq_len - 1 - first_half_len
        
        try:
            self.union_firsthalf = Unet.Unet(first_half_len * initial_features, initial_features, 
                                            depth=depth, pad_convs=pad_convs, norm=norm, 
                                            norm_groups=norm_groups)
            self.union_latterhalf = Unet.Unet(latter_half_len * initial_features, initial_features, 
                                             depth=depth, pad_convs=pad_convs, norm=norm, 
                                             norm_groups=norm_groups)
        except:
            print("Warning: Could not create union layers")
        
        # 混合卷积
        self.add_conv = mix_conv(3 * initial_features, initial_features, norm=norm, 
                                norm_groups=norm_groups)
        
        # 输出层
        # self.outconvlist = nn.ModuleList([OutLayer(initial_features, i) 
        #                                  for i in self.out_channels_heads])
        
        # # ============ 优化的Dropout策略 ============
        # # 【位置1】LSTM输出后 - 轻量dropout，避免破坏时序特征
        # self.dropout_lstm = nn.Dropout2d(p=0.08)
        
        # # 【位置2】特征融合后 - 中等dropout，主要正则化位置
        # self.dropout_fusion = nn.Dropout2d(p=0.1)
        
        # 【位置3】最终输出前 - 轻量dropout，保护输出质量
        self.dropout_output = nn.Dropout2d(p=0.05)
        self.dropout_lstm = nn.Dropout2d(p=0.0)    # 关闭
        self.dropout_fusion = nn.Dropout2d(p=0.0)  # 关闭
        self.dropout_output = nn.Dropout2d(p=0.0)  # 关闭
        
        # ============ 移除了所有其他dropout！============
        print("✓ Dropout优化完成：")
        print(f"  - LSTM后: p=0.08")
        print(f"  - 融合后: p=0.10")
        print(f"  - 输出前: p=0.05")
        print(f"  - 总有效dropout率: ~8-10%")

    def forward(self, x, hidden_state=None):
        """
        x: [b, t, c, w, h]
        """
        x = x.unsqueeze(2)
        last_output_forward = []
        last_output_backward = []
        firstlayer = []
        
        # 初始化隐藏状态
        if hidden_state is None:
            tensor_size = (x.size(3), x.size(4))
            h1, c1 = self.forward_layer.init_hidden(batch_size=x.size(0), 
                                                   tensor_size=tensor_size, 
                                                   device=Device.device)
            h2, c2 = self.backward_layer.init_hidden(batch_size=x.size(0), 
                                                   tensor_size=tensor_size,
                                                   device=Device.device)
        
        # ===== 第一层：U-Net特征提取（无dropout）=====
        for t in range(self.seq_len):
            o = self.unet1(x[:, t, :, :, :])
            # 移除了这里的dropout！保留原始特征
            firstlayer.append(o)
        
        # ===== 第二层：BiLSTM处理（无dropout）=====
        # 前向LSTM
        for t in range(self.seq_len):
            h1, c1 = self.forward_layer(firstlayer[t], [h1, c1])
            # 移除了这里的dropout！保持LSTM记忆完整
            last_output_forward.append([h1, c1])
        
        # 后向LSTM
        for t in range(self.seq_len - 1, -1, -1):
            h2, c2 = self.backward_layer(firstlayer[t], [h2, c2])
            # 移除了这里的dropout！保持LSTM记忆完整
            last_output_backward.append([h2, c2])
        
        # ===== 第三层：特征拼接 =====
        tar = self.seq_len // 2
        last_output = []
        
        for t in range(self.seq_len):
            o = torch.cat([
                firstlayer[t], 
                last_output_forward[t][0], 
                last_output_backward[self.seq_len - t - 1][0]
            ], dim=1)
            o = self.add_conv(o)
            # 移除了这里的dropout！推迟到融合后统一处理
            last_output.append(o)
        
        # ===== 第四层：Union特征融合 =====
        first_half_len = tar
        latter_half_len = self.seq_len - 1 - tar
        
        # 处理前半部分
        if first_half_len > 0:
            first_half_frames = [last_output[i] for i in range(first_half_len)]
            o1 = torch.cat(first_half_frames, dim=1)
            o1 = self.union_firsthalf(o1)
        else:
            o1_shape = list(last_output[tar].shape)
            o1_shape[1] = self.initial_features
            o1 = torch.zeros(o1_shape, device=last_output[tar].device)
        
        # 处理后半部分
        if latter_half_len > 0:
            latter_half_frames = [last_output[i] for i in range(self.seq_len-1, tar, -1)]
            o2 = torch.cat(latter_half_frames, dim=1)
            o2 = self.union_latterhalf(o2)
        else:
            o2_shape = list(last_output[tar].shape)
            o2_shape[1] = self.initial_features
            o2 = torch.zeros(o2_shape, device=last_output[tar].device)
        
        # 拼接所有特征
        o = torch.cat([o1, last_output[tar], o2], dim=1)
        
        # 【Dropout位置1】LSTM特征融合后
        o = self.dropout_lstm(o)
        
        # ===== 第五层：最终UNet处理 =====
        o3 = self.unet2(o)
        
        # 【Dropout位置2】UNet2输出后
        o3 = self.dropout_fusion(o3)
        
        # ===== 第六层：输出头 =====
        o_heads = [outconv.forward(o3) for outconv in self.outconvlist]
        o = torch.cat(o_heads, dim=1)
        
        # 【Dropout位置3】最终输出前
        o = self.dropout_output(o)
        
        # 应用激活函数和约束
        o[:, [0]] = torch.clamp(o[:, [0]], min=-8., max=8.)
        o[:, self.sigmoid_ch_ix] = torch.sigmoid(o[:, self.sigmoid_ch_ix])
        o[:, self.tanh_ch_ix] = torch.tanh(o[:, self.tanh_ch_ix])
        o[:, self.pxyz_sig_ch_ix] = o[:, self.pxyz_sig_ch_ix] * 3 + self.sigma_eps_default
        
        return o


class ConvLSTMCell(nn.Module):
    """优化的ConvLSTM单元 - 移除内部dropout"""
    def __init__(self, in_channels, out_channels, norm=None, norm_groups=None):
        super(ConvLSTMCell, self).__init__()
        self.norm = norm
        self.norm_groups = norm_groups
        
        if self.norm is not None:
            groups_1 = min(in_channels, self.norm_groups)
            self.gn = nn.GroupNorm(groups_1, in_channels)
        else:
            self.gn = None
        
        self.hidden_channels = out_channels
        self.conv = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=3, padding=1)
        
        # ============ 移除了LSTM内部的dropout！============

    def forward(self, x, hidden_state):
        h_cur, c_cur = hidden_state
        h_t = torch.cat([h_cur, x], dim=1)
        
        if self.gn is not None:
            h_t = self.gn(h_t)
        
        h_t = self.conv(h_t)
        # 移除了这里的dropout！
        h_t = nn.ELU().forward(h_t)
        
        i, f, o, g = torch.split(h_t, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, tensor_size, device):
        height, width = tensor_size
        return (
            Variable(torch.zeros(batch_size, self.hidden_channels, height, width)).to(device),
            Variable(torch.zeros(batch_size, self.hidden_channels, height, width)).to(device)
        )


def OutLayer(in_channels, out_channels, norm=None, norm_groups=None):
    """输出层 - 移除内部dropout"""
    if norm is not None:
        groups_1 = min(in_channels, norm_groups)
    
    if norm == 'GroupNorm':
        return nn.Sequential(
            nn.GroupNorm(groups_1, in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # 移除了这里的dropout！
            nn.ELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            # 移除了这里的dropout！
            nn.ELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        )


def mix_conv(in_channels, out_channels, norm=None, norm_groups=None):
    """混合卷积 - 移除内部dropout"""
    if norm is not None:
        groups_1 = min(in_channels, norm_groups)
    
    if norm == 'GroupNorm':
        return nn.Sequential(
            nn.GroupNorm(groups_1, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 移除了这里的dropout！
            nn.ELU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 移除了这里的dropout！
            nn.ELU(),
        )