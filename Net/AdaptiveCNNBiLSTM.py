# Net/AdaptiveCNNBiLSTM.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import Net.Unet as Unet

class AdaptiveCNNBiLSTM(nn.Module):
    """
    针对SMLM优化的自适应Dropout策略
    """
    ch_out = 11
    out_channels_heads = (1, 4, 4, 1)
    sigmoid_ch_ix = [0, 1, 5, 6, 7, 8, 9]
    tanh_ch_ix = [2, 3, 4]
    
    def __init__(self, in_channels=1, out_channels=11, depth=2, seq_len=5, 
                 initial_features=48, pad_convs=False, sigma_eps_default=0.005,
                 dropout_config=None, **kwargs):
        super(AdaptiveCNNBiLSTM, self).__init__()
        
        self.sigma_eps_default = sigma_eps_default
        self.initial_features = initial_features
        self.seq_len = seq_len
        
        # Dropout配置
        if dropout_config is None:
            dropout_config = {
                'spatial_dropout': True,
                'bottleneck_p': 0.3,
                'lstm_p': 0.25,
                'output_p': 0.15,
                'adaptive': True
            }
        self.dropout_config = dropout_config
        
        # 核心网络组件
        self.forward_layer = ConvLSTMCell(2 * initial_features, initial_features)
        self.backward_layer = ConvLSTMCell(2 * initial_features, initial_features)
        
        self.unet1 = Unet.Unet(in_channels, initial_features, depth=depth, pad_convs=pad_convs)
        self.unet2 = Unet.Unet(3 * initial_features, initial_features, depth=depth, pad_convs=pad_convs)
        
        first_half_len = seq_len // 2
        latter_half_len = seq_len - 1 - first_half_len
        
        if first_half_len > 0:
            self.union_firsthalf = Unet.Unet(first_half_len * initial_features, initial_features, 
                                            depth=depth, pad_convs=pad_convs)
        else:
            self.union_firsthalf = None
            
        if latter_half_len > 0:
            self.union_latterhalf = Unet.Unet(latter_half_len * initial_features, initial_features, 
                                             depth=depth, pad_convs=pad_convs)
        else:
            self.union_latterhalf = None
        
        self.add_conv = nn.Sequential(
            nn.Conv2d(3 * initial_features, initial_features, 3, padding=1),
            nn.ELU()
        )
        
        # 输出头
        self.outconvlist = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(initial_features, initial_features, 3, padding=1),
                nn.ELU(),
                nn.Conv2d(initial_features, ch, 1)
            ) for ch in self.out_channels_heads
        ])
        
        # Dropout层
        if dropout_config['spatial_dropout']:
            self.dropout_bottleneck = SpatialDropout2d(p=dropout_config['bottleneck_p'])
            self.dropout_lstm = SpatialDropout2d(p=dropout_config['lstm_p'])
            self.dropout_output = SpatialDropout2d(p=dropout_config['output_p'])
        else:
            self.dropout_bottleneck = nn.Dropout2d(p=dropout_config['bottleneck_p'])
            self.dropout_lstm = nn.Dropout2d(p=dropout_config['lstm_p'])
            self.dropout_output = nn.Dropout2d(p=dropout_config['output_p'])
        
        if dropout_config['adaptive']:
            self.adaptive_dropout = AdaptiveDropout(initial_features)
        else:
            self.adaptive_dropout = None
        
        print("✓ AdaptiveCNNBiLSTM初始化完成")
        print(f"  - 瓶颈层dropout: p={dropout_config['bottleneck_p']}")
        print(f"  - LSTM dropout: p={dropout_config['lstm_p']}")
        print(f"  - 输出dropout: p={dropout_config['output_p']}")
        print(f"  - 空间dropout: {dropout_config['spatial_dropout']}")
        print(f"  - 自适应dropout: {dropout_config['adaptive']}")

    def forward(self, x, hidden_state=None):
        device = x.device
        x = x.unsqueeze(2)
        
        if hidden_state is None:
            h1, c1 = self.forward_layer.init_hidden(x.size(0), (x.size(3), x.size(4)), device)
            h2, c2 = self.backward_layer.init_hidden(x.size(0), (x.size(3), x.size(4)), device)
        
        # UNet特征提取
        firstlayer = [self.unet1(x[:, t, :, :, :]) for t in range(self.seq_len)]
        
        # BiLSTM处理
        forward_out = []
        for t in range(self.seq_len):
            h1, c1 = self.forward_layer(firstlayer[t], [h1, c1])
            forward_out.append([h1, c1])
        
        backward_out = []
        for t in range(self.seq_len - 1, -1, -1):
            h2, c2 = self.backward_layer(firstlayer[t], [h2, c2])
            backward_out.append([h2, c2])
        backward_out = backward_out[::-1]
        
        # 特征融合
        tar = self.seq_len // 2
        combined = []
        for t in range(self.seq_len):
            o = torch.cat([firstlayer[t], forward_out[t][0], backward_out[t][0]], dim=1)
            o = self.add_conv(o)
            if t == tar:  # 只在中心帧应用
                o = self.dropout_lstm(o)
            combined.append(o)
        
        # Union处理
        first_half_len = tar
        latter_half_len = self.seq_len - 1 - tar
        
        if first_half_len > 0 and self.union_firsthalf is not None:
            o1 = self.union_firsthalf(torch.cat(combined[:first_half_len], dim=1))
        else:
            o1 = torch.zeros(combined[tar].shape, device=device)
        
        if latter_half_len > 0 and self.union_latterhalf is not None:
            o2 = self.union_latterhalf(torch.cat(combined[self.seq_len-1:tar:-1], dim=1))
        else:
            o2 = torch.zeros(combined[tar].shape, device=device)
        
        # 瓶颈层dropout
        o = torch.cat([o1, combined[tar], o2], dim=1)
        o = self.dropout_bottleneck(o)
        
        # 最终UNet处理
        o = self.unet2(o)
        
        # 自适应dropout
        if self.adaptive_dropout is not None:
            o = self.adaptive_dropout(o)
        
        # 输出层
        o_heads = [outconv(o) for outconv in self.outconvlist]
        o = torch.cat(o_heads, dim=1)
        
        # 输出dropout
        o = self.dropout_output(o)
        
        # 激活函数
        o[:, [0]] = torch.clamp(o[:, [0]], min=-8., max=8.)
        o[:, self.sigmoid_ch_ix] = torch.sigmoid(o[:, self.sigmoid_ch_ix])
        o[:, self.tanh_ch_ix] = torch.tanh(o[:, self.tanh_ch_ix])
        o[:, slice(5, 9)] = o[:, slice(5, 9)] * 3 + self.sigma_eps_default
        
        return o


class SpatialDropout2d(nn.Module):
    """空间Dropout"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        mask_shape = (x.size(0), x.size(1), 1, 1)
        mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p)).to(x.device)
        return x * mask / (1 - self.p)


class AdaptiveDropout(nn.Module):
    """自适应Dropout"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.dropout_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.base_p = 0.2
        self.max_p = 0.4
        
    def forward(self, x):
        if not self.training:
            return x
            
        dropout_rate = self.dropout_predictor(x)
        dropout_rate = self.base_p + (self.max_p - self.base_p) * dropout_rate
        
        random_tensor = torch.rand_like(x[:, :1, :, :])
        dropout_mask = (random_tensor > dropout_rate).float()
        dropout_mask = dropout_mask.expand_as(x)
        
        return x * dropout_mask / (1 - dropout_rate.mean())


class ConvLSTMCell(nn.Module):
    """ConvLSTM单元"""
    def __init__(self, in_channels, out_channels):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = out_channels
        self.conv = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=3, padding=1)

    def forward(self, x, hidden_state):
        h_cur, c_cur = hidden_state
        h_t = torch.cat([h_cur, x], dim=1)
        h_t = F.elu(self.conv(h_t))
        
        i, f, o, g = torch.split(h_t, self.hidden_channels, dim=1)
        c_next = torch.sigmoid(f) * c_cur + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, tensor_size, device):
        h, w = tensor_size
        return (
            torch.zeros(batch_size, self.hidden_channels, h, w).to(device),
            torch.zeros(batch_size, self.hidden_channels, h, w).to(device)
        )