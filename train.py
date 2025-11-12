import json
import os
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F    # ← 确保有这一行
from Net.AdaptiveCNNBiLSTM import AdaptiveCNNBiLSTM

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import decode
import decode.utils
import decode.neuralfitter.train.live_engine
import torch

path = Path('')
import copy
from pathlib import Path
import decode.evaluation
import decode.neuralfitter
import decode.neuralfitter.coord_transform
import decode.neuralfitter.utils
import decode.simulation
from decode.neuralfitter.train.random_simulation import setup_random_simulation
from decode.neuralfitter.utils import log_train_val_progress
from decode.utils.checkpoint import CheckPoint

def convert_to_serializable(obj):
    """将 Tensor 等对象转换为可序列化的格式"""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def setup_trainer(simulator_train, simulator_test, logger, model_out, ckpt_path, device, param):
    """Set model, optimiser, loss and schedulers"""
    models_available = {
        'SigmaMUNet': decode.neuralfitter.models.SigmaMUNet,
        'DoubleMUnet': decode.neuralfitter.models.model_param.DoubleMUnet,
        'SimpleSMLMNet': decode.neuralfitter.models.model_param.SimpleSMLMNet,
    }

    model = models_available[param.HyperParameter.architecture]
    # print("ch_in:%d" % param.HyperParameter.channels_in)
    model = model.parse(param)

    model_ls = decode.utils.model_io.LoadSaveModel(model,
                                                   output_file=model_out)

    model = model_ls.load_init()
    model = model.to(torch.device(device))

    # Small collection of optimisers
    """Checkpointing"""
    checkpoint = CheckPoint(path=ckpt_path)

    """Setup gradient modification"""
    grad_mod = param.HyperParameter.grad_mod

    """Log the model"""
    try:
        dummy = torch.rand((2, param.HyperParameter.channels_in,
                            *param.Simulation.img_size), requires_grad=False).to(
            torch.device(device))
        logger.add_graph(model, dummy)

    except:
        print("Did not log graph.")
        # raise RuntimeError("Your dummy input is wrong. Please update it.")

    """Transform input data, compute weight mask and target data"""
    frame_proc = decode.neuralfitter.scale_transform.AmplitudeRescale.parse(param)
    bg_frame_proc = None

    if param.HyperParameter.emitter_label_photon_min is not None:
        em_filter = decode.neuralfitter.em_filter.PhotonFilter(
            param.HyperParameter.emitter_label_photon_min)
    else:
        em_filter = decode.neuralfitter.em_filter.NoEmitterFilter()

    tar_frame_ix_train = (0, 0)
    tar_frame_ix_test = (0, param.TestSet.test_size)

    """Setup Target generator consisting possibly multiple steps in a transformation sequence."""
    tar_gen = decode.neuralfitter.utils.processing.TransformSequence(
        [
            decode.neuralfitter.target_generator.ParameterListTarget(
                n_max=param.HyperParameter.max_number_targets,
                xextent=param.Simulation.psf_extent[0],
                yextent=param.Simulation.psf_extent[1],
                ix_low=tar_frame_ix_train[0],
                ix_high=tar_frame_ix_train[1],
                squeeze_batch_dim=True),

            decode.neuralfitter.target_generator.DisableAttributes.parse(param),

            decode.neuralfitter.scale_transform.ParameterListRescale(
                phot_max=param.Scaling.phot_max,
                z_max=param.Scaling.z_max,
                bg_max=param.Scaling.bg_max)
        ])

    # setup target for test set in similar fashion, however test-set is static.
    tar_gen_test = copy.deepcopy(tar_gen)
    tar_gen_test.com[0].ix_low = tar_frame_ix_test[0]
    tar_gen_test.com[0].ix_high = tar_frame_ix_test[1]
    tar_gen_test.com[0].squeeze_batch_dim = False
    tar_gen_test.com[0].sanity_check()

    if param.Simulation.mode == 'acquisition':
        train_ds = decode.neuralfitter.dataset.SMLMLiveDataset(
            simulator=simulator_train,
            em_proc=em_filter,
            frame_proc=frame_proc,
            bg_frame_proc=bg_frame_proc,
            tar_gen=tar_gen, weight_gen=None,
            frame_window=param.HyperParameter.channels_in,
            pad=None, return_em=True)

        train_ds.sample(True)

    elif param.Simulation.mode == 'samples':
        train_ds = decode.neuralfitter.dataset.SMLMLiveSampleDataset(
            simulator=simulator_train,
            em_proc=em_filter,
            frame_proc=frame_proc,
            bg_frame_proc=bg_frame_proc,
            tar_gen=tar_gen,
            weight_gen=None,
            frame_window=param.HyperParameter.channels_in,
            return_em=False,
            ds_len=param.HyperParameter.pseudo_ds_size)

    test_ds = decode.neuralfitter.dataset.SMLMAPrioriDataset(
        simulator=simulator_test,
        em_proc=em_filter,
        frame_proc=frame_proc,
        bg_frame_proc=bg_frame_proc,
        tar_gen=tar_gen_test, weight_gen=None,
        frame_window=param.HyperParameter.channels_in,
        pad=None, return_em=True)

    test_ds.sample(True)

    """Set up post processor"""
    if param.PostProcessing is None:
        post_processor = decode.neuralfitter.post_processing.NoPostProcessing(xy_unit='px',
                                                                              px_size=param.Camera.px_size)

    elif param.PostProcessing == 'LookUp':
        post_processor = decode.neuralfitter.utils.processing.TransformSequence([

            decode.neuralfitter.scale_transform.InverseParamListRescale(
                phot_max=param.Scaling.phot_max,
                z_max=param.Scaling.z_max,
                bg_max=param.Scaling.bg_max),

            decode.neuralfitter.coord_transform.Offset2Coordinate.parse(param),

            decode.neuralfitter.post_processing.LookUpPostProcessing(
                raw_th=param.PostProcessingParam.raw_th,
                pphotxyzbg_mapping=[0, 1, 2, 3, 4, 9],
                xy_unit='px',
                px_size=param.Camera.px_size)
        ])

    elif param.PostProcessing in ('SpatialIntegration', 'NMS'):  # NMS as legacy support
        post_processor = decode.neuralfitter.utils.processing.TransformSequence([

            decode.neuralfitter.scale_transform.InverseParamListRescale(
                phot_max=param.Scaling.phot_max,
                z_max=param.Scaling.z_max,
                bg_max=param.Scaling.bg_max),

            decode.neuralfitter.coord_transform.Offset2Coordinate.parse(param),

            decode.neuralfitter.post_processing.SpatialIntegration(
                raw_th=param.PostProcessingParam.raw_th,
                xy_unit='px',
                px_size=param.Camera.px_size)
        ])

    else:
        raise NotImplementedError

    """Evaluation Specification"""
    matcher = decode.evaluation.match_emittersets.GreedyHungarianMatching.parse(param)

    return train_ds, test_ds, model, model_ls, grad_mod, post_processor, matcher, checkpoint


from typing import Union, Tuple
import numpy as np
from torch import distributions
from decode.simulation import psf_kernel


class LossFunc():
    def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple, device: Union[str, torch.device], psf):
        super().__init__()
        self._psf_loss = torch.nn.MSELoss(reduction='none')
        self._offset2coord = psf_kernel.DeltaPSF(xextent=xextent, yextent=yextent, img_shape=img_shape)
        self.device = device
        self._psf_img_gen = decode.simulation.Simulation(psf=psf)
        self.xextent = xextent
        self.yextent = xextent
        self.img_shape = img_shape

    def log(self, loss_val):
        return loss_val.mean().item(), {'gmm': loss_val[:, 0].mean().item(),
                                        'p': loss_val[:, 1].mean().item(),
                                        'bg': loss_val[:, 2].mean().item(),
                                        # 'img': loss_val[:, -1].mean().item()
                                        }

    def CELoss(self, P, em_tar, tar_mask) -> torch.Tensor:
        S = torch.zeros([len(em_tar), param.Simulation.img_size[0], param.Simulation.img_size[1]]).to(self.device)
        if tar_mask.sum():
            for i, tar in enumerate(em_tar):
                tar = tar.xyz_px.to(self.device)
                tar = torch.round(tar[:, [0, 1]], decimals=0)
                tar = (tar.transpose(0, 1)).int()
                tar = (tar[0], tar[1])
                S[i].index_put(tar, torch.ones(tar[0].size()).to(self.device))
        loss = 0
        loss += -(S * torch.log(P) + (1 - S) * torch.log(1 - P))
        loss = loss.sum(-1).sum(-1)
        return loss

    def Loss_Count(self, P, tar_mask):
        loss = 0
        prob_mean = P.sum(-1).sum(-1)
        prob_var = (P - P ** 2).sum(-1).sum(-1)
        loss += 1 / 2 * ((tar_mask.sum(-1) - prob_mean) ** 2) / prob_var + 1 / 2 * torch.log(2 * np.pi * prob_var)
        num_emitters = torch.clamp(tar_mask.sum(-1), min=1.0)
        loss = loss / num_emitters  # ← 除以emitter数量normalize
        # loss = loss * tar_mask.sum(-1)
        return loss

    def Loss_Loc(self, P, pxyz_mu, pxyz_sig, pxyz_tar, mask):
        batch_size = P.size(0)
        prob_normed = P / (P.sum(-1).sum(-1)[:, None, None])

        p_inds = tuple((P + 1).nonzero().transpose(1, 0))

        pxyz_mu = pxyz_mu[p_inds[0], :, p_inds[1], p_inds[2]]
        self._offset2coord._bin_ctr_x = self._offset2coord._bin_ctr_x.to(pxyz_mu.device)
        self._offset2coord._bin_ctr_y = self._offset2coord._bin_ctr_y.to(pxyz_mu.device)
        pxyz_mu[:, 1] = pxyz_mu[:, 1] + self._offset2coord.bin_ctr_x[p_inds[1]]
        pxyz_mu[:, 2] = pxyz_mu[:, 2] + self._offset2coord.bin_ctr_y[p_inds[2]]

        pxyz_mu = pxyz_mu.reshape(batch_size, 1, -1, 4)
        pxyz_sig = pxyz_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(batch_size, 1, -1, 4)
        PXYZ = pxyz_tar.reshape(batch_size, -1, 1, 4).repeat_interleave(self.img_shape[0] * self.img_shape[1], 2)

        numerator = -1 / 2 * ((PXYZ - pxyz_mu) ** 2)
        denominator = (pxyz_sig ** 2)
        log_p_gauss_4d = (numerator / denominator).sum(3) - 1 / 2 * (torch.log(2 * np.pi * denominator[:, :, :, 0]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 1]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 2]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 3]))

        gauss_coef = prob_normed.reshape(batch_size, 1, self.img_shape[0] * self.img_shape[1])
        gauss_coef_logits = torch.log(gauss_coef)
        gauss_coef_logmax = torch.log_softmax(gauss_coef_logits, dim=2)
        gmm_log = torch.logsumexp(log_p_gauss_4d + gauss_coef_logmax, dim=2)

        result = -(gmm_log * mask).sum(-1)
    
    # ============ 添加normalize ============
        num_emitters = torch.clamp(mask.sum(-1), min=1.0)
        result = result / num_emitters
    # ========================================
    
        return result

    def Loss_psf(self, psf_img, psf_gt):
        if psf_gt.dim() == 4:
            psf_gt = psf_gt[:, 2]
        loss = self._psf_loss(psf_img, psf_gt)
        loss = loss.sum(-1).sum(-1)
        return loss

    def get_psf_gt(self, em_tar):
        for i, em in enumerate(em_tar):
            tmp, _ = self._psf_img_gen.forward(em)
            tmp = tmp.to(self.device)
            if i == 0:
                psf_gt = tmp
            else:
                psf_gt = torch.cat((psf_gt, tmp), dim=0)
        return psf_gt

    def norm(self, nobg):
        ret = []
        for tmp in nobg:
            tmp = (tmp - torch.min(tmp)) / (torch.max(tmp) - torch.min(tmp))
            ret.append(tmp)
        return torch.stack(ret, dim=0)

    def final_loss(self, output, target, nobg, em_tar=None):
        tar_param, tar_mask, tar_bg = target
        P = output[:, 0]
        pxyz_mu = output[:, 1:5]
        pxyz_sig = output[:, 5:9]
        bg_img = output[:, 9]
        # psf_img = output[:, -1]
        # nobg = self.norm(nobg)
        # psf_gt = self.get_psf_gt(em_tar)

        loss = torch.stack((self.Loss_Loc(P, pxyz_mu, pxyz_sig, tar_param, tar_mask), self.Loss_Count(P, tar_mask)),
                           dim=1)
        loss = torch.cat((loss, self._psf_loss(bg_img, tar_bg).sum(-1).sum(-1).unsqueeze(1)), dim=1)
        # loss = torch.cat((loss,self.Loss_psf(psf_img, nobg).unsqueeze(1)*0.0001),dim=1)
        return loss
# class FixedLossFunc():
#     """修复数值稳定性的损失函数"""
#     def __init__(self, xextent: tuple, yextent: tuple, img_shape: tuple, 
#                  device, psf):
#         super().__init__()
#         self._psf_loss = torch.nn.MSELoss(reduction='none')
#         self._offset2coord = psf_kernel.DeltaPSF(xextent=xextent, yextent=yextent, 
#                                                   img_shape=img_shape)
#         self.device = device
#         self._psf_img_gen = decode.simulation.Simulation(psf=psf)
#         self.xextent = xextent
#         self.yextent = yextent
#         self.img_shape = img_shape

#     def log(self, loss_val):
#         return loss_val.mean().item(), {
#             'gmm': loss_val[:, 0].mean().item(),
#             'p': loss_val[:, 1].mean().item(),
#             'bg': loss_val[:, 2].mean().item(),
#         }

#     def Loss_Count(self, P, tar_mask):
#         prob_mean = P.sum(-1).sum(-1)
#         prob_var = (P - P ** 2).sum(-1).sum(-1)
        
#         # 【修复】添加数值稳定性
#         prob_var = torch.clamp(prob_var, min=1e-6)
        
#         loss = 1 / 2 * ((tar_mask.sum(-1) - prob_mean) ** 2) / prob_var + \
#                1 / 2 * torch.log(2 * np.pi * prob_var)
#         loss = loss * torch.clamp(tar_mask.sum(-1), min=1.0)  # 避免乘以0
#         return loss

#     def Loss_Loc(self, P, pxyz_mu, pxyz_sig, pxyz_tar, mask):
#         batch_size = P.size(0)
        
#         # 【修复】添加epsilon避免除以0
#         prob_normed = P / (P.sum(-1).sum(-1)[:, None, None] + 1e-10)

#         p_inds = tuple((P + 1).nonzero(as_tuple=False).transpose(1, 0))
        
#         # 如果没有有效位置，返回0
#         if len(p_inds[0]) == 0:
#             return torch.zeros(batch_size).to(self.device)

#         pxyz_mu = pxyz_mu[p_inds[0], :, p_inds[1], p_inds[2]]
#         self._offset2coord._bin_ctr_x = self._offset2coord._bin_ctr_x.to(pxyz_mu.device)
#         self._offset2coord._bin_ctr_y = self._offset2coord._bin_ctr_y.to(pxyz_mu.device)
#         pxyz_mu[:, 1] = pxyz_mu[:, 1] + self._offset2coord.bin_ctr_x[p_inds[1]]
#         pxyz_mu[:, 2] = pxyz_mu[:, 2] + self._offset2coord.bin_ctr_y[p_inds[2]]

#         pxyz_mu = pxyz_mu.reshape(batch_size, 1, -1, 4)
#         pxyz_sig = pxyz_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(batch_size, 1, -1, 4)
#         PXYZ = pxyz_tar.reshape(batch_size, -1, 1, 4).repeat_interleave(
#             self.img_shape[0] * self.img_shape[1], 2)

#         numerator = -1 / 2 * ((PXYZ - pxyz_mu) ** 2)
        
#         # 【修复】限制sigma的范围，避免数值问题
#         denominator = torch.clamp(pxyz_sig ** 2, min=1e-6, max=100.0)
        
#         log_p_gauss_4d = (numerator / denominator).sum(3) - 1 / 2 * (
#             torch.log(2 * np.pi * denominator[:, :, :, 0]) +
#             torch.log(2 * np.pi * denominator[:, :, :, 1]) +
#             torch.log(2 * np.pi * denominator[:, :, :, 2]) +
#             torch.log(2 * np.pi * denominator[:, :, :, 3])
#         )

#         gauss_coef = prob_normed.reshape(batch_size, 1, self.img_shape[0] * self.img_shape[1])
#         gauss_coef_logits = torch.log(gauss_coef + 1e-10)
#         gauss_coef_logmax = torch.log_softmax(gauss_coef_logits, dim=2)
#         gmm_log = torch.logsumexp(log_p_gauss_4d + gauss_coef_logmax, dim=2)

#         return -(gmm_log * mask).sum(-1)

#     def final_loss(self, output, target, nobg, em_tar=None):
#         tar_param, tar_mask, tar_bg = target
#         P = output[:, 0]
#         pxyz_mu = output[:, 1:5]
#         pxyz_sig = output[:, 5:9]
#         bg_img = output[:, 9]
        
#         loss_loc = self.Loss_Loc(P, pxyz_mu, pxyz_sig, tar_param, tar_mask)
#         loss_count = self.Loss_Count(P, tar_mask)
#         loss_bg = self._psf_loss(bg_img, tar_bg).sum(-1).sum(-1)
        
#         # 【修复】检查并限制损失值
#         loss_loc = torch.clamp(loss_loc, max=10000.0)
#         loss_count = torch.clamp(loss_count, max=10000.0)
#         loss_bg = torch.clamp(loss_bg, max=1000.0)
        
#         loss = torch.stack((loss_loc, loss_count), dim=1)
#         loss = torch.cat((loss, loss_bg.unsqueeze(1)), dim=1)
        
#         # 【调试】打印损失信息（可选）
#         if torch.isnan(loss).any() or (loss > 50000).any():
#             print(f"WARNING: Loc={loss_loc.mean():.1f}, Count={loss_count.mean():.1f}, BG={loss_bg.mean():.1f}")
        
#         return loss


import torch
import time
from typing import Union

from tqdm import tqdm
from collections import namedtuple

from decode.neuralfitter.utils import log_train_val_progress
from decode.evaluation.utils import MetricMeter
class OverfittingMonitor:
    """科学的过拟合监控器"""
    def __init__(self, patience=5, min_epochs=10):
        self.train_losses = []
        self.val_losses = []
        self.relative_gaps = []
        self.patience = patience
        self.min_epochs = min_epochs
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
    def update(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # 计算相对间隙
        if train_loss > 1e-6:
            relative_gap = abs(val_loss - train_loss) / train_loss
        else:
            relative_gap = 0.0
        self.relative_gaps.append(relative_gap)
        
        # Early stopping 检测
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            
    def get_status(self, epoch):
        """多维度评估训练状态"""
        # 👇 修改这里：即使是第一个epoch也要返回metrics
        if len(self.train_losses) < 1:
            # 如果完全没有数据，返回空状态
            return "⏳ 初始化", {
                'train_loss': 0.0,
                'val_loss': 0.0,
                'absolute_gap': 0.0,
                'relative_gap': 0.0,
                'val_trend': 0.0,
                'gap_trend': 0.0,
            }
        
        train_loss = self.train_losses[-1]
        val_loss = self.val_losses[-1]
        relative_gap = self.relative_gaps[-1]
        
        # 计算统计指标
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'absolute_gap': abs(val_loss - train_loss),
            'relative_gap': relative_gap,
            'val_trend': self._compute_trend(self.val_losses[-5:]) if len(self.val_losses) >= 2 else 0.0,
            'gap_trend': self._compute_trend(self.relative_gaps[-5:]) if len(self.relative_gaps) >= 2 else 0.0,
        }
        
        # 👇 修改判断条件
        if epoch < self.min_epochs or len(self.train_losses) < 2:
            status = "⏳ 训练中 (样本不足)"
        else:
            # 多维度判断
            status = self._diagnose(train_loss, val_loss, relative_gap, epoch, metrics)
        
        return status, metrics
    
    def _compute_trend(self, values):
        """计算趋势: 正值=上升, 负值=下降"""
        if len(values) < 2:
            return 0.0
        # 简单线性回归斜率
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        slope = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n)) / \
                (sum((x[i] - x_mean) ** 2 for i in range(n)) + 1e-8)
        return slope
    
    def _diagnose(self, train_loss, val_loss, relative_gap, epoch, metrics):
        """综合诊断"""
        # 1. 欠拟合检测
        if train_loss > 0.5:  # 根据你的任务调整这个阈值
            return "❌ 欠拟合 (训练损失过高)"
        
        if epoch < self.min_epochs:
            return "⏳ 训练中 (样本不足)"
        
        # 2. 过拟合检测 (多条件)
        overfitting_score = 0
        
        # 条件1: 相对间隙过大
        if relative_gap > 0.25:
            overfitting_score += 3
        elif relative_gap > 0.15:
            overfitting_score += 2
        elif relative_gap > 0.10:
            overfitting_score += 1
            
        # 条件2: 验证损失上升趋势
        if metrics['val_trend'] > 0.001:
            overfitting_score += 2
            
        # 条件3: 训练-验证损失背离
        if len(self.train_losses) >= 3:
            train_trend = self._compute_trend(self.train_losses[-3:])
            if train_trend < -0.001 and metrics['val_trend'] > 0.001:
                overfitting_score += 2
        
        # 条件4: Early stopping 信号
        if self.epochs_no_improve >= self.patience:
            overfitting_score += 2
            
        # 综合判断
        if overfitting_score >= 6:
            return "❌ 严重过拟合"
        elif overfitting_score >= 4:
            return "⚠️  过拟合警告"
        elif overfitting_score >= 2:
            return "⚠️  轻微过拟合"
        elif relative_gap < 0.08:
            return "✅ 优秀"
        elif relative_gap < 0.12:
            return "✅ 良好"
        else:
            return "⚠️  注意监控"
    
    def should_stop(self, epoch):
        """是否应该提前停止"""
        if epoch < self.min_epochs:
            return False
        return self.epochs_no_improve >= self.patience
    
    def get_summary(self):
        """生成训练总结"""
        if len(self.train_losses) <= 1:
            return {}
        
        valid_train = [l for l in self.train_losses[1:] if l > 0]
        valid_val = self.val_losses[1:]
        valid_gaps = [g for g in self.relative_gaps[1:] if g >= 0]
        
        if len(valid_train) == 0:
            return {}
        
        return {
            'avg_train_loss': sum(valid_train) / len(valid_train),
            'avg_val_loss': sum(valid_val) / len(valid_val),
            'avg_relative_gap': sum(valid_gaps) / len(valid_gaps),
            'min_val_loss': min(valid_val),
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_relative_gap': self.relative_gaps[-1],
            'best_epoch': valid_val.index(min(valid_val)) + 1,
            'convergence_score': self._compute_convergence_score()
        }
    
    def _compute_convergence_score(self):
        """收敛质量评分 (0-100)"""
        if len(self.relative_gaps) < 5:
            return 0
        
        score = 100
        
        # 扣分项
        final_gap = self.relative_gaps[-1]
        if final_gap > 0.25:
            score -= 40
        elif final_gap > 0.15:
            score -= 20
        elif final_gap > 0.10:
            score -= 10
            
        # 趋势扣分
        gap_trend = self._compute_trend(self.relative_gaps[-10:])
        if gap_trend > 0.01:
            score -= 20
            
        # 波动扣分
        recent_gaps = self.relative_gaps[-10:]
        if len(recent_gaps) > 0:
            gap_std = (sum((g - sum(recent_gaps)/len(recent_gaps))**2 for g in recent_gaps) / len(recent_gaps)) ** 0.5
            if gap_std > 0.05:
                score -= 15
        
        return max(0, score)


def train(model, optimizer, loss, dataloader, grad_rescale, grad_mod, epoch, device, logger) -> float:
    model.train()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.)
    t0 = time.time()
    loss_epoch = MetricMeter()

    for batch_num, (x, y_tar, weight, em_tar, nobg) in enumerate(tqdm_enum):
        t_data = time.time() - t0
        x, y_tar, weight, nobg = ship_device([x, y_tar, weight, nobg], device)
        y_out = model(x)

        # 👇 保存原始损失用于记录
        loss_val_original = loss.final_loss(y_out, y_tar, nobg, em_tar)
        loss_val = loss_val_original  # 用于反向传播的损失

        if grad_rescale:
            weight, _, _ = model.rescale_last_layer_grad(loss_val, optimizer)
            loss_val = loss_val * weight  # 只修改用于反向传播的损失

        optimizer.zero_grad()
        loss_val.mean().backward()

        if grad_mod:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.03, norm_type=2)

        optimizer.step()

        t_batch = time.time() - t0

        # 👇 使用原始损失计算指标，不是缩放后的
        loss_mean, loss_cmp = loss.log(loss_val_original)
        del loss_val, loss_val_original
        loss_epoch.update(loss_mean)
        
        tqdm_enum.set_description(f"E: {epoch} - t: {t_batch:.2} - t_dat: {t_data:.2} - L: {loss_mean:.3} \
                                  Lgmm: {loss_cmp['gmm']:.3}, Lp: {loss_cmp['p']:.3}, Lbg: {loss_cmp['bg']:.3}")
        t0 = time.time()

    log_train_val_progress.log_train(loss_p_batch=loss_epoch.vals, loss_mean=loss_epoch.mean, logger=logger, step=epoch)

    return loss_epoch.mean

from collections import namedtuple  # 确保namedtuple导入

from collections import namedtuple
_val_return = namedtuple("network_output", ["loss", "x", "y_out", "y_tar", "weight", "em_tar"])

from collections import namedtuple
_val_return = namedtuple("network_output", ["loss", "x", "y_out", "y_tar", "weight", "em_tar"])

def test_simple(model, loss, dataloader, epoch, device):
    """简化的测试函数 - 不使用MC Dropout"""
    model.eval()  # 关闭dropout
    
    x_ep, y_out_ep, y_tar_ep, weight_ep, em_tar_ep = [], [], [], [], []
    loss_cmp_ep = []
    
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.)
    
    with torch.no_grad():
        for batch_num, (x, y_tar, weight, em_tar, nobg) in enumerate(tqdm_enum):
            x, y_tar, weight, nobg = ship_device([x, y_tar, weight, nobg], device)
            
            # 直接预测
            y_out = model(x)
            
            # 计算损失
            loss_val = loss.final_loss(y_out, y_tar, nobg, em_tar)
            
            # 存储结果
            loss_cmp_ep.append(loss_val.detach().cpu())
            x_ep.append(x.cpu())
            y_out_ep.append(y_out.detach().cpu())
            
            if isinstance(y_tar, tuple):
                y_tar_cpu = tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in y_tar)
                y_tar_ep.append(y_tar_cpu)
            else:
                y_tar_ep.append(y_tar.cpu() if isinstance(y_tar, torch.Tensor) else y_tar)
            
            weight_ep.append(weight.cpu() if isinstance(weight, torch.Tensor) else weight)
            em_tar_ep.append(em_tar)
    
    loss_cmp_ep = torch.cat(loss_cmp_ep, 0)
    x_ep = torch.cat(x_ep, 0)
    y_out_ep = torch.cat(y_out_ep, 0)
    
    return loss_cmp_ep.mean(), _val_return(
        loss=loss_cmp_ep, x=x_ep, y_out=y_out_ep,
        y_tar=y_tar_ep, weight=weight_ep, em_tar=em_tar_ep
    )
def ship_device(x, device: Union[str, torch.device]):
    if x is None:
        return x

    elif isinstance(x, torch.Tensor):
        return x.to(device)

    elif isinstance(x, (tuple, list)):
        x = [ship_device(x_el, device) for x_el in x]  # a nice little recursion that worked at the first try
        return x

    elif device != 'cpu':
        raise NotImplementedError(f"Unsupported data type for shipping from host to CUDA device.")
    # ============ 在这里添加完整的模型定义 ============

class SimpleConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleConvLSTMCell, self).__init__()
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


class SimpleCNNBiLSTM(nn.Module):
    """简化版模型 - 完全关闭dropout用于调试"""
    ch_out = 11
    out_channels_heads = (1, 4, 4, 1)
    sigmoid_ch_ix = [0, 1, 5, 6, 7, 8, 9]
    tanh_ch_ix = [2, 3, 4]

    def __init__(self, in_channels=1, out_channels=11, depth=2, seq_len=5, 
                 initial_features=48, pad_convs=False, sigma_eps_default=0.005, **kwargs):
        super(SimpleCNNBiLSTM, self).__init__()
        
        self.sigma_eps_default = sigma_eps_default
        self.initial_features = initial_features
        self.seq_len = seq_len
        
        import Net.Unet as Unet
        
        # 核心网络
        self.forward_layer = SimpleConvLSTMCell(2 * initial_features, initial_features)
        self.backward_layer = SimpleConvLSTMCell(2 * initial_features, initial_features)
        
        self.unet1 = Unet.Unet(in_channels, initial_features, depth=depth, pad_convs=pad_convs)
        self.unet2 = Unet.Unet(3 * initial_features, initial_features, depth=depth, pad_convs=pad_convs)
        
        first_half_len = seq_len // 2
        latter_half_len = seq_len - 1 - first_half_len
        
        if first_half_len > 0:
            self.union_firsthalf = Unet.Unet(first_half_len * initial_features, initial_features, depth=depth, pad_convs=pad_convs)
        else:
            self.union_firsthalf = None
            
        if latter_half_len > 0:
            self.union_latterhalf = Unet.Unet(latter_half_len * initial_features, initial_features, depth=depth, pad_convs=pad_convs)
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
        
        print("✓ SimpleCNNBiLSTM初始化成功 - Dropout关闭")

    def forward(self, x, hidden_state=None):
        device = x.device
        x = x.unsqueeze(2)
        
        if hidden_state is None:
            h1, c1 = self.forward_layer.init_hidden(x.size(0), (x.size(3), x.size(4)), device)
            h2, c2 = self.backward_layer.init_hidden(x.size(0), (x.size(3), x.size(4)), device)
        
        # UNet特征提取
        firstlayer = [self.unet1(x[:, t, :, :, :]) for t in range(self.seq_len)]
        
        # BiLSTM
        forward_out = []
        for t in range(self.seq_len):
            h1, c1 = self.forward_layer(firstlayer[t], [h1, c1])
            forward_out.append([h1, c1])
        
        backward_out = []
        for t in range(self.seq_len - 1, -1, -1):
            h2, c2 = self.backward_layer(firstlayer[t], [h2, c2])
            backward_out.append([h2, c2])
        backward_out = backward_out[::-1]
        
        # 拼接
        tar = self.seq_len // 2
        combined = []
        for t in range(self.seq_len):
            o = torch.cat([firstlayer[t], forward_out[t][0], backward_out[t][0]], dim=1)
            combined.append(self.add_conv(o))
        
        # Union
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
        
        # 最终处理
        o = self.unet2(torch.cat([o1, combined[tar], o2], dim=1))
        
        # 输出
        o_heads = [outconv(o) for outconv in self.outconvlist]
        o = torch.cat(o_heads, dim=1)
        
        # 激活
        o[:, [0]] = torch.clamp(o[:, [0]], min=-8., max=8.)
        o[:, self.sigmoid_ch_ix] = torch.sigmoid(o[:, self.sigmoid_ch_ix])
        o[:, self.tanh_ch_ix] = torch.tanh(o[:, self.tanh_ch_ix])
        o[:, slice(5, 9)] = o[:, slice(5, 9)] * 3 + self.sigma_eps_default
        
        return o


if __name__ == '__main__':
    import datetime
    from decode.utils import param_io
    import decode.utils.calibration_io
    
    print("="*80)
    print("🔬 双螺旋PSF训练脚本")
    print(f"⏰ 开始时间: {datetime.datetime.now()}")
    print("="*80)
    
    # ========== 第1步：加载参数 ==========
    param_file = 'network/experiment1/param_run.yaml'
    print(f"\n[1/7] 加载参数: {param_file}")
    param = param_io.load_params(param_file)
    param.Meta.version = decode.utils.bookkeeping.decode_state()
    
    # ========== 第2步：- 先设置PSF类型 ==========
    print(f"\n[2/7] 配置双螺旋PSF")
    calibration_file = "D:/Projects/train/psfmod/spline_calibration_3d_dh_3dcal.mat"
    param.InOut.calibration_file = calibration_file
    
    # 显式设置PSF类型（原来缺少这个！）
    param.Simulation.psf_type = decode.utils.calibration_io.SMAPSplineCoefficient(
        calib_file=calibration_file
    )
    print(f"  ✓ PSF类型: {type(param.Simulation.psf_type).__name__}")
    print(f"  ✓ PSF文件: {calibration_file}")
    
    # ========== 第3步：现在才执行autoset_scaling（基于正确的PSF）==========
    print(f"\n[3/7] 重新计算scaling参数（基于双螺旋PSF）")
    param = decode.utils.param_io.autoset_scaling(param)
    print(f"  ✓ z_max: {param.Scaling.z_max}")
    print(f"  ✓ phot_max: {param.Scaling.phot_max}")
    print(f"  ✓ input_scale: {param.Scaling.input_scale}")
    
    # ========== 第4步：降低参数防止GPU OOM ==========
    print(f"\n[4/7] 优化训练参数（防止OOM）")
    
    # 关键修改：降低这些参数
    param.HyperParameter.batch_size = 4  # 从24降到4
    param.HyperParameter.channels_in = 5
    param.Simulation.emitter_av = 12  # 从15降到12
    
    print(f"  ✓ Batch size: {param.HyperParameter.batch_size} (原24，降低防OOM)")
    print(f"  ✓ 图像尺寸: {param.Simulation.img_size} (保持不变)")
    print(f"  ✓ 平均发射体: {param.Simulation.emitter_av} (原15)")
    
    # ========== 第5步：设置输出路径（带时间戳避免覆盖）==========
    print(f"\n[5/7] 配置输出路径")
    model_dir = 'network/experiment1'
    ckpt_dir = 'network/experiment1'
    from_ckpt = False
    model_dir = Path(model_dir)
    
    if not model_dir.parents[0].is_dir():
        raise FileNotFoundError(
            f"The path to the directory of 'model_out' (and even its parent folder) could not be found.")
    else:
        if not model_dir.is_dir():
            model_dir.mkdir()
            print(f"Created directory, absolute path: {model_dir.resolve()}")
    
    # 使用带日期的文件名
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_out = Path(model_dir) / f'model_dh_{date_str}.pt'
    ckpt_path = Path(ckpt_dir) / f'ckpt_dh_{date_str}.pt'
    
    param.InOut.experiment_out = str(model_dir)
    
    # 保存参数到带时间戳的文件
    param_run_path = Path(model_out).parents[0] / f'param_dh_{date_str}.yaml'
    param_io.save_params(param_run_path, param)
    
    print(f"  ✓ 模型将保存: {model_out}")
    print(f"  ✓ Checkpoint: {ckpt_path}")
    print(f"  ✓ 参数已保存: {param_run_path}")
    
    # ========== 第6步：GPU检查 ==========
    print(f"\n[6/7] GPU状态检查")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU不可用")
    
    torch.cuda.empty_cache()
    gpu_props = torch.cuda.get_device_properties(0)
    total_mem = gpu_props.total_memory / (1024**3)
    used_mem = torch.cuda.memory_allocated(0) / (1024**3)
    
    print(f"  ✓ GPU: {gpu_props.name}")
    print(f"  ✓ 总内存: {total_mem:.2f} GB")
    print(f"  ✓ 已使用: {used_mem:.2f} GB")
    print(f"  ✓ 可用: {total_mem - used_mem:.2f} GB")
    
    if total_mem < 6:
        print(f"  ⚠️ 警告：GPU内存较小，自动降低batch_size到2")
        param.HyperParameter.batch_size = 2
    
    # ========== 第7步：设置模拟器 ==========
    print(f"\n[7/7] 初始化模拟器")
    import generic.random_simulation
    
    try:
        sim_train, sim_test = generic.random_simulation.setup_random_simulation(param)
        print(f"  ✓ 模拟器设置完成")
    except Exception as e:
        print(f"  ❌ 模拟器设置失败: {e}")
        raise
    
    # ========== 开始训练设置 ==========
    print("\n" + "="*80)
    print("🚀 开始训练设置")
    print("="*80)
    
    simulator = sim_train
    from decode.neuralfitter.train import live_engine
    from decode.neuralfitter.utils import logger as logger_utils
    
    device = 'cuda'
    logger = [logger_utils.SummaryWriter(log_dir='logs',
                                         filter_keys=["dx_red_mu", "dx_red_sig",
                                                      "dy_red_mu", "dy_red_sig",
                                                      "dz_red_mu", "dz_red_sig",
                                                      "dphot_red_mu", "dphot_red_sig",
                                                      "f1",
                                                      ]),
              logger_utils.DictLogger()]
    logger = logger_utils.MultiLogger(logger)
    
    ds_train, ds_test, model, model_ls, grad_mod, post_processor, matcher, ckpt = \
        setup_trainer(sim_train, sim_test, logger, model_out, ckpt_path, device, param)
    
    dl_train, dl_test = live_engine.setup_dataloader(param, ds_train, ds_test)
    
    import Choose_Device as Device
    import Net.CNNLSTM as LS
    
    # 替换为你的自定义模型
    model = AdaptiveCNNBiLSTM(
        in_channels=1, 
        out_channels=11, 
        seq_len=param.HyperParameter.channels_in,
        pad_convs=True, 
        depth=2, 
        initial_features=48,
        sigma_eps_default=0.005,
        dropout_config={
            'spatial_dropout': True,
            'bottleneck_p': 0.3,
            'lstm_p': 0.25,  
            'output_p': 0.15,
            'adaptive': True
        }
    ).to(Device.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=0.1)
    
    # ✨ 重要：使用param中已经设置好的PSF对象
    psf = param.Simulation.psf_type.init_spline(
        xextent=param.Simulation.psf_extent[0],
        yextent=param.Simulation.psf_extent[1],
        img_shape=param.Simulation.img_size,
        device=param.Hardware.device_simulation,
        roi_size=param.Simulation.roi_size,
        roi_auto_center=param.Simulation.roi_auto_center
    )
    
    criterion = LossFunc(
        xextent=param.Simulation.psf_extent[0],
        yextent=param.Simulation.psf_extent[1],
        img_shape=param.Simulation.img_size,
        psf=psf,
        device=param.Hardware.device_simulation,
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=10)
    
    converges = False
    n = 0
    n_max = param.HyperParameter.auto_restart_param.num_restarts
    
    if from_ckpt:
        ckpt = decode.utils.checkpoint.CheckPoint.load(param.InOut.checkpoint_init)
        model.load_state_dict(ckpt.model_state)
        optimizer.load_state_dict(ckpt.optimizer_state)
        lr_scheduler.load_state_dict(ckpt.lr_sched_state)
        epoch0 = ckpt.step + 1
        model = model.train()
        print(f'Resuming training from checkpoint')
    else:
        epoch0 = 0
        while not converges and n < n_max:
            n += 1
            
            conv_check = decode.neuralfitter.utils.progress.GMMHeuristicCheck(
                ref_epoch=1,
                emitter_avg=sim_train.em_sampler.em_avg,
                threshold=param.HyperParameter.auto_restart_param.restart_treshold,
            )
            
            # 初始化科学监控器
            monitor = OverfittingMonitor(
                patience=param.HyperParameter.epochs,
                min_epochs=5
            )
            
            print("\n" + "="*80)
            print("🚀 科学化训练监控系统")
            print("="*80)
            print(f"训练轮数: {param.HyperParameter.epochs}")
            print(f"Batch size: {param.HyperParameter.batch_size}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Early Stopping Patience: {monitor.patience}")
            print("="*80 + "\n")
            
            for i in range(epoch0, param.HyperParameter.epochs):
                logger.add_scalar('learning/learning_rate', optimizer.param_groups[0]['lr'], i)
                
                train_loss = train(
                    model=model,
                    optimizer=optimizer,
                    loss=criterion,
                    dataloader=dl_train,
                    grad_rescale=param.HyperParameter.moeller_gradient_rescale,
                    grad_mod=param.HyperParameter.grad_mod,
                    epoch=i,
                    device=torch.device(param.Hardware.device),
                    logger=logger
                )
                
                val_loss, test_out = test_simple(
                    model=model, 
                    loss=criterion, 
                    dataloader=dl_test,
                    epoch=i,
                    device=torch.device(param.Hardware.device)
                )
                
                monitor.update(train_loss, val_loss)
                status, metrics = monitor.get_status(i)
                
                # 详细打印
                print(f"\n{'='*80}")
                print(f"📊 Epoch {i+1}/{param.HyperParameter.epochs} - {status}")
                print(f"{'='*80}")
                print(f"  Train Loss:     {metrics['train_loss']:.6f}")
                print(f"  Val Loss:       {metrics['val_loss']:.6f}")
                print(f"  Absolute Gap:   {metrics['absolute_gap']:.6f}")
                print(f"  Relative Gap:   {metrics['relative_gap']:.4%}")
                print(f"  Val Trend:      {metrics['val_trend']:+.6f} {'📈' if metrics['val_trend'] > 0 else '📉'}")
                print(f"  Gap Trend:      {metrics['gap_trend']:+.6f} {'📈' if metrics['gap_trend'] > 0 else '📉'}")
                print(f"  Best Val Loss:  {monitor.best_val_loss:.6f}")
                print(f"  No Improve:     {monitor.epochs_no_improve} epochs")
                print(f"{'='*80}\n")
                
                # 记录到logger
                logger.add_scalar('monitor/relative_gap', metrics['relative_gap'], i)
                logger.add_scalar('monitor/val_trend', metrics['val_trend'], i)
                logger.add_scalar('monitor/gap_trend', metrics['gap_trend'], i)
                
                """Post-Process and Evaluate"""
                decode.neuralfitter.train.live_engine.log_train_val_progress.post_process_log_test(
                    loss_cmp=test_out.loss,
                    loss_scalar=val_loss,
                    x=test_out.x,
                    y_out=test_out.y_out,
                    y_tar=test_out.y_tar,
                    weight=test_out.weight,
                    em_tar=ds_test.emitter,
                    px_border=-0.5,
                    px_size=1.,
                    post_processor=post_processor,
                    matcher=matcher,
                    logger=logger,
                    step=i
                )
                
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()
                
                model_ls.save(model, None)
                ckpt.dump(model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict(),
                        log=logger.logger[1].log_dict, step=i)
                
                # Early Stopping 检查
                if monitor.should_stop(i):
                    print("\n" + "="*80)
                    print("🛑 Early Stopping Triggered")
                    print(f"验证损失已 {monitor.patience} 轮未改善，提前停止训练")
                    print("="*80 + "\n")
                    break
                
                """Draw new samples"""
                if param.Simulation.mode in 'acquisition':
                    del ds_train._frames
                    del ds_train._emitter
                    del ds_train._bg_frames
                    del ds_train._nobg_frames
                    ds_train.sample(True)
                elif param.Simulation.mode != 'samples':
                    raise ValueError
        
            # ===== 训练总结 =====
            print("\n" + "="*80)
            print("🎓 训练完成 - 科学评估报告")
            print("="*80)
            
            summary = monitor.get_summary()
            if summary:
                print(f"\n📊 统计指标:")
                print(f"  平均训练损失:     {summary['avg_train_loss']:.6f}")
                print(f"  平均验证损失:     {summary['avg_val_loss']:.6f}")
                print(f"  平均相对Gap:      {summary['avg_relative_gap']:.4%}")
                print(f"  最佳验证损失:     {summary['min_val_loss']:.6f} (Epoch {summary['best_epoch']})")
                
                print(f"\n📊 最终指标:")
                print(f"  最终训练损失:     {summary['final_train_loss']:.6f}")
                print(f"  最终验证损失:     {summary['final_val_loss']:.6f}")
                print(f"  最终相对Gap:      {summary['final_relative_gap']:.4%}")
                
                print(f"\n🎯 收敛质量评分:   {summary['convergence_score']:.1f}/100")
                
                # 给出建议
                print(f"\n💡 优化建议:")
                if summary['convergence_score'] >= 80:
                    print("  ✅ 训练效果优秀，模型已良好收敛")
                elif summary['convergence_score'] >= 60:
                    print("  ⚠️  训练效果良好，但仍有优化空间")
                    if summary['final_relative_gap'] > 0.15:
                        print("  → 建议增大 dropout 率或使用更强的正则化")
                else:
                    print("  ❌ 训练效果欠佳，建议调整超参数")
                    if summary['final_relative_gap'] > 0.20:
                        print("  → 严重过拟合，建议: 1) 增大dropout 2) 减小模型容量 3) 增加数据增强")
                    elif summary['avg_train_loss'] > 0.5:
                        print("  → 欠拟合，建议: 1) 增大模型容量 2) 降低dropout 3) 调整学习率")
                
                print("="*80 + "\n")
                
                # 保存详细结果
                import json
                comparison_results = {
                    'train_losses': [float(l) for l in monitor.train_losses],
                    'val_losses': [float(l) for l in monitor.val_losses],
                    'relative_gaps': [float(g) for g in monitor.relative_gaps],
                    'summary': convert_to_serializable(summary)
                }
                
                result_file = model_dir / 'scientific_training_report.json'
                with open(result_file, 'w') as f:
                    json.dump(comparison_results, f, indent=2)
                
                print(f"💾 详细报告已保存: {result_file}\n")
                
            break
    
    converges = True
    if converges:
        print("\n" + "="*80)
        print("✅ 训练完成！")
        print(f"⏰ 完成时间: {datetime.datetime.now()}")
        print(f"📦 模型已保存: {model_out}")
        print("="*80)
    else:
        raise ValueError(f"Training aborted after {n_max} restarts. "
                         "You can try to reduce the learning rate by a factor of 2."
                         "\nIt is also possible that the simulated data is to challenging. "
                         "Check if your background and intensity values are correct "
                         "and possibly lower the average number of emitters.")