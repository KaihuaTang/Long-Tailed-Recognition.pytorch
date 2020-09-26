import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .bbox_head import BBoxHead


@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg


        # TDE parameters
        self.DECONFOUND_TRAIN = True
        self.NUM_HEAD = 2
        self.SCALE = 16.0 / self.NUM_HEAD
        self.SCALE_WEIGHT = 1.0 / 32.0
        self.ALPHA = 1.5
        self.CAUSAL_INFER = True
        self.KEEP_FG = True

        #self.MU = 0.9
        # scale the original mu according to the learning rate
        self.MU = 1.0 - (1 - 0.9) * 0.002

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.DECONFOUND_TRAIN:
                print('=====> Start DECONFOUND TRAINING')
                self.fc_cls = nn.Parameter(torch.Tensor(self.num_classes + 1, self.cls_last_dim).cuda(), requires_grad=True)
                self.register_buffer("causal_embed", torch.FloatTensor(1, self.cls_last_dim).zero_())
                self.reset_parameters(self.fc_cls)
            else:
                self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, gt_label=None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        if self.DECONFOUND_TRAIN:
            cls_score = self.cos_forward(x_cls, gt_label) if self.with_cls else None
        else:
            cls_score = self.fc_cls(x_cls) if self.with_cls else None

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def cos_forward(self, x, gt_label):
        normed_w = self.multi_head_call(self.causal_norm, self.fc_cls, num_head=self.NUM_HEAD, weight=self.SCALE_WEIGHT)
        normed_x = self.multi_head_call(self.l2_norm, x, num_head=self.NUM_HEAD)

        # learning adjustment vector
        self.update_embed(x, gt_label)

        y = torch.mm(normed_x * self.SCALE, normed_w.t())

        if (not self.training) and self.CAUSAL_INFER:
            normed_e = self.multi_head_call(self.l2_norm, self.causal_embed, num_head=self.NUM_HEAD)
            head_dim = x.shape[1] // self.NUM_HEAD
            x_list = torch.split(normed_x, head_dim, dim=1)
            e_list = torch.split(normed_e, head_dim, dim=1)
            w_list = torch.split(normed_w, head_dim, dim=1)
            output = []

            for nx, ne, nw in zip(x_list, e_list, w_list):
                cos_val, sin_val = self.get_cos_sin(nx, ne)
                #n = torch.norm(nx, 2, 1, keepdim=True)
                y0 = torch.mm((self.ALPHA * cos_val * ne) * self.SCALE, nw.t())
                output.append(y0)

            y0 = sum(output)
            
            if self.KEEP_FG:
                # IMPORTANT
                # the difference between lvis_old and lvis1.0 is that the ID of background category is now NUM_CLASS rather than 0
                old_score = y.softmax(-1)
                new_score = (y - y0).softmax(-1)
                # keep the fg bg the same
                final_score_bg = old_score[:, -1].contiguous().view(-1, 1)
                final_score_fg = old_score[:, :-1].sum(-1, keepdim=True) / (new_score[:, :-1].sum(-1, keepdim=True) + 1e-10) * (new_score[:, :-1] + 1e-10)
                final_score = torch.cat([final_score_fg, final_score_bg], dim=-1)
                y = final_score
            else:
                y = y - y0
        return y

    def get_cos_sin(self, x, y):
        cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
        sin_val = (1 - cos_val * cos_val).sqrt()
        return cos_val, sin_val

    def multi_head_call(self, func, x, num_head, weight=None):
        assert len(x.shape) == 2
        head_dim = x.shape[1] // num_head
        x_list = torch.split(x, head_dim, dim=1)
        if weight:
            y_list = [func(item, weight) for item in x_list]
        else:
            y_list = [func(item) for item in x_list]
        assert len(x_list) == num_head
        assert len(y_list) == num_head
        return torch.cat(y_list, dim=1)

    def l2_norm(self, x):
        normed_x = x / (torch.norm(x, 2, 1, keepdim=True) + 1e-9)
        return normed_x

    def capsule_norm(self, x):
        norm= torch.norm(x.clone(), 2, 1, keepdim=True) + 1e-9
        normed_x = (norm / (1 + norm)) * (x / norm)
        return normed_x

    def causal_norm(self, x, weight):
        norm= torch.norm(x, 2, 1, keepdim=True) + 1e-9
        normed_x = x / (norm + weight)
        return normed_x

    def update_embed(self, targets, gt_label):
        if self.training:
            # remove background
            with torch.no_grad():
                fg_target = targets[gt_label > 0].clone().detach().mean(0, keepdim=True)
                self.causal_embed = self.MU * self.causal_embed + fg_target
        return


@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared4Conv1FCBBoxHead(ConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
