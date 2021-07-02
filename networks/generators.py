import torch
import torch.nn as nn
import torch.nn.functional as F

class NLBlockND_cross(nn.Module):
    # Our implementation of the attention block referenced https://github.com/tea1528/Non-Local-NN-Pytorch

    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND_cross, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):
        #x_thisBranch for g and theta
        #x_otherBranch for phi
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        # print(x_thisBranch.shape)

        batch_size = x_thisBranch.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        # elif self.mode == "concatenate":
        else: #default as concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x_thisBranch

        return z

class dualAtt_24(nn.Module):

    def __init__(self):
        super(dualAtt_24, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv3d_7 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.pathC_bn1 = nn.BatchNorm3d(64)


        self.conv3d_8 = nn.Conv3d(in_channels=16, out_channels=4, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.conv3d_9 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.pathC_bn2 = nn.BatchNorm3d(1)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 6)

        """layers for path global"""
        self.path1_block1_conv = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path1_block1_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal11 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.path1_block2_conv = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path1_block2_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal12 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1,2,2), padding=1)
        self.path1_block3_NLCross = NLBlockND_cross(32)

        self.path2_block1_conv = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path2_block1_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal21 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.path2_block2_conv = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path2_block2_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal22 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1,2,2), padding=1)
        self.path2_block3_NLCross = NLBlockND_cross(32)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # total_start_time = time.time()
        x_path1 = torch.unsqueeze(x[:, 0, :, :, :], 1)
        x_path2 = torch.unsqueeze(x[:, 1, :, :, :], 1)

        """path global (attention)"""
        x_path1 = self.path1_block1_conv(x_path1)
        x_path1 = self.path1_block1_bn(x_path1)
        x_path1 = self.relu(x_path1)
        x_path1 = self.maxpool_downsample_pathGlobal11(x_path1)
        # print(x_path1.shape)

        x_path1 = self.path1_block2_conv(x_path1)
        x_path1 = self.path1_block2_bn(x_path1)
        x_path1 = self.relu(x_path1)
        x_path1_0 = self.maxpool_downsample_pathGlobal12(x_path1)
        # print(x_path1.shape)


        x_path2 = self.path2_block1_conv(x_path2)
        x_path2 = self.path2_block1_bn(x_path2)
        x_path2 = self.relu(x_path2)
        x_path2 = self.maxpool_downsample_pathGlobal21(x_path2)
        # print(x_path2.shape)

        x_path2 = self.path2_block2_conv(x_path2)
        x_path2 = self.path2_block2_bn(x_path2)
        x_path2 = self.relu(x_path2)
        x_path2_0 = self.maxpool_downsample_pathGlobal22(x_path2)
        # print(x_path2.shape)

        x_path1 = self.path1_block3_NLCross(x_path1_0, x_path2_0)
        x_path1 = self.relu(x_path1)

        x_path2 = self.path2_block3_NLCross(x_path2_0, x_path1_0)
        x_path2 = self.relu(x_path2)

        x_pathC = torch.cat((x_path1, x_path2), 1)

        """path combined"""
        x = x_pathC
        x = self.pathC_bn1(x)

        x = self.conv3d_7(x)
        x = self.relu(x)

        x = self.conv3d_8(x)
        x = self.relu(x)

        x = self.conv3d_9(x)
        x = self.pathC_bn2(x)

        x = x.view(x.size()[0], -1)
        x = self.relu(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        # time_cost = time.time() - total_start_time
        # print('1 whole cycle time cost {}s'.format(time_cost))
        # time.sleep(30)
        return x
class dualAtt_25(nn.Module):

    def __init__(self):
        super(dualAtt_25, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv3d_7 = nn.Conv3d(in_channels=64, out_channels=16, kernel_size=1, stride=(1, 1, 1), padding=0)
        self.pathC_bn1 = nn.BatchNorm3d(64)


        self.conv3d_8 = nn.Conv3d(in_channels=16, out_channels=4, kernel_size=3, stride=(1, 1, 1), padding=1)

        self.conv3d_9 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.pathC_bn2 = nn.BatchNorm3d(1)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 6)

        """layers for path global"""
        self.path1_block1_conv = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path1_block1_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal11 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.path1_block2_conv = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path1_block2_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal12 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1,2,2), padding=1)

        self.path2_block1_conv = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path2_block1_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal21 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.path2_block2_conv = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=1,
            bias=False)
        self.path2_block2_bn = nn.BatchNorm3d(32)
        self.maxpool_downsample_pathGlobal22 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1,2,2), padding=1)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x_path1 = torch.unsqueeze(x[:, 0, :, :, :], 1)
        x_path2 = torch.unsqueeze(x[:, 1, :, :, :], 1)

        """path global (attention)"""
        x_path1 = self.path1_block1_conv(x_path1)
        x_path1 = self.path1_block1_bn(x_path1)
        x_path1 = self.relu(x_path1)
        x_path1 = self.maxpool_downsample_pathGlobal11(x_path1)
        # print(x_path1.shape)

        x_path1 = self.path1_block2_conv(x_path1)
        x_path1 = self.path1_block2_bn(x_path1)
        x_path1 = self.relu(x_path1)
        x_path1_0 = self.maxpool_downsample_pathGlobal12(x_path1)
        # print(x_path1.shape)


        x_path2 = self.path2_block1_conv(x_path2)
        x_path2 = self.path2_block1_bn(x_path2)
        x_path2 = self.relu(x_path2)
        x_path2 = self.maxpool_downsample_pathGlobal21(x_path2)
        # print(x_path2.shape)

        x_path2 = self.path2_block2_conv(x_path2)
        x_path2 = self.path2_block2_bn(x_path2)
        x_path2 = self.relu(x_path2)
        x_path2_0 = self.maxpool_downsample_pathGlobal22(x_path2)
        # print(x_path2.shape)

        # x_path1 = self.path1_block3_NLCross(x_path1_0, x_path2_0)
        x_path1 = self.relu(x_path1_0)

        # x_path2 = self.path2_block3_NLCross(x_path2_0, x_path1_0)
        x_path2 = self.relu(x_path2_0)

        x_pathC = torch.cat((x_path1, x_path2), 1)

        """path combined"""
        x = x_pathC
        x = self.pathC_bn1(x)

        x = self.conv3d_7(x)
        x = self.relu(x)

        x = self.conv3d_8(x)
        x = self.relu(x)
        # print(x.shape)

        x = self.conv3d_9(x)
        x = self.pathC_bn2(x)

        # print(x.shape)

        x = x.view(x.size()[0], -1)
        x = self.relu(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x

def AttentionReg():
    model = dualAtt_24()
    print('using AttentionReg')
    return model
def FeatureReg():
    model = dualAtt_25()
    print('using FeatureReg, CNN network without the attention block for ablation study')
    return model


