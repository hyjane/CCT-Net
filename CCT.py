import torch
from torchvision import models as resnet_model
from torch import nn

import cv2
#decoder，解码还原结构
class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


#SEBlock
class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y


#our module
class CCT_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(CCT_Net, self).__init__()
        # 从torchhub加载预训练的模型
        transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        # 调用pytorch中自带的resnet网络模型
        resnet = resnet_model.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.patch_embed = transformer.patch_embed
        # 执行2个transformer模块,transformer4
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(2)]
        )

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, padding=0)
        self.se = SEBlock(channel=128)
        self.conv2d = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0)

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)
        #第一次
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=2)
        self.tf_conv1 = nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=3, stride=17)
        self.in_tf2conv = nn.ConvTranspose2d(in_channels=192, out_channels=3, kernel_size=3, stride=17)
        #第二次
        self.tf2_conv = nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=8, stride=8)
        self.se2 = SEBlock(channel=256)
        self.conv2d2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        #第三次
        self.tf3_conv = nn.ConvTranspose2d(in_channels=192, out_channels=256, kernel_size=4, stride=4)
        self.se3 = SEBlock(channel=512)
        self.conv2d3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        #第四次
        self.tf4_conv = nn.ConvTranspose2d(in_channels=192, out_channels=512, kernel_size=2, stride=2)
        self.se4 = SEBlock(channel=1024)
        self.conv2d4= nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.conv(x)
        f_c0 = self.firstconv(x)
        f_c0 = self.firstbn(f_c0)
        f_c0 = self.firstrelu(f_c0)
        f_c1 = self.encoder1(f_c0)
        f_c2 = self.encoder2(f_c1)
        f_c3 = self.encoder3(f_c2)
        feature_cnn = self.encoder4(f_c3)

        for i in range(2):
            emb = self.transformers[i](emb)
        f_tf1 = emb.permute(0, 2, 1)
        f_tf1 = f_tf1.view(b, 192, 14, 14) 
        in_tf2 = self.in_tf2conv(f_tf1)
        f_tf1 = self.tf_conv1(f_tf1)

        f_tf1 = self.conv_seq_img(f_tf1)

        # 融合第1层特征,并进行se操作
        feature_cat1 = torch.cat((f_c0,f_tf1), dim=1)
        feature_att1 = self.se(feature_cat1)
        feature_out1 = self.conv2d(feature_att1)

        # 第2层Transformer特征，f_tf->feature_transformer
        emb = self.patch_embed(in_tf2)
        for i in range(2):
            emb = self.transformers[i](emb)
        f_tf2 = emb.permute(0, 2, 1)
        f_tf2 = f_tf2.view(b, 192, 14, 14)
        in_tf3 = self.in_tf2conv(f_tf2)
        f_tf2 = self.tf2_conv(f_tf2)

        # 融合第2层特征,并进行se操作
        feature_cat2 = torch.cat((f_c2, f_tf2), dim=1)
        feature_att2 = self.se2(feature_cat2)
        feature_out2 = self.conv2d2(feature_att2)

        # 第3层Transformer特征，f_tf->feature_transformer
        emb = self.patch_embed(in_tf3)
        for i in range(2):
            emb = self.transformers[i](emb)
        f_tf3 = emb.permute(0, 2, 1)
        f_tf3 = f_tf3.view(b, 192, 14, 14)
        in_tf4 = self.in_tf2conv(f_tf3)
        f_tf3 = self.tf3_conv(f_tf3)

        # 融合第3层特征,并进行se操作
        feature_cat3 = torch.cat((f_c3, f_tf3), dim=1)
        feature_att3 = self.se3(feature_cat3)
        feature_out3 = self.conv2d3(feature_att3)
        
        # 第4层Transformer特征，f_tf->feature_transformer
        emb = self.patch_embed(in_tf4)
        for i in range(2):
            emb = self.transformers[i](emb)
        f_tf4 = emb.permute(0, 2, 1)
        f_tf4 = f_tf4.view(b, 192, 14, 14)
        fu_ftf4 = self.tf4_conv(f_tf4)
        f_tf4 = self.conv_seq_img(f_tf4)
        print("f_tf43_size:", f_tf4.shape)

        # 融合第4层特征,并进行se操作
        feature_cat = torch.cat((feature_cnn, fu_ftf4), dim=1)
        feature_att4 = self.se4(feature_cat)
        feature_out4 = self.conv2d4(feature_att4)

        # 上采样还原部分，跳跃连接，加入之前下采样的特征进行丰富特征
        d4 = self.decoder4(feature_out4)+feature_out3
        d3 = self.decoder3(d4) + feature_out2
        d2 = self.decoder2(d3) + feature_out1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out
