import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionAdapter(nn.Module):
    def __init__(self, in_channels=768, features=192, out_channels=[96, 192, 384, 768]):
        super(VisionAdapter, self).__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        # feature fusion, 每一层的特征都投影到192维
        self.feature_align = self._make_feature_align(in_shapes=out_channels, out_shape=features, groups=1)
        # self.feature_align.output_conv = nn.Conv2d(features*4, features, kernel_size=1, stride=1, padding=0, bias=False)

        # 空间压缩投影: 使用 Stride=2 的深度可分离卷积将 40x30 压至 20x15
        # 这样既保留了空间局部性，又避免了 Attention 的计算开销
        self.spatial_downsample = nn.Sequential(
            nn.Conv2d(features * 4, features * 4, kernel_size=3, stride=2, padding=1, groups=features * 4, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 4, in_channels, kernel_size=1, stride=1, bias=False) # 确保最终维度是 768
        )

    def _make_feature_align(self, in_shapes, out_shape=192, groups=1):
        align = nn.Module()
        align.layer1 = nn.Conv2d(in_shapes[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        align.layer2 = nn.Conv2d(in_shapes[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        align.layer3 = nn.Conv2d(in_shapes[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        align.layer4 = nn.Conv2d(in_shapes[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        return align
    
    def forward(self, out_features, patch_h, patch_w):
        """
        input : out_features is a list of [batch, num_patches, embed_dim], num=4 --> [B, 1200, 768] * 4
        """
        out = []
        # 将Transformer的序列格式 [B, N, C]重整为图像格式 [B, C, patch_h, patch_w]-> [B, 768, patch_h, patch_w]
        # 分别对每个层的特征进行卷积投影，得到4个不同通道数的特征图:[96, 192, 384, 768]
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)
        
        # [B, C, 768], C不同,代表关注信息维度不同,浅层关注的特征维度更少,深层更精细
        layer1, layer2, layer3, layer4 = out
        # 再对每个层的特征进行通道对齐,对齐到192通道数  -->  [B, 192, 40, 30]
        layer1 = self.feature_align.layer1(layer1)
        layer2 = self.feature_align.layer2(layer2)
        layer3 = self.feature_align.layer3(layer3)
        layer4 = self.feature_align.layer4(layer4)

        # 层级融合 (Concatenate) -> [B, 768, 40, 30]
        combined = torch.cat([layer1, layer2, layer3, layer4], dim=1)

        # 空间下采样 -> [B, 768, 20, 15]
        downsampled = self.spatial_downsample(combined)

        vision_tokens = downsampled.flatten(2).permute(0, 2, 1)
        # 每个图片最后的token_shape是[B, 300, 768]
        return vision_tokens

class VisionEncoder(nn.Module):
    def __init__(self, dino_type='base', 
                 features=192, 
                 out_channels=[96, 192, 384, 768], 
                 backbone = None):
        super(VisionEncoder, self).__init__()

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            "large": [4, 11, 17, 23],
        }

        self.dino_type = dino_type

        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.adapter = VisionAdapter(in_channels=self.backbone.embed_dim,
                               features=features,
                               out_channels=out_channels)
        
    
    def forward(self, x):
        # 每个patch是16x16像素，这里计算patch块的高宽数量
        patch_h = x.shape[-2] // 16
        patch_w = x.shape[-1] // 16
        num_patches = patch_h * patch_w

        # 从DINO每个层的提取出的特征形状都是相同的：
        # features是一个包含4个张量的列表：[tensor1, tensor2, tensor3, tensor4]
        # 每个形状:[batch, num_patches, embed_dim]
        # embed_dim = 768 (ViT-Base的特征维度)
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.dino_type]
            )
        out = self.adapter(features, patch_h, patch_w)
        return out
    
if __name__ == '__main__':
    repo_dir = "./dinov3" 
    dino_ckpt = "web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = torch.hub.load(
            repo_dir, 
            'dinov3_vits16', 
            source='local', 
            weights="/home/silei/WorkSpace_git/Mobile_Manipulation/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        )

