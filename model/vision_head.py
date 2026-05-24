from typing import Optional

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Wraps a frozen DINOv3 backbone. Extracts CLS token per image."""

    intermediate_layer_idx = {
        'small': [2, 5, 8, 11],
        'base': [2, 5, 8, 11],
        'large': [4, 11, 17, 23],
    }

    def __init__(self, backbone, dino_type: str = 'base', use_last_layer: bool = True):
        super().__init__()
        self.backbone = backbone
        self.dino_type = dino_type
        self.use_last_layer = use_last_layer
        for p in self.backbone.parameters():
            p.requires_grad = False

    @property
    def embed_dim(self) -> int:
        return self.backbone.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token from DINOv3.

        Args:
            x: [B, 3, H, W]
        Returns:
            [B, embed_dim] CLS token from the last intermediate layer.
        """
        layer_indices = self.intermediate_layer_idx[self.dino_type]
        if self.use_last_layer:
            features = self.backbone.get_intermediate_layers(x, n=[layer_indices[-1]])
            cls_token = features[0][:, 0]  # [B, embed_dim]
        else:
            features = self.backbone.get_intermediate_layers(x, n=layer_indices)
            # average CLS tokens across all intermediate layers
            cls_tokens = [f[:, 0] for f in features]
            cls_token = torch.stack(cls_tokens, dim=0).mean(dim=0)
        return cls_token


class MultiViewVisionEncoder(nn.Module):
    """Processes multi-view images, outputs one CLS token per view.

    Input:  images [B, V, 3, H, W]
    Output: [B, V, proj_dim]
    """

    def __init__(self, backbone, dino_type: str = 'base', proj_dim: int = 512):
        super().__init__()
        self.encoder = VisionEncoder(backbone, dino_type=dino_type)
        self.proj = nn.Linear(self.encoder.embed_dim, proj_dim)

    def forward(
        self,
        images: torch.Tensor,
        camera_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            images: [B, V, 3, H, W]
            camera_types: [B, V] or [V], 0=fixed, 1=dynamic (unused here, available for callers)
        Returns:
            vision_tokens: [B, V, proj_dim]
        """
        B, V, C, H, W = images.shape
        x = images.view(B * V, C, H, W)
        cls_tokens = self.encoder(x)  # [B*V, embed_dim]
        cls_tokens = cls_tokens.view(B, V, -1)  # [B, V, embed_dim]
        vision_tokens = self.proj(cls_tokens)  # [B, V, proj_dim]
        return vision_tokens


if __name__ == '__main__':
    repo_dir = "./dinov3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = torch.hub.load(
        repo_dir,
        'dinov3_vitb16',
        source='local',
        weights="web_pth/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    )

    # test single-view
    encoder = VisionEncoder(backbone, dino_type='base').to(device)
    dummy = torch.randn(2, 3, 640, 480).to(device)
    out = encoder(dummy)
    print(f"VisionEncoder output: {out.shape}")  # [2, 768]
    assert out.shape == (2, backbone.embed_dim)

    # test multi-view
    mv_encoder = MultiViewVisionEncoder(backbone, dino_type='base', proj_dim=512).to(device)
    dummy_mv = torch.randn(4, 6, 3, 640, 480).to(device)
    out_mv = mv_encoder(dummy_mv)
    print(f"MultiViewVisionEncoder output: {out_mv.shape}")  # [4, 6, 512]
    assert out_mv.shape == (4, 6, 512)

    print("--- All vision shape tests passed! ---")
