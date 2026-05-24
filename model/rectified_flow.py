import torch
import torch.nn.functional as F


def consistency_loss(h_base: torch.Tensor, h_joint: torch.Tensor) -> torch.Tensor:
    """Hidden consistency loss between base and joint branch representations.

    Encourages the two branches to share a common latent structure
    while still being free to specialize.
    """
    return F.mse_loss(h_base.detach(), h_joint) + F.mse_loss(h_base, h_joint.detach())


class ActionRectifiedFlow:
    def euler(self, x_t, v, dt):
        """ 使用欧拉方法步进 ODE: f(t+dt) = f(t) + dt * v """
        return x_t + v * dt

    def create_flow(self, x_1, t, x_0=None):
        """ 
        构建从噪声 x_0 到 动作 x_1 的直线轨迹
        公式: x_t = t * x_1 + (1 - t) * x_0
        
        Args:
            x_1: 原始动作序列 [B, 10, 1]
            t: 时间步长 [B]
            x_0: 噪声 [B, 10, 1]
        """
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        # 优化点：动态增加维度以适配 (B, 10, 1)，而不是写死 unsqueeze 次数
        # 使得 t 的形状从 [B] 变为 [B, 1, 1]
        while t.ndim < x_1.ndim:
            t = t.unsqueeze(-1)
            
        x_t = t * x_1 + (1 - t) * x_0
        return x_t, x_0
    
    def mse_loss(self, v_pred, x_1, x_0):
        """ 
        计算速度向量场的 MSE 损失
        目标速度 v_target = x_1 - x_0 (匀速直线运动的导数)
        """
        v_target = x_1 - x_0
        return F.mse_loss(v_pred, v_target)


class DualStreamRectifiedFlow:
    def __init__(self):
        self.base_flow = ActionRectifiedFlow()
        self.joint_flow = ActionRectifiedFlow()

    def make_targets(self, base_action_rel, joint_action_rel, t, base_noise=None, joint_noise=None):
        base_xt, base_noise = self.base_flow.create_flow(base_action_rel, t, base_noise)
        joint_xt, joint_noise = self.joint_flow.create_flow(joint_action_rel, t, joint_noise)
        return {
            "base_xt": base_xt,
            "joint_xt": joint_xt,
            "base_noise": base_noise,
            "joint_noise": joint_noise,
        }

    def loss(self, base_pred, joint_pred, base_action_rel, joint_action_rel, base_noise, joint_noise):
        base_loss = self.base_flow.mse_loss(base_pred, base_action_rel, base_noise)
        joint_loss = self.joint_flow.mse_loss(joint_pred, joint_action_rel, joint_noise)
        return {
            "base_loss": base_loss,
            "joint_loss": joint_loss,
            "total_loss": base_loss + joint_loss,
        }