import torch
import torch.nn.functional as F

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