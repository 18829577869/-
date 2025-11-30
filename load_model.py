import sys
import warnings
# 抑制 Gym 相关的废弃警告（已使用 Gymnasium）
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
warnings.filterwarnings('ignore', message='.*Please upgrade to Gymnasium.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 临时重定向 stderr 以捕获 gym 的警告输出
class SuppressGymWarning:
    def __init__(self):
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        sys.stderr = self
        
    def __exit__(self, *args):
        sys.stderr = self.original_stderr
        
    def write(self, text):
        if 'Gym has been unmaintained' in text or 'Please upgrade to Gymnasium' in text:
            return  # 忽略这些警告
        self.original_stderr.write(text)
        
    def flush(self):
        self.original_stderr.flush()

# 在导入可能触发 gym 的包之前抑制警告
with SuppressGymWarning():
    import gymnasium as gym  # 使用 Gymnasium 替换 Gym 以避免警告
    from stable_baselines3 import PPO

# 加载模型（替换为实际文件路径，在抑制警告的上下文中）
with SuppressGymWarning():
    model = PPO.load("ppo_stock_v7.zip")

print("模型加载成功！")