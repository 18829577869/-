"""
动态参数调整和自动学习优化模块
支持实时参数调整、自动优化和自适应学习
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import json
import os


class ParameterRange:
    """参数范围定义"""
    
    def __init__(self, min_val: float, max_val: float, step: Optional[float] = None, 
                 param_type: str = 'continuous'):
        """
        初始化参数范围
        
        参数:
            min_val: 最小值
            max_val: 最大值
            step: 步长（对于离散参数）
            param_type: 参数类型 ('continuous', 'discrete', 'integer')
        """
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.param_type = param_type
        
        # 验证范围
        if min_val >= max_val:
            raise ValueError(f"最小值 ({min_val}) 必须小于最大值 ({max_val})")
    
    def sample(self, n: int = 1) -> np.ndarray:
        """在范围内采样"""
        if self.param_type == 'continuous':
            return np.random.uniform(self.min_val, self.max_val, n)
        elif self.param_type == 'discrete':
            values = np.arange(self.min_val, self.max_val + self.step, self.step)
            return np.random.choice(values, n)
        elif self.param_type == 'integer':
            return np.random.randint(int(self.min_val), int(self.max_val) + 1, n)
        else:
            raise ValueError(f"不支持的参数类型: {self.param_type}")


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化性能跟踪器
        
        参数:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.performance_history = []
        self.parameter_history = []
        self.timestamps = []
        
    def add(self, performance: float, parameters: Dict, timestamp: Optional[datetime] = None):
        """
        添加性能记录
        
        参数:
            performance: 性能指标（越高越好）
            parameters: 参数字典
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.performance_history.append(performance)
        self.parameter_history.append(parameters.copy())
        self.timestamps.append(timestamp)
        
        # 保持窗口大小
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
            self.parameter_history.pop(0)
            self.timestamps.pop(0)
    
    def get_best_parameters(self, n: int = 1) -> List[Dict]:
        """
        获取最佳参数
        
        参数:
            n: 返回前N个最佳参数
        
        返回:
            最佳参数字典列表
        """
        if len(self.performance_history) == 0:
            return []
        
        # 排序（性能从高到低）
        sorted_indices = np.argsort(self.performance_history)[::-1]
        top_n = min(n, len(sorted_indices))
        
        best_params = []
        for i in range(top_n):
            idx = sorted_indices[i]
            best_params.append({
                'parameters': self.parameter_history[idx].copy(),
                'performance': self.performance_history[idx],
                'timestamp': self.timestamps[idx]
            })
        
        return best_params
    
    def get_recent_trend(self, window: int = 10) -> float:
        """
        获取最近趋势（滑动平均）
        
        参数:
            window: 窗口大小
        
        返回:
            趋势值（正数表示改善，负数表示恶化）
        """
        if len(self.performance_history) < window:
            return 0.0
        
        recent = self.performance_history[-window:]
        earlier = self.performance_history[-window*2:-window] if len(self.performance_history) >= window*2 else recent
        
        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier) if len(earlier) > 0 else recent_mean
        
        return recent_mean - earlier_mean


class DynamicParameterOptimizer:
    """动态参数优化器"""
    
    def __init__(self,
                 parameter_ranges: Dict[str, ParameterRange],
                 optimization_method: str = 'bayesian',  # 'random', 'grid', 'bayesian', 'gradient'
                 adaptation_rate: float = 0.1,
                 exploration_rate: float = 0.2,
                 performance_window: int = 100):
        """
        初始化动态参数优化器
        
        参数:
            parameter_ranges: 参数范围字典 {param_name: ParameterRange}
            optimization_method: 优化方法
            adaptation_rate: 适应率（参数调整速度）
            exploration_rate: 探索率（随机探索的概率）
            performance_window: 性能跟踪窗口大小
        """
        self.parameter_ranges = parameter_ranges
        self.optimization_method = optimization_method
        self.adaptation_rate = adaptation_rate
        self.exploration_rate = exploration_rate
        self.performance_tracker = PerformanceTracker(window_size=performance_window)
        
        # 当前参数（初始化为范围中点）
        self.current_parameters = {}
        for param_name, param_range in parameter_ranges.items():
            mid_val = (param_range.min_val + param_range.max_val) / 2
            self.current_parameters[param_name] = mid_val
        
        # 优化历史
        self.optimization_history = []
        
    def suggest_parameters(self, use_exploration: bool = False) -> Dict:
        """
        建议下一组参数
        
        参数:
            use_exploration: 是否使用探索模式
        
        返回:
            参数字典
        """
        if use_exploration or np.random.random() < self.exploration_rate:
            # 探索模式：随机采样
            return self._random_sample()
        else:
            # 利用模式：基于历史优化
            if self.optimization_method == 'bayesian':
                return self._bayesian_optimization()
            elif self.optimization_method == 'gradient':
                return self._gradient_based_optimization()
            elif self.optimization_method == 'random':
                return self._random_sample()
            else:
                return self.current_parameters.copy()
    
    def _random_sample(self) -> Dict:
        """随机采样参数"""
        params = {}
        for param_name, param_range in self.parameter_ranges.items():
            samples = param_range.sample(1)
            params[param_name] = float(samples[0])
        return params
    
    def _bayesian_optimization(self) -> Dict:
        """贝叶斯优化（简化版，使用UCB策略）"""
        # 获取最佳参数
        best_params_list = self.performance_tracker.get_best_parameters(n=3)
        
        if len(best_params_list) == 0:
            return self._random_sample()
        
        # 使用最佳参数作为基础，添加小的随机扰动
        best_params = best_params_list[0]['parameters']
        suggested_params = {}
        
        for param_name, param_range in self.parameter_ranges.items():
            if param_name in best_params:
                base_val = best_params[param_name]
                # 在最佳值附近搜索
                perturbation = (param_range.max_val - param_range.min_val) * self.adaptation_rate
                new_val = base_val + np.random.uniform(-perturbation, perturbation)
                # 限制在范围内
                new_val = np.clip(new_val, param_range.min_val, param_range.max_val)
                suggested_params[param_name] = float(new_val)
            else:
                samples = param_range.sample(1)
                suggested_params[param_name] = float(samples[0])
        
        return suggested_params
    
    def _gradient_based_optimization(self) -> Dict:
        """基于梯度的优化"""
        trend = self.performance_tracker.get_recent_trend(window=10)
        
        # 如果性能改善，继续当前方向；如果恶化，调整参数
        if abs(trend) < 0.01:
            # 性能稳定，轻微随机调整
            return self._random_sample()
        
        suggested_params = {}
        for param_name, param_range in self.parameter_ranges.items():
            current_val = self.current_parameters.get(param_name, 
                                                     (param_range.min_val + param_range.max_val) / 2)
            
            # 根据趋势调整
            step = (param_range.max_val - param_range.min_val) * self.adaptation_rate
            if trend > 0:
                # 性能改善，保持方向
                new_val = current_val + np.random.uniform(0, step)
            else:
                # 性能恶化，反向调整
                new_val = current_val - np.random.uniform(0, step)
            
            # 限制在范围内
            new_val = np.clip(new_val, param_range.min_val, param_range.max_val)
            suggested_params[param_name] = float(new_val)
        
        return suggested_params
    
    def update_performance(self, performance: float, parameters: Dict):
        """
        更新性能并调整参数
        
        参数:
            performance: 性能指标（越高越好）
            parameters: 使用的参数字典
        """
        # 记录性能
        self.performance_tracker.add(performance, parameters)
        
        # 更新当前参数（如果性能更好）
        best_params_list = self.performance_tracker.get_best_parameters(n=1)
        if len(best_params_list) > 0:
            if best_params_list[0]['performance'] > performance:
                # 不是最佳性能，不做更新
                pass
            else:
                # 更新为最佳参数
                self.current_parameters = best_params_list[0]['parameters'].copy()
        
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'performance': performance,
            'parameters': parameters.copy(),
            'current_best': self.current_parameters.copy()
        })
    
    def get_best_parameters(self) -> Dict:
        """获取历史最佳参数"""
        best_params_list = self.performance_tracker.get_best_parameters(n=1)
        if len(best_params_list) > 0:
            return best_params_list[0]['parameters']
        else:
            return self.current_parameters.copy()
    
    def adaptive_adjust(self, performance_change: float) -> Dict:
        """
        自适应调整参数
        
        参数:
            performance_change: 性能变化（正数表示改善，负数表示恶化）
        
        返回:
            调整后的参数字典
        """
        if abs(performance_change) < 0.01:
            # 变化很小，不需要调整
            return self.current_parameters.copy()
        
        adjusted_params = {}
        for param_name, param_range in self.parameter_ranges.items():
            current_val = self.current_parameters.get(param_name,
                                                     (param_range.min_val + param_range.max_val) / 2)
            
            # 根据性能变化调整
            adjustment_range = (param_range.max_val - param_range.min_val) * self.adaptation_rate
            adjustment = performance_change * adjustment_range
            
            new_val = current_val + adjustment
            new_val = np.clip(new_val, param_range.min_val, param_range.max_val)
            adjusted_params[param_name] = float(new_val)
        
        return adjusted_params
    
    def save_state(self, filepath: str):
        """保存状态到文件"""
        state = {
            'current_parameters': self.current_parameters,
            'optimization_history': [
                {
                    'timestamp': h['timestamp'].isoformat(),
                    'performance': h['performance'],
                    'parameters': h['parameters'],
                    'current_best': h['current_best']
                }
                for h in self.optimization_history[-100:]  # 只保存最近100条
            ],
            'performance_history': self.performance_tracker.performance_history[-100:],
            'parameter_ranges': {
                name: {
                    'min_val': pr.min_val,
                    'max_val': pr.max_val,
                    'step': pr.step,
                    'param_type': pr.param_type
                }
                for name, pr in self.parameter_ranges.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self, filepath: str):
        """从文件加载状态"""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.current_parameters = state.get('current_parameters', {})
        
        # 重建参数范围
        if 'parameter_ranges' in state:
            for name, pr_data in state['parameter_ranges'].items():
                if name in self.parameter_ranges:
                    # 更新范围
                    self.parameter_ranges[name].min_val = pr_data['min_val']
                    self.parameter_ranges[name].max_val = pr_data['max_val']
                    self.parameter_ranges[name].step = pr_data.get('step')
                    self.parameter_ranges[name].param_type = pr_data.get('param_type', 'continuous')


class AutoLearningOptimizer:
    """自动学习优化器（集成参数优化和学习策略）"""
    
    def __init__(self,
                 parameter_optimizer: DynamicParameterOptimizer,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 decay_rate: float = 0.95):
        """
        初始化自动学习优化器
        
        参数:
            parameter_optimizer: 参数优化器
            learning_rate: 学习率
            momentum: 动量系数
            decay_rate: 衰减率
        """
        self.parameter_optimizer = parameter_optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay_rate = decay_rate
        
        # 动量历史
        self.parameter_velocity = {}
        for param_name in parameter_optimizer.parameter_ranges.keys():
            self.parameter_velocity[param_name] = 0.0
        
        # 学习历史
        self.learning_history = []
        
    def learn_step(self, performance: float, parameters: Dict) -> Dict:
        """
        执行一个学习步骤
        
        参数:
            performance: 当前性能
            parameters: 当前参数
        
        返回:
            下一组建议参数
        """
        # 更新性能
        self.parameter_optimizer.update_performance(performance, parameters)
        
        # 获取历史性能
        if len(self.parameter_optimizer.performance_tracker.performance_history) >= 2:
            recent_perf = self.parameter_optimizer.performance_tracker.performance_history[-1]
            prev_perf = self.parameter_optimizer.performance_tracker.performance_history[-2]
            performance_change = recent_perf - prev_perf
        else:
            performance_change = 0.0
        
        # 自适应调整
        adjusted_params = self.parameter_optimizer.adaptive_adjust(performance_change)
        
        # 应用动量和学习率
        next_params = {}
        for param_name in self.parameter_optimizer.parameter_ranges.keys():
            current_val = parameters.get(param_name, adjusted_params.get(param_name, 0))
            target_val = adjusted_params.get(param_name, current_val)
            
            # 计算更新方向
            direction = target_val - current_val
            
            # 应用动量
            self.parameter_velocity[param_name] = (
                self.momentum * self.parameter_velocity[param_name] +
                self.learning_rate * direction
            )
            
            # 更新参数
            param_range = self.parameter_optimizer.parameter_ranges[param_name]
            new_val = current_val + self.parameter_velocity[param_name]
            new_val = np.clip(new_val, param_range.min_val, param_range.max_val)
            next_params[param_name] = float(new_val)
        
        # 衰减学习率
        self.learning_rate *= self.decay_rate
        
        # 记录学习历史
        self.learning_history.append({
            'timestamp': datetime.now(),
            'performance': performance,
            'parameters': parameters.copy(),
            'next_parameters': next_params.copy(),
            'learning_rate': self.learning_rate
        })
        
        return next_params
    
    def reset_learning_rate(self, new_rate: Optional[float] = None):
        """重置学习率"""
        if new_rate is not None:
            self.learning_rate = new_rate
        else:
            # 重置为初始值的80%
            initial_rate = 0.01  # 可以改为从配置读取
            self.learning_rate = initial_rate * 0.8
    
    def get_learning_statistics(self) -> Dict:
        """获取学习统计信息"""
        if len(self.learning_history) == 0:
            return {}
        
        recent_history = self.learning_history[-50:]  # 最近50步
        performances = [h['performance'] for h in recent_history]
        
        return {
            'current_performance': performances[-1] if performances else 0,
            'average_performance': np.mean(performances) if performances else 0,
            'best_performance': max(performances) if performances else 0,
            'performance_std': np.std(performances) if len(performances) > 1 else 0,
            'learning_rate': self.learning_rate,
            'total_steps': len(self.learning_history)
        }

