# -*- coding: utf-8 -*-
"""
问题二：时间-深度感知的设备推荐 (Recommend)
====================================================================
基于第一问粒子云结果，给出分时段设备推荐清单与深海触发规则

核心逻辑：
1. 从Q1提取时间-深度特征（z_mode, z10, z50, z90, P_s(t)）
2. 深度可行性过滤（z90(t) > Z_d → 不推荐）
3. 推荐评分计算（不算概率，只算适配度）
4. 分时段输出装备包（0-6h / 6-24h / 24-72h）
5. 深海触发规则（何时必须动员深海平台）

作者：MCM 2024 Problem B 推荐系统版
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import pickle
import json

# 添加第一问路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '第一问'))

# 导入第一问的类定义
try:
    from problem1_locate import (
        Particle, Mode, WaterLayer, LayerParams, SubParams, 
        RealBathymetryEnvironment, Observation
    )
    HAS_PROBLEM1 = True
except ImportError:
    HAS_PROBLEM1 = False
    print("[警告] 无法导入第一问模块")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 场景定义
# ============================================================================

class SearchScenario(str, Enum):
    """搜索场景分类"""
    SURFACE = "S"          # 近表层
    MID_WATER = "M"        # 水体中层
    SHALLOW_SEAFLOOR = "B" # 浅海底
    DEEP_SEAFLOOR = "D"    # 深海底
    COMPLEX = "C"          # 复杂地形

# ============================================================================
# 2. 设备定义（简化版：只保留推荐必要字段）
# ============================================================================

class EquipmentType(str, Enum):
    """设备类型"""
    UAV = "UAV"
    EO_IR = "EO_IR"
    PASSIVE_SONAR = "PASSIVE"
    TOWFISH_SSS = "SSS"
    MBES = "MBES"
    SHALLOW_ROV = "ROV_S"
    DEEP_TOW_SSS = "DEEP_SSS"
    DEEP_ROV = "ROV_D"
    AUV = "AUV"
    MAGNETOMETER = "MAG"
    CTD = "CTD"
    WINCH = "WINCH"
    LARS = "LARS"

@dataclass
class Equipment:
    """设备数据（推荐必要字段）"""
    name: str
    equipment_type: EquipmentType
    platform: str  # "HOST" (母船) 或 "RESCUE" (救援船)
    
    # 性能参数
    max_depth: float            # 最大工作深度 (m)，Zd
    scan_width: float           # 扫幅 Wd (m)
    platform_speed: float       # 速度 vd (m/s)
    base_efficiency: float      # 基础效率 [0,1]
    
    # 场景适配度 η_{d,s}
    scenario_efficiency: Dict[SearchScenario, float] = field(default_factory=dict)
    
    # 资源占用
    cost_buy: float = 0.0       # 采购成本 ($)
    usage_cost: float = 0.0     # 使用成本 ($/h)
    maintenance_cost: float = 0.0
    deck_space: float = 0.0     # 甲板空间 (m²)
    power_requirement: float = 0.0  # 功率 (kW)
    crew_requirement: int = 0
    
    # 时间参数
    readiness_time: float = 0.0    # 准备时间 (h)
    transit_time: float = 0.0      # 运输时间 (h) - 救援船
    
    # 依赖关系
    requires: List[str] = field(default_factory=list)
    
    # 覆盖能力指标（vd * Wd 的等级化）
    capability_index: float = 1.0

def create_equipment_database() -> List[Equipment]:
    """创建设备数据库"""
    equipment_db = []
    
    # ========== 支撑设备 ==========
    equipment_db.append(Equipment(
        name="绞车系统",
        equipment_type=EquipmentType.WINCH,
        platform="HOST",
        max_depth=0, scan_width=0, platform_speed=0, base_efficiency=0,
        cost_buy=80000, deck_space=3, power_requirement=4, crew_requirement=1,
        readiness_time=0
    ))
    
    equipment_db.append(Equipment(
        name="吊放系统LARS",
        equipment_type=EquipmentType.LARS,
        platform="HOST",
        max_depth=0, scan_width=0, platform_speed=0, base_efficiency=0,
        cost_buy=150000, deck_space=6, power_requirement=8, crew_requirement=2,
        readiness_time=0
    ))
    
    # ========== 母船设备 ==========
    # 表层搜索
    equipment_db.append(Equipment(
        name="无人机巡航系统",
        equipment_type=EquipmentType.UAV,
        platform="HOST",
        max_depth=0, scan_width=500, platform_speed=15, base_efficiency=0.85,
        scenario_efficiency={
            SearchScenario.SURFACE: 0.95,
            SearchScenario.MID_WATER: 0.0,
            SearchScenario.SHALLOW_SEAFLOOR: 0.0,
            SearchScenario.DEEP_SEAFLOOR: 0.0,
            SearchScenario.COMPLEX: 0.0
        },
        cost_buy=30000, usage_cost=200, maintenance_cost=5000,
        deck_space=2, power_requirement=1, crew_requirement=1,
        readiness_time=0.5,
        capability_index=15*500  # vd * Wd
    ))
    
    equipment_db.append(Equipment(
        name="光电/红外相机",
        equipment_type=EquipmentType.EO_IR,
        platform="HOST",
        max_depth=0, scan_width=300, platform_speed=8, base_efficiency=0.75,
        scenario_efficiency={
            SearchScenario.SURFACE: 0.80,
            SearchScenario.MID_WATER: 0.0,
            SearchScenario.SHALLOW_SEAFLOOR: 0.0,
            SearchScenario.DEEP_SEAFLOOR: 0.0,
            SearchScenario.COMPLEX: 0.0
        },
        cost_buy=20000, usage_cost=100, maintenance_cost=2000,
        deck_space=1, power_requirement=0.5, crew_requirement=1,
        readiness_time=0.25,
        capability_index=8*300
    ))
    
    # 水下声学
    equipment_db.append(Equipment(
        name="被动声学阵列",
        equipment_type=EquipmentType.PASSIVE_SONAR,
        platform="HOST",
        max_depth=1000, scan_width=2000, platform_speed=3, base_efficiency=0.70,
        scenario_efficiency={
            SearchScenario.SURFACE: 0.40,
            SearchScenario.MID_WATER: 0.85,
            SearchScenario.SHALLOW_SEAFLOOR: 0.75,
            SearchScenario.DEEP_SEAFLOOR: 0.60,
            SearchScenario.COMPLEX: 0.30
        },
        cost_buy=220000, usage_cost=500, maintenance_cost=15000,
        deck_space=8, power_requirement=5, crew_requirement=2,
        readiness_time=1.0,
        capability_index=3*2000
    ))
    
    equipment_db.append(Equipment(
        name="拖曳侧扫声呐",
        equipment_type=EquipmentType.TOWFISH_SSS,
        platform="HOST",
        max_depth=500, scan_width=150, platform_speed=2.5, base_efficiency=0.88,
        scenario_efficiency={
            SearchScenario.SURFACE: 0.0,
            SearchScenario.MID_WATER: 0.50,
            SearchScenario.SHALLOW_SEAFLOOR: 0.92,
            SearchScenario.DEEP_SEAFLOOR: 0.0,
            SearchScenario.COMPLEX: 0.60
        },
        cost_buy=120000, usage_cost=800, maintenance_cost=25000,
        deck_space=12, power_requirement=8, crew_requirement=3,
        readiness_time=2.0,
        requires=["绞车系统"],
        capability_index=2.5*150
    ))
    
    equipment_db.append(Equipment(
        name="多波束测深系统",
        equipment_type=EquipmentType.MBES,
        platform="HOST",
        max_depth=1000, scan_width=200, platform_speed=4, base_efficiency=0.80,
        scenario_efficiency={
            SearchScenario.SURFACE: 0.0,
            SearchScenario.MID_WATER: 0.40,
            SearchScenario.SHALLOW_SEAFLOOR: 0.85,
            SearchScenario.DEEP_SEAFLOOR: 0.70,
            SearchScenario.COMPLEX: 0.75
        },
        cost_buy=300000, usage_cost=600, maintenance_cost=20000,
        deck_space=5, power_requirement=6, crew_requirement=2,
        readiness_time=1.5,
        capability_index=4*200
    ))
    
    equipment_db.append(Equipment(
        name="浅水ROV",
        equipment_type=EquipmentType.SHALLOW_ROV,
        platform="HOST",
        max_depth=1000, scan_width=30, platform_speed=1, base_efficiency=0.95,
        scenario_efficiency={
            SearchScenario.SURFACE: 0.0,
            SearchScenario.MID_WATER: 0.80,
            SearchScenario.SHALLOW_SEAFLOOR: 0.95,
            SearchScenario.DEEP_SEAFLOOR: 0.0,
            SearchScenario.COMPLEX: 0.90
        },
        cost_buy=500000, usage_cost=1500, maintenance_cost=40000,
        deck_space=15, power_requirement=20, crew_requirement=4,
        readiness_time=3.0,
        requires=["吊放系统LARS"],
        capability_index=1*30
    ))
    
    equipment_db.append(Equipment(
        name="CTD温盐深剖面仪",
        equipment_type=EquipmentType.CTD,
        platform="HOST",
        max_depth=2000, scan_width=0, platform_speed=0.5, base_efficiency=0.0,
        scenario_efficiency={},
        cost_buy=15000, usage_cost=100, maintenance_cost=5000,
        deck_space=2, power_requirement=1, crew_requirement=1,
        readiness_time=0.5,
        capability_index=0  # 辅助设备
    ))
    
    # ========== 救援船设备 ==========
    equipment_db.append(Equipment(
        name="深海深拖侧扫",
        equipment_type=EquipmentType.DEEP_TOW_SSS,
        platform="RESCUE",
        max_depth=6000, scan_width=200, platform_speed=2, base_efficiency=0.90,
        scenario_efficiency={
            SearchScenario.SURFACE: 0.0,
            SearchScenario.MID_WATER: 0.0,
            SearchScenario.SHALLOW_SEAFLOOR: 0.85,
            SearchScenario.DEEP_SEAFLOOR: 0.92,
            SearchScenario.COMPLEX: 0.70
        },
        cost_buy=1200000, usage_cost=3000, maintenance_cost=100000,
        deck_space=30, power_requirement=50, crew_requirement=6,
        readiness_time=6.0, transit_time=12.0,
        capability_index=2*200
    ))
    
    equipment_db.append(Equipment(
        name="深海级ROV",
        equipment_type=EquipmentType.DEEP_ROV,
        platform="RESCUE",
        max_depth=4000, scan_width=40, platform_speed=0.8, base_efficiency=0.98,
        scenario_efficiency={
            SearchScenario.SURFACE: 0.0,
            SearchScenario.MID_WATER: 0.70,
            SearchScenario.SHALLOW_SEAFLOOR: 0.95,
            SearchScenario.DEEP_SEAFLOOR: 0.98,
            SearchScenario.COMPLEX: 0.95
        },
        cost_buy=2500000, usage_cost=5000, maintenance_cost=200000,
        deck_space=40, power_requirement=80, crew_requirement=8,
        readiness_time=8.0, transit_time=12.0,
        capability_index=0.8*40
    ))
    
    equipment_db.append(Equipment(
        name="自主水下航行器AUV",
        equipment_type=EquipmentType.AUV,
        platform="RESCUE",
        max_depth=6000, scan_width=120, platform_speed=1.5, base_efficiency=0.85,
        scenario_efficiency={
            SearchScenario.SURFACE: 0.0,
            SearchScenario.MID_WATER: 0.80,
            SearchScenario.SHALLOW_SEAFLOOR: 0.88,
            SearchScenario.DEEP_SEAFLOOR: 0.90,
            SearchScenario.COMPLEX: 0.75
        },
        cost_buy=1000000, usage_cost=2500, maintenance_cost=80000,
        deck_space=25, power_requirement=30, crew_requirement=4,
        readiness_time=4.0, transit_time=12.0,
        capability_index=1.5*120
    ))
    
    equipment_db.append(Equipment(
        name="磁力仪（补盲）",
        equipment_type=EquipmentType.MAGNETOMETER,
        platform="RESCUE",
        max_depth=3000, scan_width=100, platform_speed=2, base_efficiency=0.70,
        scenario_efficiency={
            SearchScenario.SURFACE: 0.0,
            SearchScenario.MID_WATER: 0.0,
            SearchScenario.SHALLOW_SEAFLOOR: 0.65,
            SearchScenario.DEEP_SEAFLOOR: 0.70,
            SearchScenario.COMPLEX: 0.80
        },
        cost_buy=150000, usage_cost=1200, maintenance_cost=30000,
        deck_space=10, power_requirement=10, crew_requirement=2,
        readiness_time=3.0, transit_time=12.0,
        capability_index=2*100
    ))
    
    return equipment_db

@dataclass
class PlatformConstraints:
    """平台约束"""
    budget_total: float = 3500000.0  # 总预算 ($)
    deck_space: float = 100.0        # 甲板空间 (m²)
    power: float = 200.0             # 功率 (kW)
    crew: int = 20                   # 人员

# ============================================================================
# 3. 从Q1提取时间-深度特征（核心）
# ============================================================================

def load_problem1_results(problem1_dir: str) -> Dict:
    """从第一问读取结果"""
    results_file = os.path.join(problem1_dir, "outputs_locate", "locate_results.pkl")
    
    if not os.path.exists(results_file):
        results_file = os.path.join(problem1_dir, "locate_results.pkl")
    
    if os.path.exists(results_file):
        print(f"✓ 加载第一问结果: {results_file}")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        # 转换字符串键为Enum
        str_to_enum = {
            'S': SearchScenario.SURFACE,
            'M': SearchScenario.MID_WATER,
            'B': SearchScenario.SHALLOW_SEAFLOOR,
            'D': SearchScenario.DEEP_SEAFLOOR,
            'C': SearchScenario.COMPLEX
        }
        
        if 'scenario_probs' in results:
            results['scenario_probs'] = {
                str_to_enum.get(k, k): v for k, v in results['scenario_probs'].items()
            }
        if 'scenario_areas' in results:
            results['scenario_areas'] = {
                str_to_enum.get(k, k): v for k, v in results['scenario_areas'].items()
            }
        if 'scenario_depths' in results:
            results['scenario_depths'] = {
                str_to_enum.get(k, k): v for k, v in results['scenario_depths'].items()
            }
        
        return results
    else:
        print("[警告] 未找到Q1结果，使用模拟数据")
        return create_mock_results()

def extract_time_depth_features(problem1_results: Dict) -> Dict:
    """
    从Q1结果提取时间-深度特征（新大纲核心要求）
    
    返回:
        times: 时间数组 (h)
        z_mode: 最可能深度轨迹 (m)
        z_10, z_50, z_90: 深度分位数 (m)
        scenario_probs_t: 场景概率随时间 {scenario: array}
    """
    # 检查是否有时间序列
    if 'times' in problem1_results and 'particles_history' in problem1_results:
        times_sec = np.array(problem1_results['times'])
        times_h = times_sec / 3600.0
        
        particles_history = problem1_results['particles_history']
        weights_history = problem1_results['weights_history']
        
        n_steps = len(times_h)
        z_mode = np.zeros(n_steps)
        z_10 = np.zeros(n_steps)
        z_50 = np.zeros(n_steps)
        z_90 = np.zeros(n_steps)
        
        for i in range(n_steps):
            particles = particles_history[i]
            weights = weights_history[i]
            
            # 安全地提取z值
            z_list = []
            for p in particles:
                if isinstance(p, np.ndarray) and len(p) >= 3:
                    # numpy数组格式 (x, y, z)
                    z_list.append(p[2])
                elif hasattr(p, 'z'):
                    # Particle对象格式
                    z_list.append(p.z)
                elif isinstance(p, dict) and 'z' in p:
                    # 字典格式
                    z_list.append(p['z'])
                elif isinstance(p, (list, tuple)) and len(p) >= 3:
                    # 列表/元组格式
                    z_list.append(p[2])
                else:
                    z_list.append(2800.0)  # 默认值
            z_array = np.array(z_list)
            
            # 加权平均作为mode
            z_mode[i] = np.average(z_array, weights=weights)
            z_10[i] = np.percentile(z_array, 10)
            z_50[i] = np.percentile(z_array, 50)
            z_90[i] = np.percentile(z_array, 90)
        
        # 场景概率（简化：使用最终场景分布）
        scenario_probs = problem1_results.get('scenario_probs', {})
        scenario_probs_t = {s: np.full(n_steps, p) for s, p in scenario_probs.items()}
        
        print(f"✓ 提取时间序列: {n_steps}步，{times_h[-1]:.1f}小时")
        print(f"  深度范围: z10={z_10[-1]:.0f}m, z50={z_50[-1]:.0f}m, z90={z_90[-1]:.0f}m")
        
    else:
        # 回退到最终状态
        print("  [提示] 使用Q1最终状态（无时间序列）")
        particles = problem1_results.get('particles', [])
        
        if len(particles) == 0:
            # 完全没有粒子数据，使用场景深度
            print("  [警告] 无粒子数据，使用场景深度估计")
            scenario_depths = problem1_results.get('scenario_depths', {})
            if scenario_depths:
                avg_depth = np.mean([d for d in scenario_depths.values() if d > 0])
            else:
                avg_depth = 2800.0
            
            times_h = np.array([0, 72.0])
            z_mode = np.array([avg_depth] * 2)
            z_10 = np.array([avg_depth * 0.9] * 2)
            z_50 = np.array([avg_depth] * 2)
            z_90 = np.array([avg_depth * 1.1] * 2)
        else:
            weights = problem1_results.get('weights', np.ones(len(particles)) / len(particles))
            
            # 安全地提取z值
            z_list = []
            for p in particles:
                if isinstance(p, np.ndarray) and len(p) >= 3:
                    z_list.append(p[2])
                elif hasattr(p, 'z'):
                    z_list.append(p.z)
                elif isinstance(p, dict) and 'z' in p:
                    z_list.append(p['z'])
                else:
                    z_list.append(2800.0)  # 默认值
            z_array = np.array(z_list)
            
            times_h = np.array([0, 72.0])
            z_mode = np.array([np.average(z_array, weights=weights)] * 2)
            z_10 = np.array([np.percentile(z_array, 10)] * 2)
            z_50 = np.array([np.percentile(z_array, 50)] * 2)
            z_90 = np.array([np.percentile(z_array, 90)] * 2)
        
        scenario_probs = problem1_results.get('scenario_probs', {
            SearchScenario.MID_WATER: 0.58,
            SearchScenario.DEEP_SEAFLOOR: 0.42
        })
        scenario_probs_t = {s: np.array([p, p]) for s, p in scenario_probs.items()}
    
    return {
        'times': times_h,
        'z_mode': z_mode,
        'z_10': z_10,
        'z_50': z_50,
        'z_90': z_90,
        'scenario_probs_t': scenario_probs_t,
        'scenario_probs': problem1_results.get('scenario_probs', {}),
        'scenario_areas': problem1_results.get('scenario_areas', {}),
        'scenario_depths': problem1_results.get('scenario_depths', {})
    }

def create_mock_results() -> Dict:
    """创建模拟Q1结果"""
    return {
        'particles': [],
        'weights': np.array([]),
        'scenario_probs': {
            SearchScenario.MID_WATER: 0.58,
            SearchScenario.DEEP_SEAFLOOR: 0.42
        },
        'scenario_areas': {
            SearchScenario.MID_WATER: 10293.0,
            SearchScenario.DEEP_SEAFLOOR: 1540.0
        },
        'scenario_depths': {
            SearchScenario.MID_WATER: 2520.0,
            SearchScenario.DEEP_SEAFLOOR: 3317.0
        }
    }

# ============================================================================
# 4. 深度可行性过滤（新大纲核心）
# ============================================================================

def compute_depth_feasibility(equipment: Equipment, z_90: float) -> float:
    """
    深度可行性检查
    
    若 z90 > Zd，则该设备在此深度不可行 → 返回 0
    否则 → 返回 1
    
    参数:
        equipment: 设备
        z_90: 当前时刻90%分位深度 (m)
    
    返回:
        0 或 1
    """
    if equipment.max_depth == 0:
        # 表层设备（max_depth=0）在有深度时不可行
        return 1.0 if z_90 <= 10 else 0.0
    
    # 一般设备：z90超过max_depth则不可行
    if z_90 > equipment.max_depth:
        return 0.0
    else:
        return 1.0

# ============================================================================
# 5. 推荐评分计算（新大纲核心）
# ============================================================================

def compute_recommendation_score(
    equipment: Equipment,
    t: float,
    z_90: float,
    scenario_probs: Dict[SearchScenario, float],
    time_window_start: float = 0.0
) -> float:
    """
    计算设备在时刻t的推荐评分
    
    Score_d(t) = Σ_s P_s(t) · η_{d,s} · Feasible_d(t) · Capability_d · TimePenalty_d(t)
    
    参数:
        equipment: 设备
        t: 时间 (h)
        z_90: 90%分位深度 (m)
        scenario_probs: 场景概率 {scenario: P_s}
        time_window_start: 时间窗起点 (h)
    
    返回:
        推荐评分 (无量纲)
    """
    # 1. 深度可行性
    feasible = compute_depth_feasibility(equipment, z_90)
    if feasible == 0:
        return 0.0  # 不可行直接返回0
    
    # 2. 场景适配度加权
    scenario_score = 0.0
    for scenario, P_s in scenario_probs.items():
        eta = equipment.scenario_efficiency.get(scenario, 0.0)
        scenario_score += P_s * eta
    
    # 3. 覆盖能力指标（归一化）
    capability = equipment.capability_index / 10000.0  # 简单归一化
    
    # 4. 时间惩罚（到位延迟）
    arrival_time = time_window_start + equipment.readiness_time + equipment.transit_time
    if t < arrival_time:
        # 还未到位，权重降低
        time_penalty = 0.5
    else:
        time_penalty = 1.0
    
    # 5. 综合评分
    score = scenario_score * feasible * capability * time_penalty
    
    return score

def compute_time_averaged_score(
    equipment: Equipment,
    times: np.ndarray,
    z_90_series: np.ndarray,
    scenario_probs_t: Dict[SearchScenario, np.ndarray],
    time_start: float,
    time_end: float
) -> float:
    """
    计算时间段内的平均推荐评分
    
    参数:
        equipment: 设备
        times: 时间数组 (h)
        z_90_series: z90时间序列 (m)
        scenario_probs_t: 场景概率时间序列
        time_start, time_end: 时间窗 (h)
    
    返回:
        平均评分
    """
    # 筛选时间窗内的点
    mask = (times >= time_start) & (times <= time_end)
    if not np.any(mask):
        # 如果没有时间点在窗内，使用端点
        t_sample = time_start
        idx = np.argmin(np.abs(times - t_sample))
        z_90 = z_90_series[idx]
        scenario_probs = {s: probs[idx] for s, probs in scenario_probs_t.items()}
        return compute_recommendation_score(equipment, t_sample, z_90, scenario_probs, time_start)
    
    times_window = times[mask]
    z_90_window = z_90_series[mask]
    
    scores = []
    for i, t in enumerate(times_window):
        z_90 = z_90_window[i]
        
        # 提取当前时刻场景概率
        idx = np.where(times == t)[0][0]
        scenario_probs = {s: probs[idx] for s, probs in scenario_probs_t.items()}
        
        score = compute_recommendation_score(equipment, t, z_90, scenario_probs, time_start)
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0

# ============================================================================
# 6. 分时间段推荐（新大纲核心）
# ============================================================================

def recommend_by_time_phase(
    equipment_db: List[Equipment],
    time_features: Dict,
    constraints: PlatformConstraints,
    top_k: int = 5
) -> Dict:
    """
    分三个时间段输出推荐装备包
    
    时间段:
        Phase 1: 0-6h   - 快速响应包（表层侦察）
        Phase 2: 6-24h  - 主力搜索包（声学/成像）
        Phase 3: 24-72h - 长尾与深海包（深海能力）
    
    返回:
        {
            'phase1': {'host': [...], 'rescue': [...]},
            'phase2': {...},
            'phase3': {...}
        }
    """
    phases = [
        {'name': 'Phase 1 (0-6h)', 'start': 0.0, 'end': 6.0},
        {'name': 'Phase 2 (6-24h)', 'start': 6.0, 'end': 24.0},
        {'name': 'Phase 3 (24-72h)', 'start': 24.0, 'end': 72.0}
    ]
    
    times = time_features['times']
    z_90_series = time_features['z_90']
    scenario_probs_t = time_features['scenario_probs_t']
    
    results = {}
    
    for phase in phases:
        print(f"\n{'='*60}")
        print(f"  {phase['name']}: 快速响应包" if 'Phase 1' in phase['name'] else f"  {phase['name']}: 主力搜索包" if 'Phase 2' in phase['name'] else f"  {phase['name']}: 长尾与深海包")
        print(f"{'='*60}")
        
        # 为每个设备计算该时段平均评分
        scored_equipment = []
        for eq in equipment_db:
            avg_score = compute_time_averaged_score(
                eq, times, z_90_series, scenario_probs_t,
                phase['start'], phase['end']
            )
            scored_equipment.append((eq, avg_score))
        
        # 按平台分类并排序
        host_equipment = [(eq, score) for eq, score in scored_equipment if eq.platform == "HOST" and score > 0]
        rescue_equipment = [(eq, score) for eq, score in scored_equipment if eq.platform == "RESCUE" and score > 0]
        
        host_equipment.sort(key=lambda x: x[1], reverse=True)
        rescue_equipment.sort(key=lambda x: x[1], reverse=True)
        
        # 选择Top-K（满足约束）
        host_selected = select_equipment_with_constraints(host_equipment, constraints, top_k)
        rescue_selected = select_equipment_with_constraints(rescue_equipment, constraints, top_k)
        
        # 打印推荐结果
        print(f"\n【母船装备 Top-{min(top_k, len(host_selected))}】")
        for i, (eq, score) in enumerate(host_selected[:top_k], 1):
            reason = get_recommendation_reason(eq, time_features, phase['start'], phase['end'])
            print(f"  {i}. {eq.name:<15} | Score={score:.3f} | {reason}")
        
        print(f"\n【救援船装备 Top-{min(top_k, len(rescue_selected))}】")
        for i, (eq, score) in enumerate(rescue_selected[:top_k], 1):
            reason = get_recommendation_reason(eq, time_features, phase['start'], phase['end'])
            print(f"  {i}. {eq.name:<15} | Score={score:.3f} | {reason}")
        
        results[phase['name']] = {
            'host': host_selected[:top_k],
            'rescue': rescue_selected[:top_k],
            'time_range': (phase['start'], phase['end'])
        }
    
    return results

def select_equipment_with_constraints(
    scored_equipment: List[Tuple[Equipment, float]],
    constraints: PlatformConstraints,
    max_count: int
) -> List[Tuple[Equipment, float]]:
    """
    从排序列表中选择设备（考虑约束与依赖）
    
    简化逻辑：
    1. 按评分从高到低
    2. 检查资源约束（预算/甲板/功率/人员）
    3. 检查依赖关系（自动补齐支撑设备）
    
    返回:
        [(equipment, score), ...]
    """
    selected = []
    selected_names = set()
    
    budget_used = 0.0
    deck_used = 0.0
    power_used = 0.0
    crew_used = 0
    
    for eq, score in scored_equipment:
        # 检查是否已选
        if eq.name in selected_names:
            continue
        
        # 检查资源约束
        total_cost = eq.cost_buy + eq.maintenance_cost
        if (budget_used + total_cost > constraints.budget_total or
            deck_used + eq.deck_space > constraints.deck_space or
            power_used + eq.power_requirement > constraints.power or
            crew_used + eq.crew_requirement > constraints.crew):
            continue
        
        # 检查依赖（简化：仅检查，不自动补齐）
        deps_satisfied = all(dep in selected_names for dep in eq.requires)
        if not deps_satisfied and eq.requires:
            # 尝试补齐依赖
            for dep_name in eq.requires:
                if dep_name not in selected_names:
                    # 查找依赖设备（从原始列表）
                    dep_eq = next((e for e, s in scored_equipment if e.name == dep_name), None)
                    if dep_eq:
                        dep_cost = dep_eq.cost_buy + dep_eq.maintenance_cost
                        if (budget_used + dep_cost <= constraints.budget_total and
                            deck_used + dep_eq.deck_space <= constraints.deck_space):
                            selected.append((dep_eq, 0.0))  # 依赖设备评分为0（辅助）
                            selected_names.add(dep_name)
                            budget_used += dep_cost
                            deck_used += dep_eq.deck_space
                            power_used += dep_eq.power_requirement
                            crew_used += dep_eq.crew_requirement
        
        # 添加设备
        selected.append((eq, score))
        selected_names.add(eq.name)
        budget_used += total_cost
        deck_used += eq.deck_space
        power_used += eq.power_requirement
        crew_used += eq.crew_requirement
        
        if len([s for s in selected if s[1] > 0]) >= max_count:
            break
    
    return selected

def get_recommendation_reason(
    equipment: Equipment,
    time_features: Dict,
    time_start: float,
    time_end: float
) -> str:
    """生成推荐理由（一句话）"""
    # 获取该时段的平均深度
    times = time_features['times']
    z_90 = time_features['z_90']
    mask = (times >= time_start) & (times <= time_end)
    avg_z90 = np.mean(z_90[mask]) if np.any(mask) else z_90[-1]
    
    # 获取主导场景
    scenario_probs = time_features['scenario_probs']
    dominant_scenario = max(scenario_probs.items(), key=lambda x: x[1])
    scenario_name = {
        SearchScenario.SURFACE: "表层",
        SearchScenario.MID_WATER: "中层",
        SearchScenario.SHALLOW_SEAFLOOR: "浅海底",
        SearchScenario.DEEP_SEAFLOOR: "深海底",
        SearchScenario.COMPLEX: "复杂地形"
    }.get(dominant_scenario[0], "未知")
    
    # 深度可行性
    if equipment.max_depth > 0:
        depth_ok = "✓深度可行" if avg_z90 <= equipment.max_depth else "✗深度超限"
    else:
        depth_ok = "表层设备"
    
    # 场景适配
    eta = equipment.scenario_efficiency.get(dominant_scenario[0], 0.0)
    
    return f"{scenario_name}(P={dominant_scenario[1]:.1%}) | η={eta:.2f} | {depth_ok} (z90≈{avg_z90:.0f}m)"

# ============================================================================
# 7. 深海触发规则（新大纲核心）
# ============================================================================

def compute_deep_sea_trigger(
    time_features: Dict,
    Z_host_max: float = 1500.0,
    lookahead_hours: float = 12.0
) -> Dict:
    """
    深海平台动员触发规则
    
    规则:
        若在未来T小时内，max_t z90(t) > Z_host_max
        ⇒ 立即动员救援船深海装备
    
    参数:
        time_features: 时间-深度特征
        Z_host_max: 母船最大深度能力 (m)
        lookahead_hours: 前瞻时间窗 (h)
    
    返回:
        {
            'triggered': bool,
            'trigger_time': float (h),
            'trigger_depth': float (m),
            'reason': str
        }
    """
    times = time_features['times']
    z_90 = time_features['z_90']
    
    print(f"\n{'='*60}")
    print(f"  深海触发规则评估")
    print(f"{'='*60}")
    print(f"  母船最大深度能力: {Z_host_max}m")
    print(f"  前瞻窗口: {lookahead_hours}h")
    
    for i, t in enumerate(times):
        # 查看未来lookahead_hours内的最大深度
        future_mask = (times >= t) & (times <= t + lookahead_hours)
        if not np.any(future_mask):
            continue
        
        max_z90_future = np.max(z_90[future_mask])
        
        if max_z90_future > Z_host_max:
            print(f"\n  ✓ 触发深海动员！")
            print(f"    触发时间: t={t:.1f}h")
            print(f"    预测最大深度: z90={max_z90_future:.0f}m > {Z_host_max}m")
            print(f"    建议: 立即动员深海深拖侧扫、深海ROV、AUV等装备")
            
            return {
                'triggered': True,
                'trigger_time': t,
                'trigger_depth': max_z90_future,
                'reason': f"未来{lookahead_hours}h内深度将超过母船能力"
            }
    
    print(f"\n  ✗ 未触发深海动员")
    print(f"    最大深度: z90_max={np.max(z_90):.0f}m < {Z_host_max}m")
    
    return {
        'triggered': False,
        'trigger_time': None,
        'trigger_depth': np.max(z_90),
        'reason': "母船装备足以覆盖预测深度范围"
    }

# ============================================================================
# 8. 可视化（深度-设备可行性对照图）
# ============================================================================

def plot_depth_equipment_feasibility(
    time_features: Dict,
    equipment_db: List[Equipment],
    output_dir: str = "outputs_prepare"
):
    """
    绘制深度—设备可行性对照图
    
    显示:
        - z50/z90随时间变化
        - 各设备max_depth水平线
        - 一眼看懂为什么选/不选
    """
    os.makedirs(output_dir, exist_ok=True)
    
    times = time_features['times']
    z_10 = time_features['z_10']
    z_50 = time_features['z_50']
    z_90 = time_features['z_90']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制深度不确定性区间
    ax.fill_between(times, z_10, z_90, alpha=0.2, color='steelblue', label='10%-90%深度区间')
    ax.plot(times, z_50, 'b-', linewidth=2, label='z₅₀ (中位数)')
    ax.plot(times, z_90, 'r--', linewidth=1.5, label='z₉₀ (90%分位)')
    
    # 绘制设备最大深度线
    host_equipment = [eq for eq in equipment_db if eq.platform == "HOST" and eq.max_depth > 0]
    rescue_equipment = [eq for eq in equipment_db if eq.platform == "RESCUE" and eq.max_depth > 0]
    
    colors_host = plt.cm.Greens(np.linspace(0.4, 0.9, len(host_equipment)))
    colors_rescue = plt.cm.Reds(np.linspace(0.4, 0.9, len(rescue_equipment)))
    
    for i, eq in enumerate(host_equipment):
        ax.axhline(eq.max_depth, color=colors_host[i], linestyle=':', linewidth=1.5, 
                   label=f'{eq.name} (母船, {eq.max_depth}m)', alpha=0.7)
    
    for i, eq in enumerate(rescue_equipment):
        ax.axhline(eq.max_depth, color=colors_rescue[i], linestyle='-.', linewidth=1.5,
                   label=f'{eq.name} (救援船, {eq.max_depth}m)', alpha=0.7)
    
    ax.set_xlabel('时间 (小时)', fontsize=12, fontweight='bold')
    ax.set_ylabel('深度 (米)', fontsize=12, fontweight='bold')
    ax.set_title('深度演化 vs 设备最大工作深度\n（一眼看懂为什么选/不选）', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "depth_equipment_feasibility.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\n✓ 深度-设备可行性图已保存: {filepath}")
    plt.close()

def plot_recommendation_summary(
    recommendations: Dict,
    output_dir: str = "outputs_prepare"
):
    """绘制推荐汇总图（3时段 × 2平台）"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('分时段设备推荐汇总', fontsize=16, fontweight='bold')
    
    phase_names = list(recommendations.keys())
    
    for row, phase_name in enumerate(phase_names):
        phase_data = recommendations[phase_name]
        
        # 母船
        ax_host = axes[row, 0]
        host_eq = phase_data['host']
        if host_eq:
            names = [eq.name for eq, score in host_eq]
            scores = [score for eq, score in host_eq]
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(names)))
            ax_host.barh(names, scores, color=colors, edgecolor='white', linewidth=2)
            ax_host.set_xlabel('推荐评分', fontweight='bold')
            ax_host.set_title(f'{phase_name} - 母船装备', fontweight='bold')
            ax_host.grid(axis='x', alpha=0.3)
        
        # 救援船
        ax_rescue = axes[row, 1]
        rescue_eq = phase_data['rescue']
        if rescue_eq:
            names = [eq.name for eq, score in rescue_eq]
            scores = [score for eq, score in rescue_eq]
            colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(names)))
            ax_rescue.barh(names, scores, color=colors, edgecolor='white', linewidth=2)
            ax_rescue.set_xlabel('推荐评分', fontweight='bold')
            ax_rescue.set_title(f'{phase_name} - 救援船装备', fontweight='bold')
            ax_rescue.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "recommendation_summary.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ 推荐汇总图已保存: {filepath}")
    plt.close()

# ============================================================================
# 9. 主函数
# ============================================================================

def main():
    """主流程"""
    print("\n" + "="*70)
    print("  问题二：时间-深度感知的设备推荐系统 (v2.0)")
    print("="*70)
    
    # 1. 加载Q1结果
    print("\n[步骤1] 加载第一问结果")
    problem1_dir = os.path.join(os.path.dirname(__file__), '..', '第一问')
    problem1_results = load_problem1_results(problem1_dir)
    
    # 2. 提取时间-深度特征
    print("\n[步骤2] 提取时间-深度特征")
    time_features = extract_time_depth_features(problem1_results)
    
    # 3. 创建设备库
    print("\n[步骤3] 加载设备数据库")
    equipment_db = create_equipment_database()
    print(f"  ✓ 加载 {len(equipment_db)} 件设备")
    
    # 4. 平台约束
    constraints = PlatformConstraints()
    print(f"\n[步骤4] 平台约束")
    print(f"  总预算: ${constraints.budget_total:,.0f}")
    print(f"  甲板空间: {constraints.deck_space}m²")
    print(f"  功率: {constraints.power}kW")
    print(f"  人员: {constraints.crew}人")
    
    # 5. 分时段推荐
    print("\n[步骤5] 分时段设备推荐")
    recommendations = recommend_by_time_phase(equipment_db, time_features, constraints, top_k=5)
    
    # 6. 深海触发规则
    print("\n[步骤6] 深海触发规则评估")
    trigger_result = compute_deep_sea_trigger(time_features, Z_host_max=1500.0, lookahead_hours=12.0)
    
    # 7. 输出资源汇总
    print("\n[步骤7] 资源占用汇总")
    for phase_name, phase_data in recommendations.items():
        print(f"\n  {phase_name}:")
        for platform in ['host', 'rescue']:
            equipment_list = phase_data[platform]
            if equipment_list:
                total_cost = sum(eq.cost_buy + eq.maintenance_cost for eq, score in equipment_list)
                total_deck = sum(eq.deck_space for eq, score in equipment_list)
                total_power = sum(eq.power_requirement for eq, score in equipment_list)
                total_crew = sum(eq.crew_requirement for eq, score in equipment_list)
                
                platform_name = "母船" if platform == 'host' else "救援船"
                print(f"    {platform_name}: 成本=${total_cost:,.0f}, 甲板={total_deck}m², 功率={total_power}kW, 人员={total_crew}人")
    
    # 8. 可视化
    print("\n[步骤8] 生成可视化")
    output_dir = "outputs_prepare"
    plot_depth_equipment_feasibility(time_features, equipment_db, output_dir)
    plot_recommendation_summary(recommendations, output_dir)
    
    # 9. 保存结果
    print("\n[步骤9] 保存结果")
    
    # 转换scenario_probs_t中的numpy数组
    scenario_probs_t_serializable = {}
    for k, v in time_features.get('scenario_probs_t', {}).items():
        key_str = k.value if hasattr(k, 'value') else str(k)
        scenario_probs_t_serializable[key_str] = v.tolist() if isinstance(v, np.ndarray) else v
    
    results = {
        'time_features': {
            'times': time_features['times'].tolist(),
            'z_mode': time_features['z_mode'].tolist(),
            'z_10': time_features['z_10'].tolist(),
            'z_50': time_features['z_50'].tolist(),
            'z_90': time_features['z_90'].tolist(),
            'scenario_probs_t': scenario_probs_t_serializable,
            'scenario_probs': {(k.value if hasattr(k, 'value') else str(k)): v 
                             for k, v in time_features.get('scenario_probs', {}).items()},
            'scenario_areas': {(k.value if hasattr(k, 'value') else str(k)): v 
                             for k, v in time_features.get('scenario_areas', {}).items()},
            'scenario_depths': {(k.value if hasattr(k, 'value') else str(k)): v 
                              for k, v in time_features.get('scenario_depths', {}).items()}
        },
        'recommendations': {
            phase: {
                platform: [(eq.name, float(score)) for eq, score in eq_list]
                for platform, eq_list in data.items() if platform in ['host', 'rescue']
            }
            for phase, data in recommendations.items()
        },
        'deep_trigger': trigger_result
    }
    
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "recommendation_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果已保存: {result_file}")
    
    print("\n" + "="*70)
    print("  推荐系统运行完成！")
    print("="*70)

if __name__ == "__main__":
    main()
