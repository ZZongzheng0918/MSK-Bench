import os
from loco_mujoco.smpl.retargeting import load_retargeted_amass_trajectory, load_robot_conf_file

env_name = "MyoFullBody"
# 你的自定义文件路径（不需要加 .npz 后缀）
motion_name = "MyCustom/04/pullup_amass"

print(f"🚀 开始重定向 {motion_name} 到 {env_name} ...")

# 加载环境模型配置
robot_conf = load_robot_conf_file(env_name)

# 核心配置：一次性解决所有问题！
gmr_config = {
    "src_human": "smplh",
    "target_fps": 120,          # 👈 保持 120 帧，防止 GMR 强行降到 30 帧
    "solver": "daqp",
    "damping": 0.5,
    "offset_to_ground": False,  # 👈 核心：关闭落地对齐，保留空中的 1.5 米高度
    "use_velocity_limit": False,
    "verbose": True
}

# 调用底层函数，直接绕过官方脚本的“名字检查”拦截
traj = load_retargeted_amass_trajectory(
    env_name=env_name,
    dataset_name=motion_name,
    robot_conf=robot_conf,
    retargeting_method="gmr",
    gmr_config=gmr_config,
    clear_cache=True  # 强制覆盖旧的错误缓存
)

print("\n✅ 重定向完成！健康的缓存文件已生成！")