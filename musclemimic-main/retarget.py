import os
import numpy as np
import logging

# 1. 设定环境变量
os.environ["AMASS_PATH"] = "/home/pangjr/mengtao.o/.musclemimic/AMASS"

import loco_mujoco.smpl.retargeting
from loco_mujoco.smpl.retargeting import fit_gmr_motion, load_robot_conf_file

# =========================================================================
# 🌟 绝杀黑科技：异常拦截法 🌟
# 我们故意抛出一个异常，把 qpos 数组安全地“带”出来，直接跳过所有会触发段错误的 C 代码！
# =========================================================================
class InterceptSuccess(Exception):
    def __init__(self, qpos):
        self.qpos = qpos

def safe_intercept(qpos, fps, free_joint_name, model):
    print("\n🔧 成功在段错误(Segfault)发生前拦截到 89 维位姿数据！")
    raise InterceptSuccess(qpos)

# 暴力替换原本会崩溃的底层函数
loco_mujoco.smpl.retargeting._compute_qvel_from_qpos = safe_intercept
# =========================================================================

def main():
    logger = logging.getLogger()
    env_name = "MyoFullBody"
    robot_conf = load_robot_conf_file(env_name)
    motion_data = "/home/pangjr/mengtao.o/.musclemimic/AMASS/KIT/316/13_35_poses.npz"
    gmr_config = {"target_fps": 100, "offset_to_ground": False}
    
    print("🚀 开始安全重定向 (拦截模式已启动)...")
    try:
        # 运行官方重定向，它算完 89 维数据后，会被我们的 Exception 强行打断
        fit_gmr_motion(env_name, robot_conf, motion_data, logger, gmr_config)
    except InterceptSuccess as e:
        # 捕获成功！提取出极其宝贵的 89 维 qpos
        final_qpos = e.qpos
        
        # 因为你的 NavigateStairEnvV0 里的 NPZReferenceMotion 会自己算速度，所以这里塞个空数组就行
        dummy_qvel = np.zeros((final_qpos.shape[0], 88))
        
        # 存储路径
        out_path = "/home/pangjr/mengtao.o/stair_prior_89d.npz"
        np.savez(
            out_path,
            states=final_qpos,
            qvel=dummy_qvel
        )
        
        print(f"\n✅ 绝杀成功！完美绕过了所有底层崩溃！")
        print(f"🎉 专为上楼梯环境打造的先验文件已保存至: {out_path}")
        print("现在，你可以把这个路径填入 NavigateStairEnvV0 中，直接开启残差训练了！")

if __name__ == "__main__":
    main()