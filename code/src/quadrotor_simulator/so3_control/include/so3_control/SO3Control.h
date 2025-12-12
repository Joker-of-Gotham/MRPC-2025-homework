#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

class SO3Control
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SO3Control();

  // 参数设置
  void setMass(double mass);
  void setGravity(double g);

  // 当前状态（在 world 坐标系下）
  void setPosition(const Eigen::Vector3d& position);
  void setVelocity(const Eigen::Vector3d& velocity);
  void setAcc(const Eigen::Vector3d& acc);

  /**
   * @brief 基于期望轨迹和当前状态，计算控制输出
   *
   * @param des_pos     期望位置 p_d
   * @param des_vel     期望速度 v_d
   * @param des_acc     期望加速度 a_d
   * @param des_yaw     期望偏航角 ψ_d（rad）
   * @param des_yaw_dot 期望偏航角速度 ψ̇_d（暂不使用，仅保留接口）
   * @param kx          位置误差增益向量 [Kpx, Kpy, Kpz]
   * @param kv          速度误差增益向量 [Kvx, Kvy, Kvz]
   *
   * 计算结果保存在内部变量 force_（世界系合力向量）和 orientation_（机体姿态）
   */
  void calculateControl(const Eigen::Vector3d& des_pos,
                        const Eigen::Vector3d& des_vel,
                        const Eigen::Vector3d& des_acc,
                        double des_yaw,
                        double des_yaw_dot,
                        const Eigen::Vector3d& kx,
                        const Eigen::Vector3d& kv);

  // 结果读取
  const Eigen::Vector3d& getComputedForce();
  const Eigen::Quaterniond& getComputedOrientation();

private:
  double mass_;   // 无人机质量
  double g_;      // 重力加速度

  // 当前状态（world frame）
  Eigen::Vector3d pos_;
  Eigen::Vector3d vel_;
  Eigen::Vector3d acc_;

  // 计算结果
  Eigen::Vector3d     force_;
  Eigen::Quaterniond  orientation_;
};
