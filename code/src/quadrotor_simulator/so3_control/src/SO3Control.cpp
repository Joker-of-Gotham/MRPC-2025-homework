#include <iostream>
#include <fstream>

#include <ros/ros.h>
#include <so3_control/SO3Control.h>

namespace
{
// 简单的调试日志，用于画 tracking 曲线（保持你原来的需求）
std::ofstream g_ctrl_data_file;
} // namespace

SO3Control::SO3Control()
  : mass_(0.5)
  , g_(9.81)
{
  pos_.setZero();
  vel_.setZero();
  acc_.setZero();
  force_.setZero();
  orientation_.setIdentity();
}

void SO3Control::setMass(const double mass)
{
  mass_ = mass;
}

void SO3Control::setGravity(const double g)
{
  g_ = g;
}

void SO3Control::setPosition(const Eigen::Vector3d& position)
{
  pos_ = position;
}

void SO3Control::setVelocity(const Eigen::Vector3d& velocity)
{
  vel_ = velocity;
}

void SO3Control::setAcc(const Eigen::Vector3d& acc)
{
  acc_ = acc;
}

void SO3Control::calculateControl(const Eigen::Vector3d& des_pos,
                                  const Eigen::Vector3d& des_vel,
                                  const Eigen::Vector3d& des_acc,
                                  const double            des_yaw,
                                  const double /*des_yaw_dot*/,
                                  const Eigen::Vector3d&  kx,
                                  const Eigen::Vector3d&  kv)
{
  // 打开日志文件（只打开一次）
  if (!g_ctrl_data_file.is_open())
  {
    g_ctrl_data_file.open(
      "/home/stuwork/MRPC-2025-homework/code/src/quadrotor_simulator/"
      "so3_control/src/control_data.txt",
      std::ios::out | std::ios::trunc);

    if (!g_ctrl_data_file)
    {
      ROS_WARN_STREAM("SO3Control: failed to open control_data.txt for logging");
    }
  }

  // 位置 / 速度误差
  const Eigen::Vector3d e_p = des_pos - pos_;
  const Eigen::Vector3d e_v = des_vel - vel_;

  // 对应 README 中公式：a_cmd = a_d + Kp*(p_d - p) + Kv*(v_d - v) + g*e3
  Eigen::Vector3d a_cmd =
    des_acc +
    kx.asDiagonal() * e_p +
    kv.asDiagonal() * e_v +
    g_ * Eigen::Vector3d(0.0, 0.0, 1.0);

  // 若 a_cmd 非法（极小或 NaN），则退化为纯重力
  if (!a_cmd.allFinite() || a_cmd.norm() < 1e-4)
  {
    a_cmd = g_ * Eigen::Vector3d(0.0, 0.0, 1.0);
  }

  // 世界系合力向量（注意：quadrotor_simulator 会把这个 force_ 转到机体系）
  force_ = mass_ * a_cmd;

  // ==================== 姿态构造 ====================

  // 期望推力方向 b3（世界坐标系）
  Eigen::Vector3d b3c = a_cmd.normalized();

  // 期望机体 x 轴方向（仅在水平面内，和 des_yaw 对齐）
  const double cy = std::cos(des_yaw);
  const double sy = std::sin(des_yaw);
  Eigen::Vector3d b1d(cy, sy, 0.0);

  // 保证 b2, b1 与 b3 正交
  Eigen::Vector3d b2c = b3c.cross(b1d);
  const double b2_norm = b2c.norm();
  if (b2_norm < 1e-4)
  {
    // 如果 b3 和 b1d 近乎平行，稍微扰动一下 b1d
    b1d = Eigen::Vector3d(cy * 0.99, sy * 0.99, 0.1).normalized();
    b2c = b3c.cross(b1d);
  }
  b2c.normalize();
  Eigen::Vector3d b1c = b2c.cross(b3c);
  b1c.normalize();

  Eigen::Matrix3d R;
  R.col(0) = b1c;
  R.col(1) = b2c;
  R.col(2) = b3c;

  orientation_ = Eigen::Quaterniond(R);

  // ==================== 简单日志 ====================
  if (g_ctrl_data_file)
  {
    const ros::Time t_now = ros::Time::now();
    g_ctrl_data_file << t_now.toSec() << " "
                     << des_pos.x() << " " << des_pos.y() << " " << des_pos.z() << " "
                     << pos_.x()   << " " << pos_.y()   << " " << pos_.z()   << " "
                     << std::endl;
  }
}

const Eigen::Vector3d& SO3Control::getComputedForce(void)
{
  return force_;
}

const Eigen::Quaterniond& SO3Control::getComputedOrientation(void)
{
  return orientation_;
}
