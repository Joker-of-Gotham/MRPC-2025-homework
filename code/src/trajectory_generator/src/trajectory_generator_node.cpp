#include <algorithm>
#include <fstream>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <quadrotor_msgs/PolynomialTrajectory.h>
#include <quadrotor_msgs/PositionCommand.h>  // *** 新增
#include <random>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// Useful customized headers
#include "Astar_searcher.h"
#include "backward.hpp"
#include "trajectory_generator_waypoint.h"

using namespace std;
using namespace Eigen;

TrajectoryGeneratorWaypoint *_trajGene = new TrajectoryGeneratorWaypoint();
Astarpath *_astar_path_finder        = new Astarpath();

// Set the obstacle map
double _resolution, _inv_resolution, _path_resolution;
double _x_size, _y_size, _z_size;
Vector3d _map_lower, _map_upper;
int _max_x_id, _max_y_id, _max_z_id;
int demox = 0;
ros::Time start_record_time;
ros::Time finish_record_time;

// Param from launch file
double _vis_traj_width;
double _Vel, _Acc;
int _dev_order, _min_order;

// ros related
ros::Subscriber _map_sub, _pts_sub, _odom_sub;
ros::Publisher _traj_vis_pub, _traj_before_vis_pub, _traj_pub, _path_vis_pub,
    _astar_path_vis_pub;
ros::Publisher _pos_cmd_pub;  // *** 新增：位置指令发布器

// for planning
Vector3d odom_pt, odom_vel, start_pt, target_pt, start_vel;
int _poly_num1D;
MatrixXd _polyCoeff;
VectorXd _polyTime;
double time_duration;
ros::Time time_traj_start;
bool has_odom   = false;
bool has_target = false;
bool has_traj   = false;  // *** 当前是否有可用轨迹

// for replanning
enum STATE
{
  INIT,
  WAIT_TARGET,
  GEN_NEW_TRAJ,
  EXEC_TRAJ,
  REPLAN_TRAJ,
  EMER_STOP
};
STATE exec_state;
double no_replan_thresh, replan_thresh;
ros::Timer _exec_timer;
ros::Timer _safety_timer_;
void execCallback(const ros::TimerEvent &e);
bool cracked;
int time_Count;
int target_Count;
ros::Timer _record_timer_;

// declare
void changeState(STATE new_state, string pos_call);
void printState();
void visTrajectory(MatrixXd polyCoeff, VectorXd time);
void visTrajectory_before(MatrixXd polyCoeff, VectorXd time);
void visPath(MatrixXd nodes);
void visPathA(MatrixXd nodes);
bool trajOptimization(const Eigen::MatrixXd &path, const Eigen::MatrixXd &path_raw);
void rcvOdomCallback(const nav_msgs::Odometry::ConstPtr &odom);
void rcvWaypointsCallBack(const nav_msgs::Path &wp);
void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 &pointcloud_map);
void trajPublish(MatrixXd polyCoeff, VectorXd time);
bool trajGeneration();
VectorXd timeAllocation(MatrixXd Path);
Vector3d getPos(double t_cur);
Vector3d getVel(double t_cur);
Vector3d getAcc(double t_cur);
bool evalTraj(double t_cur, Vector3d *pos, Vector3d *vel, Vector3d *acc);
void issafe(const ros::TimerEvent &e);
void publishPositionCommand(double t_cur);  // *** 新增：按当前时间发布位置指令
bool trajUnsafeAhead(double t_now, double horizon_s);

// change the state to the new state
void changeState(STATE new_state, string pos_call)
{
  string state_str[6] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ",
                         "EXEC_TRAJ", "REPLAN_TRAJ", "EMER_STOP"};
  int    pre_s        = int(exec_state);
  exec_state          = new_state;
  cout << "[" + pos_call + "]: from " + state_str[pre_s] + " to " +
              state_str[int(new_state)]
       << endl;
}

void printState()
{
  string state_str[6] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ",
                         "EXEC_TRAJ", "REPLAN_TRAJ", "EMER_STOP"};
  cout << "[Clock]: state: " + state_str[int(exec_state)] << endl;
}

void rcvOdomCallback(const nav_msgs::Odometry::ConstPtr &odom)
{
  odom_pt(0) = odom->pose.pose.position.x;
  odom_pt(1) = odom->pose.pose.position.y;
  odom_pt(2) = odom->pose.pose.position.z;

  odom_vel(0) = odom->twist.twist.linear.x;
  odom_vel(1) = odom->twist.twist.linear.y;
  odom_vel(2) = odom->twist.twist.linear.z;

  has_odom = true;
}

// Control the State changes
void execCallback(const ros::TimerEvent &e)
{
  static int num = 0;
  // 规划失败的简单退避，避免失败后紧密循环占满 CPU
  static bool      plan_failed_recently = false;
  static ros::Time last_plan_fail_t(0.0);
  num++;
  if (num == 100)
  {
    printState();
    if (!has_odom)
      cout << "no odom." << endl;
    if (!has_target)
      cout << "wait for goal." << endl;
    num = 0;
  }

  switch (exec_state)
  {
    case INIT:
    {
      if (!has_odom)
        return;
      if (!has_target)
        return;
      changeState(WAIT_TARGET, "STATE");  // odom
      break;
    }

    case WAIT_TARGET:
    {
      if (!has_target)
        return;

      // 若刚刚规划失败，稍作等待再重试（例如地图/点云尚未刷新）
      if (plan_failed_recently &&
          (ros::Time::now() - last_plan_fail_t).toSec() < 0.5)
        return;

      plan_failed_recently = false;
      changeState(GEN_NEW_TRAJ, "STATE");
      break;
    }

    case GEN_NEW_TRAJ:
    {
      bool success = trajGeneration();  // 路径点搜索程序从这里进入
      if (success)
      {
        changeState(EXEC_TRAJ, "STATE");
      }
      else
      {
        // 关键修复：不要因为一次规划失败就丢掉目标点，否则会出现“停在空旷处卡死”
        // 这里进入 WAIT_TARGET 并保持 has_target=true，后续会带退避地自动重试规划。
        plan_failed_recently = true;
        last_plan_fail_t = ros::Time::now();
        has_traj = false;
        changeState(WAIT_TARGET, "STATE");
      }
      break;
    }

    case EXEC_TRAJ:
    {
      if (!has_traj)
        return;

      time_Count++;
      ros::Time time_now = ros::Time::now();
      double    t_cur    = (time_now - time_traj_start).toSec();
      double    t_replan = ros::Duration(1, 0).toSec();
      t_cur              = min(time_duration, t_cur);

      // *** 在这里实时发送 PositionCommand
      publishPositionCommand(t_cur);

      // 关键修复：用最新点云对“未来一小段轨迹”做碰撞预测，若即将穿障则立即触发重规划
      // 这解决“障碍已在视野内但仍沿旧轨迹撞上去”的问题。
      static ros::Time last_force_replan_t(0.0);
      if ((time_now - last_force_replan_t).toSec() > 0.20)  // 5Hz 上限，避免抖动
      {
        double horizon_s = 1.2;  // 预测 1.2s 内的轨迹是否安全（可按速度自适应）
        ros::param::param("~collision_check_horizon", horizon_s, 1.2);
        if (trajUnsafeAhead(t_cur, horizon_s))
        {
          last_force_replan_t = time_now;
          changeState(REPLAN_TRAJ, "COLLISION_AHEAD");
          return;
        }
      }

      // 关键修复：到达目标附近且速度足够小 -> 直接进入悬停，避免目标附近“转圈乱跑”
      double goal_tol = 0.45;
      double vel_tol  = 0.25;
      ros::param::param("~goal_tolerance", goal_tol, 0.45);
      ros::param::param("~goal_vel_tolerance", vel_tol, 0.25);
      if ((target_pt - odom_pt).norm() < goal_tol && odom_vel.norm() < vel_tol)
      {
        has_traj   = false;
        has_target = false;
        changeState(WAIT_TARGET, "GOAL_REACHED");
        return;
      }

      if (t_cur > time_duration - 1e-2)
      {
        // 关键修复：只有真正到达目标附近才清空 has_target。
        // 若本段轨迹提前结束但距离目标仍远，则立即进入下一次规划，避免“卡死等待新目标”。
        const double dist_to_goal = (target_pt - odom_pt).norm();

        has_traj = false;
        if (dist_to_goal < goal_tol && odom_vel.norm() < vel_tol)
        {
          has_target = false;
          changeState(WAIT_TARGET, "STATE");
        }
        else
        {
          // 继续向同一个 target_pt 前进
          has_target = true;
          start_pt   = odom_pt;
          start_vel  = odom_vel;
          changeState(GEN_NEW_TRAJ, "STATE");
        }
        return;
      }
      else if ((target_pt - odom_pt).norm() < no_replan_thresh)
      {
        return;
      }
      else if ((start_pt - odom_pt).norm() < replan_thresh)
      {
        return;
      }
      else if (t_cur < t_replan)
      {
        return;
      }
      else
      {
        changeState(REPLAN_TRAJ, "STATE");
      }
      break;
    }

    case REPLAN_TRAJ:
    {
      if (!has_traj)
        return;

      ros::Time time_now = ros::Time::now();
      double    t_cur    = (time_now - time_traj_start).toSec();
      // 预估一点点前向时间，补偿重规划延迟（原实现 ros::Duration(0,50) 仅 50ns，几乎为 0）
      const double t_delta = 0.05;  // 50ms
      t_cur = std::max(0.0, std::min(time_duration, t_cur + t_delta));

      // 用真实里程计作为重规划边界条件：减少“停-走-停”与由于预测误差带来的抖动
      start_pt  = odom_pt;
      start_vel = odom_vel;

      // *** REPLAN 时也先发一次当前 PositionCommand，防止控制器断指令
      publishPositionCommand(t_cur);

      bool success = trajGeneration();
      if (success)
        changeState(EXEC_TRAJ, "STATE");
      else
        changeState(GEN_NEW_TRAJ, "STATE");
      break;
    }

    case EMER_STOP:
    {
      // 可以在这里发布一个“原地悬停”的 PositionCommand（略）
      break;
    }
  }
}

void rcvWaypointsCallBack(const nav_msgs::Path &wp)
{
  if (wp.poses[0].pose.position.z < 0.0)
    return;

  if (demox == 0)
  {
    // 按 reference 约定：demox==0 时目标点的 x/y 固定为 (12, -4)，z 由 RViz 的 goal.z 指定
    // （这样不会出现“终点随意定制”的情况）
    target_pt << 12.0, -4.0, wp.poses[0].pose.position.z;

    // 兼容：若 RViz 发送 z=0（常见 2D goal），则保持当前高度，避免无意义下钻
    if (target_pt(2) <= 1e-3)
      target_pt(2) = has_odom ? odom_pt(2) : 2.0;
  }
  else
  {
    target_pt << 0, 0, 5;
  }

  ROS_INFO("[node] receive the planning target");
  start_pt   = odom_pt;
  start_vel  = odom_vel;
  has_target = true;

  if (exec_state == WAIT_TARGET)
    changeState(GEN_NEW_TRAJ, "STATE");
  else if (exec_state == EXEC_TRAJ)
    changeState(REPLAN_TRAJ, "STATE");
}

void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 &pointcloud_map)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;

  _astar_path_finder->resetOccupy();  // 新的一帧点云到来，需将占据容器清零；

  pcl::fromROSMsg(pointcloud_map, cloud);

  if ((int)cloud.points.size() == 0)
    return;

  pcl::PointXYZ pt;
  for (int idx = 0; idx < (int)cloud.points.size(); idx++)
  {
    pt = cloud.points[idx];
    // set obstacles into grid map for path planning
    _astar_path_finder->set_barrier(pt.x, pt.y, pt.z);
  }
}

bool trajGeneration()
{
  bool astar_success = _astar_path_finder->AstarSearch(start_pt, target_pt);

  if (!astar_success)
  {
    _astar_path_finder->resetUsedGrids();
    has_traj = false;
    return false;
  }

  auto grid_path = _astar_path_finder->getPath();
  // Reset map for next call
  _astar_path_finder->resetUsedGrids();

  MatrixXd path_raw(int(grid_path.size()), 3);
  for (int k = 0; k < int(grid_path.size()); k++)
  {
    path_raw.row(k) = grid_path[k].transpose();
  }
  visPathA(path_raw);

  grid_path = _astar_path_finder->pathSimplify(grid_path, _path_resolution);
  MatrixXd path = MatrixXd::Zero(int(grid_path.size()), 3);
  for (int k = 0; k < int(grid_path.size()); k++)
  {
    path.row(k) = grid_path[k].transpose();
  }

  ROS_INFO("[trajGeneration] path rows = %d", (int)path.rows());

  // 若路径点太少，直接认为生成失败，避免后续矩阵维度错误
  if (path.rows() < 2)
  {
    ROS_WARN("[trajGeneration] path has less than 2 points, treat as failure.");
    has_traj = false;
    return false;
  }

  if (!trajOptimization(path, path_raw))
  {
    ROS_WARN("[trajGeneration] trajOptimization failed (unsafe).");
    has_traj = false;
    return false;
  }
  time_duration   = _polyTime.sum();
  time_traj_start = ros::Time::now();
  has_traj        = (_polyCoeff.rows() > 0);

  // 保持向 traj_server 发布 PolynomialTrajectory（可视化 / 备用）
  if (has_traj)
    trajPublish(_polyCoeff, _polyTime);

  return has_traj;
}

void issafe(const ros::TimerEvent &e)
{
  if (!has_odom)
    return;

  Eigen::Vector3d now_pos    = odom_pt;
  Eigen::Vector3i odom_index = _astar_path_finder->c2i(now_pos);
  if (_astar_path_finder->is_occupy(odom_index))
  {
    ROS_WARN("now place is in obstacle, the drone has cracked!!!");
    cracked = true;

    // 关键：立刻写入并 flush/close，确保评测脚本能稳定读到碰撞标志（即便 roslaunch 强制杀进程）
    const char *kIssafePath =
        "/home/stuwork/MRPC-2025-homework/code/src/quadrotor_simulator/"
        "so3_control/src/issafe.txt";
    std::ofstream f(kIssafePath, std::ios::out | std::ios::trunc);
    if (f)
    {
      f << 1 << std::endl;
      f.flush();
    }
    else
    {
      ROS_WARN("[issafe] failed to open issafe.txt for writing");
    }
  }
}

bool trajOptimization(const Eigen::MatrixXd &path_in, const Eigen::MatrixXd &path_raw_in)
{
  auto try_generate = [&](const Eigen::MatrixXd &path, const char *tag) -> bool {
    if (path.rows() < 2)
      return false;

    MatrixXd vel = MatrixXd::Zero(2, 3);
    MatrixXd acc = MatrixXd::Zero(2, 3);
    vel.row(0) = start_vel.transpose();
    vel.row(1) << 0, 0, 0;
    acc.row(0) << 0, 0, 0;
    acc.row(1) << 0, 0, 0;

    _polyTime  = timeAllocation(path);
    _polyCoeff = _trajGene->PolyQPGeneration(_dev_order, path, vel, acc, _polyTime);

    const int unsafe_segment = _astar_path_finder->safeCheck(_polyCoeff, _polyTime);
    if (unsafe_segment != -1)
    {
      ROS_WARN("[trajOptimization] %s traj unsafe at seg=%d (rows=%d).", tag, unsafe_segment, (int)path.rows());
      return false;
    }

    visPath(path);
    visTrajectory(_polyCoeff, _polyTime);
    return true;
  };

  if (path_in.rows() < 2)
  {
    ROS_WARN("[trajOptimization] path has less than 2 points, fail!");
    _polyCoeff.resize(0, 0);
    _polyTime.resize(0);
    return false;
  }

  // 1) 先用简化后的路径生成（更快、更少段）
  if (try_generate(path_in, "simplified"))
    return true;

  // 2) 回退：用更密的原始 A* 路径（减少多项式“抄近路”穿障的概率）
  Eigen::MatrixXd raw = path_raw_in;
  if (raw.rows() >= 2)
  {
    // 确保端点与当前规划一致（目标点永不变）
    raw.row(0) = path_in.row(0);
    raw.row(raw.rows() - 1) = path_in.row(path_in.rows() - 1);
  }

  if (try_generate(raw, "raw"))
    return true;

  ROS_WARN("[trajOptimization] failed to generate a safe trajectory (simplified+raw).");
  _polyCoeff.resize(0, 0);
  _polyTime.resize(0);
  return false;
}

// 关键：traj_server 内部用“归一化时间” s=t/T 来评估多项式，
// 因此这里需要把 PolyQP 生成的系数（以真实时间 t 为自变量）转换为以 s 为自变量：
//   p(t)=Σ a_j t^j = Σ (a_j T^j) s^j  (t=T*s)
void trajPublish(MatrixXd polyCoeff, VectorXd time)
{
  if (polyCoeff.size() == 0 || time.size() == 0)
  {
    ROS_WARN("[trajectory_generator_waypoint] empty trajectory, nothing to "
             "publish.");
    return;
  }

  static int count = 1;  // The first trajectory_id must be greater than 0.

  quadrotor_msgs::PolynomialTrajectory traj_msg;

  traj_msg.header.seq      = count;
  traj_msg.header.stamp    = ros::Time::now();
  traj_msg.header.frame_id = std::string("world");
  traj_msg.trajectory_id   = count;
  traj_msg.action          = quadrotor_msgs::PolynomialTrajectory::ACTION_ADD;

  traj_msg.num_order   = 2 * _dev_order - 1;
  traj_msg.num_segment = time.size();

  unsigned int poly_number = traj_msg.num_order + 1;

  Vector3d initialVel =
      _trajGene->getVelPoly(_polyCoeff, 0, 0.0);
  Vector3d finalVel = _trajGene->getVelPoly(
      _polyCoeff, traj_msg.num_segment - 1,
      _polyTime(traj_msg.num_segment - 1));
  traj_msg.start_yaw = atan2(initialVel(1), initialVel(0));
  traj_msg.final_yaw = atan2(finalVel(1), finalVel(0));

  for (unsigned int i = 0; i < traj_msg.num_segment; ++i)
  {
    for (unsigned int j = 0; j < poly_number; ++j)
    {
      traj_msg.coef_x.push_back(polyCoeff(i, j) * std::pow(time(i), (double)j));
      traj_msg.coef_y.push_back(polyCoeff(i, poly_number + j) * std::pow(time(i), (double)j));
      traj_msg.coef_z.push_back(polyCoeff(i, 2 * poly_number + j) * std::pow(time(i), (double)j));
    }
    traj_msg.time.push_back(time(i));
    traj_msg.order.push_back(traj_msg.num_order);
  }

  traj_msg.mag_coeff = 1.0;

  ROS_WARN("[traj..gen...node] traj_msg publish (id=%d, seg=%d)",
           count, traj_msg.num_segment);
  count++;

  _traj_pub.publish(traj_msg);
}

VectorXd timeAllocation(MatrixXd Path)
{
  VectorXd time(Path.rows() - 1);
  MatrixXd piece;
  double   dist;
  const double t = _Vel / _Acc;
  const double d = 0.5 * _Acc * t * t;

  // 可调的时间缩放与最小段时长：提高可跟踪性，降低 RMSE/抖动
  static bool inited = false;
  static double time_scale = 1.5;
  static double min_seg_time = 0.2;
  if (!inited)
  {
    ros::param::param("~planning/time_scale", time_scale, 1.5);
    ros::param::param("~planning/min_seg_time", min_seg_time, 0.2);
    inited = true;
  }

  for (int i = 0; i < int(time.size()); i++)
  {
    piece = Path.row(i + 1) - Path.row(i);
    dist  = piece.norm();
    if (dist < d + d)
    {
      time(i) = 2.0 * sqrt(dist / _Acc);
    }
    else
    {
      time(i) = 2.0 * t + (dist - 2.0 * d) / _Vel;
    }

    if (time(i) < min_seg_time) time(i) = min_seg_time;
  }
  return time_scale * time;
}

void visTrajectory(MatrixXd polyCoeff, VectorXd time)
{
  visualization_msgs::Marker _traj_vis;

  _traj_vis.header.stamp    = ros::Time::now();
  _traj_vis.header.frame_id = "world";

  _traj_vis.ns   = "traj_node/trajectory";
  _traj_vis.id   = 0;
  _traj_vis.type = visualization_msgs::Marker::SPHERE_LIST;
  _traj_vis.action           = visualization_msgs::Marker::ADD;
  _traj_vis.scale.x          = _vis_traj_width;
  _traj_vis.scale.y          = _vis_traj_width;
  _traj_vis.scale.z          = _vis_traj_width;
  _traj_vis.pose.orientation.x = 0.0;
  _traj_vis.pose.orientation.y = 0.0;
  _traj_vis.pose.orientation.z = 0.0;
  _traj_vis.pose.orientation.w = 1.0;

  _traj_vis.color.a = 1.0;
  _traj_vis.color.r = 0.0;
  _traj_vis.color.g = 0.5;
  _traj_vis.color.b = 1.0;

  _traj_vis.points.clear();
  Vector3d             pos;
  geometry_msgs::Point pt;

  for (int i = 0; i < time.size(); i++)
  {
    for (double t = 0.0; t < time(i); t += 0.01)
    {
      pos = _trajGene->getPosPoly(polyCoeff, i, t);
      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = pos(2);
      _traj_vis.points.push_back(pt);
    }
  }
  _traj_vis_pub.publish(_traj_vis);
}

void visTrajectory_before(MatrixXd polyCoeff, VectorXd time)
{
  visualization_msgs::Marker _traj_vis;

  _traj_vis.header.stamp    = ros::Time::now();
  _traj_vis.header.frame_id = "world";

  _traj_vis.ns   = "traj_node/trajectory";
  _traj_vis.id   = 0;
  _traj_vis.type = visualization_msgs::Marker::SPHERE_LIST;
  _traj_vis.action           = visualization_msgs::Marker::ADD;
  _traj_vis.scale.x          = _vis_traj_width;
  _traj_vis.scale.y          = _vis_traj_width;
  _traj_vis.scale.z          = _vis_traj_width;
  _traj_vis.pose.orientation.x = 0.0;
  _traj_vis.pose.orientation.y = 0.0;
  _traj_vis.pose.orientation.z = 0.0;
  _traj_vis.pose.orientation.w = 1.0;

  _traj_vis.color.a = 1.0;
  _traj_vis.color.r = 1.0;
  _traj_vis.color.g = 0.5;
  _traj_vis.color.b = 0.0;

  _traj_vis.points.clear();
  Vector3d             pos;
  geometry_msgs::Point pt;

  for (int i = 0; i < time.size(); i++)
  {
    for (double t = 0.0; t < time(i); t += 0.01)
    {
      pos = _trajGene->getPosPoly(polyCoeff, i, t);
      pt.x = pos(0);
      pt.y = pos(1);
      pt.z = pos(2);
      _traj_vis.points.push_back(pt);
    }
  }
  _traj_before_vis_pub.publish(_traj_vis);
}

void visPath(MatrixXd nodes)
{
  visualization_msgs::Marker points;

  int id     = 0;
  points.id   = id;
  points.type = visualization_msgs::Marker::SPHERE_LIST;
  points.header.frame_id       = "world";
  points.header.stamp          = ros::Time::now();
  points.ns                    = "traj_node/path";
  points.action                = visualization_msgs::Marker::ADD;
  points.pose.orientation.w    = 1.0;
  points.pose.orientation.x    = 0.0;
  points.pose.orientation.y    = 0.0;
  points.pose.orientation.z    = 0.0;
  points.scale.x               = 0.2;
  points.scale.y               = 0.2;
  points.scale.z               = 0.2;
  points.color.a               = 1.0;
  points.color.r               = 0.0;
  points.color.g               = 0.0;
  points.color.b               = 1.0;

  geometry_msgs::Point p;
  for (int i = 0; i < int(nodes.rows()); i++)
  {
    p.x = nodes(i, 0);
    p.y = nodes(i, 1);
    p.z = nodes(i, 2);

    points.points.push_back(p);
  }
  _path_vis_pub.publish(points);
}

void visPathA(MatrixXd nodes)
{
  visualization_msgs::Marker points;

  int id     = 0;
  points.id   = id;
  points.type = visualization_msgs::Marker::SPHERE_LIST;
  points.header.frame_id       = "world";
  points.header.stamp          = ros::Time::now();
  points.ns                    = "traj_node/path";
  points.action                = visualization_msgs::Marker::ADD;
  points.pose.orientation.w    = 1.0;
  points.pose.orientation.x    = 0.0;
  points.pose.orientation.y    = 0.0;
  points.pose.orientation.z    = 0.0;
  points.scale.x               = 0.2;
  points.scale.y               = 0.2;
  points.scale.z               = 0.2;
  points.color.a               = 1.0;
  points.color.r               = 1.0;
  points.color.g               = 0.0;
  points.color.b               = 0.0;

  geometry_msgs::Point p;
  for (int i = 0; i < int(nodes.rows()); i++)
  {
    p.x = nodes(i, 0);
    p.y = nodes(i, 1);
    p.z = nodes(i, 2);

    points.points.push_back(p);
  }
  _astar_path_vis_pub.publish(points);
}

Vector3d getPos(double t_cur)
{
  Vector3d p;
  if (evalTraj(t_cur, &p, nullptr, nullptr))
    return p;
  return Vector3d::Zero();
}

Vector3d getVel(double t_cur)
{
  Vector3d v;
  if (evalTraj(t_cur, nullptr, &v, nullptr))
    return v;
  return Vector3d::Zero();
}

Vector3d getAcc(double t_cur)
{
  Vector3d a;
  if (evalTraj(t_cur, nullptr, nullptr, &a))
    return a;
  return Vector3d::Zero();
}

// 精确地在分段多项式上求值（旧版用 0.01 离散积分，容易因为 > / 超界导致返回 0，引发倒飞/乱跑）
bool evalTraj(double t_cur, Vector3d *pos, Vector3d *vel, Vector3d *acc)
{
  if (!has_traj) return false;
  if (_polyCoeff.rows() <= 0 || _polyTime.size() <= 0) return false;

  // clamp time into [0, total]
  const double total = _polyTime.sum();
  double t = std::max(0.0, std::min(total, t_cur));

  for (int i = 0; i < _polyTime.size(); ++i)
  {
    const double Ti = _polyTime(i);
    const bool last = (i == _polyTime.size() - 1);
    if (t <= Ti + 1e-9 || last)
    {
      const double tl = std::max(0.0, std::min(Ti, t));
      if (pos) *pos = _trajGene->getPosPoly(_polyCoeff, i, tl);
      if (vel) *vel = _trajGene->getVelPoly(_polyCoeff, i, tl);
      if (acc) *acc = _trajGene->getAccPoly(_polyCoeff, i, tl);
      return true;
    }
    t -= Ti;
  }
  return false;
}

// 预测未来一段轨迹是否会碰撞（使用最新占据栅格；可选基于 clearance 的“近碰撞”触发）
bool trajUnsafeAhead(double t_now, double horizon_s)
{
  if (!has_traj) return false;
  if (!_astar_path_finder) return false;

  int hard_xy_cells = 1, hard_z_cells = 0;
  ros::param::param("~astar_hard_clearance_xy", hard_xy_cells, 1);
  ros::param::param("~astar_hard_clearance_z",  hard_z_cells,  0);

  // 为了避免“缝明明能过却反复重规划/乱飞”，默认只对“真实占据”触发重规划。
  // 如需更保守，可开启 use_clearance 并设置最小 clearance（米）。
  bool   use_clearance = false;
  double min_clearance_m = 0.15;
  ros::param::param("~collision_check_use_clearance", use_clearance, false);
  ros::param::param("~collision_check_min_clearance_m", min_clearance_m, 0.15);

  double dt = 0.05;  // 20Hz 预测检查
  ros::param::param("~collision_check_dt", dt, 0.05);
  const double t_end = std::max(t_now, std::min(time_duration, t_now + std::max(0.2, horizon_s)));

  for (double t = t_now; t <= t_end + 1e-9; t += dt)
  {
    Vector3d pos = getPos(t);

    // map bounds check（越界直接认为不安全，触发重规划）
    if (pos(0) < _map_lower(0) || pos(0) > _map_upper(0) ||
        pos(1) < _map_lower(1) || pos(1) > _map_upper(1) ||
        pos(2) < _map_lower(2) || pos(2) > _map_upper(2))
      return true;

    const Vector3i idx = _astar_path_finder->c2i(pos);
    if (_astar_path_finder->is_occupy(idx)) return true;

    if (use_clearance)
    {
      // 使用距离场近似（局部扫描）来做“近碰撞”判断，避免 hard-clearance 过于二值化导致频繁触发
      const double clr = _astar_path_finder->nearestObsDistM(idx, 8);
      if (clr < std::max(0.0, min_clearance_m)) return true;

      // 可选：如果你仍想保留 hard-clearance 的语义（但只在非常贴近时触发）
      if (_astar_path_finder->isTooCloseHard(idx, hard_xy_cells, hard_z_cells) &&
          clr < std::max(0.0, min_clearance_m + 1e-3))
        return true;
    }
  }

  return false;
}

// *** 新增：根据当前时间 t_cur 发布 /position_cmd
void publishPositionCommand(double t_cur)
{
  if (!has_traj)
    return;

  if (t_cur < 0.0)
    t_cur = 0.0;
  if (t_cur > time_duration)
    t_cur = time_duration;

  Vector3d pos = getPos(t_cur);
  Vector3d vel = getVel(t_cur);
  Vector3d acc = getAcc(t_cur);

  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp    = ros::Time::now();
  cmd.header.frame_id = "world";

  cmd.position.x = pos(0);
  cmd.position.y = pos(1);
  cmd.position.z = pos(2);

  cmd.velocity.x = vel(0);
  cmd.velocity.y = vel(1);
  cmd.velocity.z = vel(2);

  cmd.acceleration.x = acc(0);
  cmd.acceleration.y = acc(1);
  cmd.acceleration.z = acc(2);

  // yaw：低速/停住时保持上一次 yaw，避免 atan2(0,0) 抖动导致“目标附近乱跑”
  static double last_yaw = 0.0;
  const double vxy = std::sqrt(vel(0) * vel(0) + vel(1) * vel(1));
  if (vxy > 0.15)
    last_yaw = std::atan2(vel(1), vel(0));
  cmd.yaw     = last_yaw;
  cmd.yaw_dot = 0.0;

  cmd.trajectory_id   = 0;
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;

  _pos_cmd_pub.publish(cmd);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "traj_node");
  ros::NodeHandle nh("~");

  nh.param("planning/vel", _Vel, 1.0);
  nh.param("planning/acc", _Acc, 1.0);
  nh.param("planning/dev_order", _dev_order, 3);
  nh.param("planning/min_order", _min_order, 3);
  nh.param("vis/vis_traj_width", _vis_traj_width, 0.15);
  nh.param("map/resolution", _resolution, 0.2);
  nh.param("map/x_size", _x_size, 50.0);
  nh.param("map/y_size", _y_size, 50.0);
  nh.param("map/z_size", _z_size, 5.0);
  nh.param("path/resolution", _path_resolution, 0.05);
  nh.param("replanning/thresh_replan", replan_thresh, -1.0);
  nh.param("replanning/thresh_no_replan", no_replan_thresh, -1.0);
  nh.param("planning/demox", demox, 0);

  _poly_num1D = 2 * _dev_order;
  cracked     = false;
  exec_state  = INIT;
  time_Count  = 0;
  target_Count = 0;
  
  std::string issafe_path = "/home/stuwork/MRPC-2025-homework/code/src/quadrotor_simulator/so3_control/src/issafe.txt";
  std::ofstream clear_file(issafe_path, std::ios::out | std::ios::trunc);
  clear_file.close();

  _exec_timer    = nh.createTimer(ros::Duration(0.01), execCallback);
  _safety_timer_ = nh.createTimer(ros::Duration(0.05), issafe);

  _odom_sub = nh.subscribe("odom", 10, rcvOdomCallback);
  _map_sub  = nh.subscribe("local_pointcloud", 1, rcvPointCloudCallBack);
  _pts_sub  = nh.subscribe("waypoints", 1, rcvWaypointsCallBack);

  // *** PolynomialTrajectory 给 traj_server / 可视化
  _traj_pub =
      nh.advertise<quadrotor_msgs::PolynomialTrajectory>("trajectory", 50);

  _traj_vis_pub =
      nh.advertise<visualization_msgs::Marker>("vis_trajectory", 1);
  _traj_before_vis_pub =
      nh.advertise<visualization_msgs::Marker>("vis_trajectory_before", 1);
  _path_vis_pub =
      nh.advertise<visualization_msgs::Marker>("vis_path", 1);
  _astar_path_vis_pub =
      nh.advertise<visualization_msgs::Marker>("vis_path_astar", 1);

  // *** 关键：直接给 so3_control 的位置指令
  _pos_cmd_pub =
      nh.advertise<quadrotor_msgs::PositionCommand>("/position_cmd", 10);

  // set the obstacle map
  _map_lower << -_x_size / 2.0, -_y_size / 2.0, 0.0;
  _map_upper << +_x_size / 2.0, +_y_size / 2.0, _z_size;
  _inv_resolution = 1.0 / _resolution;
  _max_x_id       = (int)(_x_size * _inv_resolution);
  _max_y_id       = (int)(_y_size * _inv_resolution);
  _max_z_id       = (int)(_z_size * _inv_resolution);

  _astar_path_finder = new Astarpath();
  _astar_path_finder->begin_grid_map(_resolution, _map_lower, _map_upper,
                                     _max_x_id, _max_y_id, _max_z_id);

  ros::Rate rate(100);
  bool      status = ros::ok();
  while (status)
  {
    ros::spinOnce();
    status = ros::ok();
    rate.sleep();
  }

  return 0;
}
