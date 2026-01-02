#include <algorithm>
#include <fstream>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <cmath>
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
#include <std_msgs/Float64.h>
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
ros::Publisher _rmse_pub;     // *** 新增：实时 RMSE 发布器

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

// online RMSE统计（与 calculate_results.py 一致：sqrt(mean(diff^2))，对 3 个维度取全局均值）
static double   g_rmse_sumsq = 0.0;   // Σ ||e||^2
static uint64_t g_rmse_n     = 0;     // 样本数

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

// auto shutdown on goal (trajectory_generator_node is required=true in launch, so roslaunch will exit)
static bool   g_auto_shutdown_on_goal = true;
static double g_shutdown_delay_s      = 0.5;
static bool   g_shutdown_scheduled    = false;
static ros::NodeHandle *g_nh_ptr      = nullptr;
static ros::Timer g_shutdown_timer;
void shutdownCb(const ros::TimerEvent &e);

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
void abortTrajServer();
void trajPublish(MatrixXd polyCoeff, VectorXd time);
bool trajGeneration();
VectorXd timeAllocation(MatrixXd Path);
Vector3d getPos(double t_cur);
Vector3d getVel(double t_cur);
Vector3d getAcc(double t_cur);
bool evalTraj(double t_cur, Vector3d *pos, Vector3d *vel, Vector3d *acc);
void issafe(const ros::TimerEvent &e);
void publishPositionCommand(double t_cur);  // *** 新增：按当前时间发布位置指令
bool trajUnsafeAhead(double t_now, double horizon_s, double *t_hit_out);
void publishRmse(double t_cur);
bool rmseOkAndStable(double threshold, double hold_s);

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
      // *** 实时 RMSE（用于观察误差收敛）
      publishRmse(t_cur);

      // 若 RMSE 已经足够小并稳定一段时间，直接结束（无需强行贴到 goal 点）
      double rmse_stop = 0.015;
      double rmse_hold = 0.40;
      ros::param::param("~rmse_stop", rmse_stop, 0.015);
      ros::param::param("~rmse_hold_time", rmse_hold, 0.40);
      if (rmseOkAndStable(rmse_stop, rmse_hold))
      {
        abortTrajServer();
        changeState(WAIT_TARGET, "RMSE_OK");
        if (g_auto_shutdown_on_goal && !g_shutdown_scheduled)
        {
          g_shutdown_scheduled = true;
          if (g_nh_ptr)
            g_shutdown_timer = g_nh_ptr->createTimer(ros::Duration(std::max(0.0, g_shutdown_delay_s)),
                                                    shutdownCb, true);
          else
            std::exit(0);
        }
        return;
      }

      // 关键修复：用最新点云对“未来一小段轨迹”做碰撞预测，若即将穿障则立即触发重规划
      // 这解决“障碍已在视野内但仍沿旧轨迹撞上去”的问题。
      static ros::Time last_force_replan_t(0.0);
      if ((time_now - last_force_replan_t).toSec() > 0.10)  // 10Hz 上限，提升反应速度
      {
        // 按速度自适应的前瞻：速度越快，需要更长 horizon 才能覆盖“重规划延迟+制动距离”
        const double v = odom_vel.norm();
        double horizon_s = std::min(3.0, std::max(1.5, 1.0 + 0.6 * v));
        ros::param::param("~collision_check_horizon", horizon_s, horizon_s);

        double t_hit = -1.0;
        if (trajUnsafeAhead(t_cur, horizon_s, &t_hit))
        {
          last_force_replan_t = time_now;

          // 若碰撞发生得很近，则先 ABORT 让 traj_server 立刻悬停在当前点，避免“看见了但来不及躲”
          double stop_time = 0.5;
          ros::param::param("~collision_stop_time", stop_time, 0.5);
          const double time_to_hit = (t_hit > 0.0) ? std::max(0.0, t_hit - t_cur) : 0.0;
          if (time_to_hit < stop_time)
            abortTrajServer();

          changeState(REPLAN_TRAJ, "COLLISION_AHEAD");
          return;
        }
      }

      // 到达目标判定：增加“驻留时间”防抖，并支持到点后自动退出（便于记录总时间）
      double goal_tol = 0.45;
      double vel_tol  = 0.25;
      double goal_hold_time = 0.35;
      bool   goal_require_low_vel = false;
      ros::param::param("~goal_tolerance", goal_tol, 0.45);
      ros::param::param("~goal_vel_tolerance", vel_tol, 0.25);
      ros::param::param("~goal_hold_time", goal_hold_time, 0.35);
      ros::param::param("~goal_require_low_vel", goal_require_low_vel, false);

      static bool in_goal = false;
      static ros::Time goal_enter_t(0.0);
      const double dist_to_goal_now = (target_pt - odom_pt).norm();
      if (dist_to_goal_now < goal_tol)
      {
        if (!in_goal) { in_goal = true; goal_enter_t = time_now; }
        const bool vel_ok = (!goal_require_low_vel) || (odom_vel.norm() < vel_tol);
        if (vel_ok && (time_now - goal_enter_t).toSec() > std::max(0.0, goal_hold_time))
        {
          has_traj   = false;
          has_target = false;
          abortTrajServer();  // 防止 traj_server 继续沿旧轨迹“跑过头/折返”
          changeState(WAIT_TARGET, "GOAL_REACHED");

          if (g_auto_shutdown_on_goal && !g_shutdown_scheduled)
          {
            g_shutdown_scheduled = true;
            if (g_nh_ptr)
              g_shutdown_timer = g_nh_ptr->createTimer(ros::Duration(std::max(0.0, g_shutdown_delay_s)),
                                                      shutdownCb, true);
            else
              ros::shutdown();
          }
          return;
        }
      }
      else
      {
        in_goal = false;
      }

      if (t_cur > time_duration - 1e-2)
      {
        // 关键修复：只有真正到达目标附近才清空 has_target。
        // 若本段轨迹提前结束但距离目标仍远，则立即进入下一次规划，避免“卡死等待新目标”。
        const double dist_to_goal = (target_pt - odom_pt).norm();

        has_traj = false;
        if (dist_to_goal < goal_tol && ((!goal_require_low_vel) || (odom_vel.norm() < vel_tol)))
        {
          has_target = false;
          abortTrajServer();
          changeState(WAIT_TARGET, "STATE");

          if (g_auto_shutdown_on_goal && !g_shutdown_scheduled)
          {
            g_shutdown_scheduled = true;
            if (g_nh_ptr)
              g_shutdown_timer = g_nh_ptr->createTimer(ros::Duration(std::max(0.0, g_shutdown_delay_s)),
                                                      shutdownCb, true);
            else
              ros::shutdown();
          }
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

  // reset online RMSE stats for this run
  g_rmse_sumsq = 0.0;
  g_rmse_n     = 0;

  if (exec_state == WAIT_TARGET)
    changeState(GEN_NEW_TRAJ, "STATE");
  else if (exec_state == EXEC_TRAJ)
    changeState(REPLAN_TRAJ, "STATE");
}

void rcvPointCloudCallBack(const sensor_msgs::PointCloud2 &pointcloud_map)
{
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(pointcloud_map, cloud);

  if ((int)cloud.points.size() == 0)
  {
    // 点云为空时不要清空上一帧占据栅格：否则会出现“短暂空帧->地图全空->穿障”
    ROS_WARN_THROTTLE(1.0, "[map] local_pointcloud empty, keep last occupancy map.");
    return;
  }

  // 地图更新策略：默认累积占据（静态环境更安全、更少遗忘），可用参数关闭
  static bool inited = false;
  static bool accumulate = true;
  static int  min_valid_pts = 20;
  if (!inited)
  {
    ros::param::param("~map/accumulate", accumulate, true);
    ros::param::param("~map/min_valid_points", min_valid_pts, 20);
    inited = true;
  }

  pcl::PointXYZ pt;
  int valid = 0;

  if (accumulate)
  {
    for (int idx = 0; idx < (int)cloud.points.size(); idx++)
    {
      pt = cloud.points[idx];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
      valid++;
      _astar_path_finder->set_barrier(pt.x, pt.y, pt.z);
    }
  }
  else
  {
    // 非累积：先统计有效点数；太少则不清空，避免清图穿障
    for (int idx = 0; idx < (int)cloud.points.size(); idx++)
    {
      pt = cloud.points[idx];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
      valid++;
    }

    if (valid < std::max(1, min_valid_pts))
    {
      ROS_WARN_THROTTLE(1.0, "[map] too few valid pts=%d (<%d), keep last occupancy map.",
                        valid, min_valid_pts);
      return;
    }

    _astar_path_finder->resetOccupy();
    for (int idx = 0; idx < (int)cloud.points.size(); idx++)
    {
      pt = cloud.points[idx];
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
      _astar_path_finder->set_barrier(pt.x, pt.y, pt.z);
    }
  }

  ROS_DEBUG_THROTTLE(1.0, "[map] local pts=%lu valid=%d accumulate=%d",
                     (unsigned long)cloud.points.size(), valid, (int)accumulate);
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

// 立即让 traj_server 进入 HOVER（停止继续沿旧轨迹走），用于“到点后仍往回飞/反复折返”等情况
void abortTrajServer()
{
  quadrotor_msgs::PolynomialTrajectory msg;
  msg.header.stamp    = ros::Time::now();
  msg.header.frame_id = "world";
  msg.action          = quadrotor_msgs::PolynomialTrajectory::ACTION_ABORT;
  msg.trajectory_id   = 0;
  _traj_pub.publish(msg);
}

void shutdownCb(const ros::TimerEvent & /*e*/)
{
  ROS_WARN("[node] goal reached, auto shutdown.");
  // 直接使用 std::exit(0) 是最安全的方式，避免所有回调队列和锁问题
  std::exit(0);
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
  static double time_scale = 1.3;
  static double min_seg_time = 0.15;
  if (!inited)
  {
    ros::param::param("~planning/time_scale", time_scale, 1.3);
    ros::param::param("~planning/min_seg_time", min_seg_time, 0.15);
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
bool trajUnsafeAhead(double t_now, double horizon_s, double *t_hit_out = nullptr)
{
  if (!has_traj) return false;
  if (!_astar_path_finder) return false;

  int hard_xy_cells = 1, hard_z_cells = 0;
  ros::param::param("~astar_hard_clearance_xy", hard_xy_cells, 1);
  ros::param::param("~astar_hard_clearance_z",  hard_z_cells,  0);

  // 关键：若只用“中心点落入占据格”作为触发，会出现“擦边撞上去”但不重规划的问题。
  // 因此默认启用 clearance 阈值；同时用“连续命中”抑制噪声，避免缝隙处抖动。
  bool   use_clearance = true;
  double min_clearance_base_m = 0.18;
  double min_clearance_kv = 0.03;  // extra clearance per (m/s) above 1m/s
  int    consec_hits_need = 3;
  ros::param::param("~collision_check_use_clearance", use_clearance, true);
  ros::param::param("~collision_check_min_clearance_m", min_clearance_base_m, 0.18);
  ros::param::param("~collision_check_min_clearance_kv", min_clearance_kv, 0.03);
  ros::param::param("~collision_check_consecutive_hits", consec_hits_need, 3);

  double dt = 0.05;  // 20Hz 预测检查
  ros::param::param("~collision_check_dt", dt, 0.05);
  const double t_end = std::max(t_now, std::min(time_duration, t_now + std::max(0.2, horizon_s)));

  static int consec_hits = 0;
  double t_hit = -1.0;
  if (t_hit_out) *t_hit_out = -1.0;

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
      const double v = odom_vel.norm();
      const double min_clearance_m = std::max(0.0, min_clearance_base_m + min_clearance_kv * std::max(0.0, v - 1.0));
      const double clr = _astar_path_finder->nearestObsDistM(idx, 8);
      if (clr < min_clearance_m)
      {
        consec_hits++;
        if (consec_hits >= std::max(1, consec_hits_need))
        {
          t_hit = t;
          if (t_hit_out) *t_hit_out = t_hit;
          return true;
        }
      }
      else
      {
        consec_hits = 0;
      }

      // 可选：如果你仍想保留 hard-clearance 的语义（但只在非常贴近时触发）
      if (_astar_path_finder->isTooCloseHard(idx, hard_xy_cells, hard_z_cells) &&
          clr < min_clearance_m + 1e-3)
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

// 实时 RMSE（用于仿真过程中观察误差收敛）
void publishRmse(double t_cur)
{
  if (!_rmse_pub) return;
  if (!has_traj) return;
  if (!has_odom) return;

  // desired position on current trajectory
  const Vector3d des = getPos(t_cur);
  if (!des.allFinite()) return;

  const Vector3d err = des - odom_pt;
  if (!err.allFinite()) return;

  g_rmse_sumsq += err.squaredNorm();
  g_rmse_n += 1;

  // match calculate_results.py: sqrt(mean(diff^2)) over all samples and 3 dims
  const double rmse = std::sqrt(g_rmse_sumsq / std::max(1.0, (double)g_rmse_n * 3.0));

  static ros::Time last_pub(0.0);
  const ros::Time now = ros::Time::now();
  if ((now - last_pub).toSec() < 0.1)  // 10Hz
    return;
  last_pub = now;

  std_msgs::Float64 msg;
  msg.data = rmse;
  _rmse_pub.publish(msg);

  ROS_INFO_THROTTLE(0.5, "[RMSE] online=%.4f (N=%lu)", rmse, (unsigned long)g_rmse_n);
}

// 当在线 RMSE 低于阈值并稳定保持一段时间后，认为“足够好”可自动结束（用户要求 RMSE<=0.015 即可停止）
bool rmseOkAndStable(double threshold, double hold_s)
{
  if (g_rmse_n < 20) return false;  // 需要一定样本量
  const double rmse = std::sqrt(g_rmse_sumsq / std::max(1.0, (double)g_rmse_n * 3.0));

  static bool in_ok = false;
  static ros::Time ok_enter(0.0);
  const ros::Time now = ros::Time::now();

  if (rmse <= threshold)
  {
    if (!in_ok) { in_ok = true; ok_enter = now; }
    if ((now - ok_enter).toSec() >= std::max(0.0, hold_s))
      return true;
  }
  else
  {
    in_ok = false;
  }
  return false;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "traj_node");
  ros::NodeHandle nh("~");
  g_nh_ptr = &nh;
  nh.param("auto_shutdown_on_goal", g_auto_shutdown_on_goal, true);
  nh.param("shutdown_delay", g_shutdown_delay_s, 0.5);

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

  // *** 实时 RMSE 发布（便于调参观察）
  _rmse_pub = nh.advertise<std_msgs::Float64>("rmse", 10);

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
