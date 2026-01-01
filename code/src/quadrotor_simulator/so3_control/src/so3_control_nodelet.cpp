#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <nodelet/nodelet.h>
#include <quadrotor_msgs/Corrections.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <quadrotor_msgs/SO3Command.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <so3_control/SO3Control.h>
#include <std_msgs/Bool.h>
#include <tf/transform_datatypes.h>

// for logging + paths
#include <ros/package.h>   // ros::package::getPath (roslib)
#include <fstream>
#include <mutex>
#include <cstdlib>

class SO3ControlNodelet : public nodelet::Nodelet
{
public:
  SO3ControlNodelet()
    : position_cmd_updated_(false)
    , position_cmd_init_(false)
    , des_yaw_(0)
    , des_yaw_dot_(0)
    , current_yaw_(0)
    , enable_motors_(true)
    , use_external_yaw_(false)
    , mass_(0.98)
    , timedata_log_dt_(0.02)   // 50Hz log by default
    , last_timedata_log_t_(-1.0)
  {
  }

  void onInit(void);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  void publishSO3Command(void);
  void position_cmd_callback(const quadrotor_msgs::PositionCommand::ConstPtr& cmd);
  void odom_callback(const nav_msgs::Odometry::ConstPtr& odom);
  void enable_motors_callback(const std_msgs::Bool::ConstPtr& msg);
  void corrections_callback(const quadrotor_msgs::Corrections::ConstPtr& msg);
  void imu_callback(const sensor_msgs::Imu& imu);

  // logging helpers
  void openTimedataLogs_(const ros::NodeHandle& nh);
  void appendTimedata_(double t);

  SO3Control      controller_;
  ros::Publisher  so3_command_pub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber position_cmd_sub_;
  ros::Subscriber enable_motors_sub_;
  ros::Subscriber corrections_sub_;
  ros::Subscriber imu_sub_;

  bool        position_cmd_updated_, position_cmd_init_;
  std::string frame_id_;

  Eigen::Vector3d des_pos_, des_vel_, des_acc_, kx_, kv_;
  double          des_yaw_, des_yaw_dot_;
  double          current_yaw_;
  bool            enable_motors_;
  bool            use_external_yaw_;
  double          kR_[3], kOm_[3], corrections_[3];

  // tuned parameters / logging
  double mass_;

  std::mutex  log_mtx_;
  std::ofstream timedata_pkg_file_;
  std::ofstream timedata_roshome_file_;
  std::string timedata_pkg_path_;
  std::string timedata_roshome_path_;
  double timedata_log_dt_;
  double last_timedata_log_t_;
};

void SO3ControlNodelet::publishSO3Command(void)
{
  controller_.calculateControl(des_pos_, des_vel_, des_acc_, des_yaw_,
                               des_yaw_dot_, kx_, kv_);

  const Eigen::Vector3d&    force       = controller_.getComputedForce();
  const Eigen::Quaterniond& orientation = controller_.getComputedOrientation();

  quadrotor_msgs::SO3Command::Ptr so3_command(new quadrotor_msgs::SO3Command);
  so3_command->header.stamp    = ros::Time::now();
  so3_command->header.frame_id = frame_id_;
  so3_command->force.x         = force(0);
  so3_command->force.y         = force(1);
  so3_command->force.z         = force(2);
  so3_command->orientation.x   = orientation.x();
  so3_command->orientation.y   = orientation.y();
  so3_command->orientation.z   = orientation.z();
  so3_command->orientation.w   = orientation.w();

  for (int i = 0; i < 3; i++)
  {
    so3_command->kR[i]  = kR_[i];
    so3_command->kOm[i] = kOm_[i];
  }

  so3_command->aux.current_yaw          = current_yaw_;
  so3_command->aux.kf_correction        = corrections_[0];
  so3_command->aux.angle_corrections[0] = corrections_[1];
  so3_command->aux.angle_corrections[1] = corrections_[2];
  so3_command->aux.enable_motors        = enable_motors_;
  so3_command->aux.use_external_yaw     = use_external_yaw_;
  so3_command_pub_.publish(so3_command);
}

void SO3ControlNodelet::appendTimedata_(double t)
{
  std::lock_guard<std::mutex> lk(log_mtx_);

  // throttle
  if (last_timedata_log_t_ >= 0.0 && (t - last_timedata_log_t_) < timedata_log_dt_)
    return;

  if (timedata_pkg_file_.is_open())
  {
    timedata_pkg_file_ << std::fixed << std::setprecision(9) << t << "\n";
    timedata_pkg_file_.flush();
  }
  if (timedata_roshome_file_.is_open())
  {
    timedata_roshome_file_ << std::fixed << std::setprecision(9) << t << "\n";
    timedata_roshome_file_.flush();
  }
  last_timedata_log_t_ = t;
}

void SO3ControlNodelet::openTimedataLogs_(const ros::NodeHandle& nh)
{
  // 1) package src path (preferred for your calculate_results.py hardcoded path)
  const std::string pkg_path = ros::package::getPath("so3_control");
  if (!pkg_path.empty())
  {
    timedata_pkg_path_ = pkg_path + "/src/control_timedata.txt";
    timedata_pkg_file_.open(timedata_pkg_path_.c_str(), std::ios::out | std::ios::trunc);
    if (!timedata_pkg_file_.is_open())
    {
      ROS_WARN_STREAM("[so3_control] Failed to open timedata log at: " << timedata_pkg_path_);
    }
    else
    {
      ROS_INFO_STREAM("[so3_control] Writing timedata log to: " << timedata_pkg_path_);
    }
  }
  else
  {
    ROS_WARN("[so3_control] ros::package::getPath(\"so3_control\") returned empty. "
             "Timedata will only be written to ROS_HOME fallback.");
  }

  // 2) ROS_HOME fallback
  std::string ros_home;
  const char* ros_home_env = std::getenv("ROS_HOME");
  if (ros_home_env && std::string(ros_home_env).size() > 0)
  {
    ros_home = std::string(ros_home_env);
  }
  else
  {
    const char* home_env = std::getenv("HOME");
    if (home_env && std::string(home_env).size() > 0)
      ros_home = std::string(home_env) + "/.ros";
    else
      ros_home = ".";
  }

  timedata_roshome_path_ = ros_home + "/control_timedata.txt";
  timedata_roshome_file_.open(timedata_roshome_path_.c_str(), std::ios::out | std::ios::trunc);
  if (!timedata_roshome_file_.is_open())
  {
    ROS_WARN_STREAM("[so3_control] Failed to open timedata log at: " << timedata_roshome_path_);
  }
  else
  {
    ROS_INFO_STREAM("[so3_control] Writing timedata log to: " << timedata_roshome_path_);
  }

  // allow user override of log rate
  nh.param("log/timedata_dt", timedata_log_dt_, 0.02);
  if (timedata_log_dt_ < 0.0) timedata_log_dt_ = 0.0;
}

void SO3ControlNodelet::position_cmd_callback(
  const quadrotor_msgs::PositionCommand::ConstPtr& cmd)
{
  des_pos_ = Eigen::Vector3d(cmd->position.x, cmd->position.y, cmd->position.z);
  des_vel_ = Eigen::Vector3d(cmd->velocity.x, cmd->velocity.y, cmd->velocity.z);
  des_acc_ = Eigen::Vector3d(cmd->acceleration.x, cmd->acceleration.y,
                             cmd->acceleration.z);

  /**
   * 任务三：调参建议（更稳、少急冲急停）
   *
   * 线性化平动误差模型常写作：
   *   m e¨ = -k_x e - k_v e˙
   * 选定自然频率 w_n 和阻尼比 ζ，可取：
   *   k_x = m w_n^2,   k_v = 2 m ζ w_n
   *
   * 这里给一组偏“稳、阻尼足”的默认值（你可继续微调）：
   *  - XY：w_n=3.0, ζ=0.95
   *  - Z ：w_n=3.6, ζ=1.00
   */
  const double wn_xy = 3.0;
  const double zeta_xy = 0.95;
  const double wn_z  = 3.6;
  const double zeta_z = 1.00;

  const double kx_xy = mass_ * wn_xy * wn_xy;
  const double kv_xy = 2.0 * mass_ * zeta_xy * wn_xy;

  const double kx_z  = mass_ * wn_z * wn_z;
  const double kv_z  = 2.0 * mass_ * zeta_z * wn_z;

  kx_ = Eigen::Vector3d(kx_xy, kx_xy, kx_z);
  kv_ = Eigen::Vector3d(kv_xy, kv_xy, kv_z);

  // 记录 timedata（用于 calculate_results.py 的 total_time 计算）
  // 用仿真时间/ROS time（若 use_sim_time=true，则与 /clock 同步）
  appendTimedata_(ros::Time::now().toSec());

  des_yaw_              = cmd->yaw;
  des_yaw_dot_          = cmd->yaw_dot;
  position_cmd_updated_ = true;
  position_cmd_init_    = true;

  publishSO3Command();
}

void SO3ControlNodelet::odom_callback(const nav_msgs::Odometry::ConstPtr& odom)
{
  const Eigen::Vector3d position(odom->pose.pose.position.x,
                                 odom->pose.pose.position.y,
                                 odom->pose.pose.position.z);
  const Eigen::Vector3d velocity(odom->twist.twist.linear.x,
                                 odom->twist.twist.linear.y,
                                 odom->twist.twist.linear.z);

  current_yaw_ = tf::getYaw(odom->pose.pose.orientation);

  controller_.setPosition(position);
  controller_.setVelocity(velocity);

  if (position_cmd_init_)
  {
    // 如果本次 odom 后没有新的 position_cmd，则沿用上一次的 des_* 继续发布（避免控制空窗）
    if (!position_cmd_updated_)
      publishSO3Command();
    position_cmd_updated_ = false;
  }
}

void SO3ControlNodelet::enable_motors_callback(const std_msgs::Bool::ConstPtr& msg)
{
  if (msg->data)
    ROS_INFO("Enabling motors");
  else
    ROS_INFO("Disabling motors");

  enable_motors_ = msg->data;
}

void SO3ControlNodelet::corrections_callback(
  const quadrotor_msgs::Corrections::ConstPtr& msg)
{
  corrections_[0] = msg->kf_correction;
  corrections_[1] = msg->angle_corrections[0];
  corrections_[2] = msg->angle_corrections[1];
}

void SO3ControlNodelet::imu_callback(const sensor_msgs::Imu& imu)
{
  const Eigen::Vector3d acc(imu.linear_acceleration.x,
                            imu.linear_acceleration.y,
                            imu.linear_acceleration.z);
  controller_.setAcc(acc);
}

void SO3ControlNodelet::onInit(void)
{
  ros::NodeHandle n(getPrivateNodeHandle());

  std::string quadrotor_name;
  n.param("quadrotor_name", quadrotor_name, std::string("quadrotor"));
  frame_id_ = "/" + quadrotor_name;

  double mass;
  n.param("mass", mass, 0.98);
  mass_ = mass;
  controller_.setMass(mass_);

  n.param("use_external_yaw", use_external_yaw_, true);

  n.param("gains/rot/x", kR_[0], 1.5);
  n.param("gains/rot/y", kR_[1], 1.5);
  n.param("gains/rot/z", kR_[2], 1.0);
  n.param("gains/ang/x", kOm_[0], 0.13);
  n.param("gains/ang/y", kOm_[1], 0.13);
  n.param("gains/ang/z", kOm_[2], 0.1);

  n.param("corrections/z", corrections_[0], 0.0);
  n.param("corrections/r", corrections_[1], 0.0);
  n.param("corrections/p", corrections_[2], 0.0);

  // open timedata logs
  openTimedataLogs_(n);

  so3_command_pub_ = n.advertise<quadrotor_msgs::SO3Command>("so3_cmd", 10);

  odom_sub_ = n.subscribe("odom", 10, &SO3ControlNodelet::odom_callback, this,
                          ros::TransportHints().tcpNoDelay());
  position_cmd_sub_ =
    n.subscribe("position_cmd", 10, &SO3ControlNodelet::position_cmd_callback,
                this, ros::TransportHints().tcpNoDelay());

  enable_motors_sub_ =
    n.subscribe("motors", 2, &SO3ControlNodelet::enable_motors_callback, this,
                ros::TransportHints().tcpNoDelay());
  corrections_sub_ =
    n.subscribe("corrections", 10, &SO3ControlNodelet::corrections_callback,
                this, ros::TransportHints().tcpNoDelay());

  imu_sub_ = n.subscribe("imu", 10, &SO3ControlNodelet::imu_callback, this,
                         ros::TransportHints().tcpNoDelay());
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(SO3ControlNodelet, nodelet::Nodelet);
