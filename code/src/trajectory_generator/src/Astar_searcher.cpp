#include "Astar_searcher.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <vector>

#include <ros/ros.h>

using namespace std;
using namespace Eigen;

// ============================================================
// Grid/map init & bookkeeping
// ============================================================

void Astarpath::begin_grid_map(double _resolution, Vector3d global_xyz_l,
                               Vector3d global_xyz_u, int max_x_id,
                               int max_y_id, int max_z_id)
{
  gl_xl = global_xyz_l(0);
  gl_yl = global_xyz_l(1);
  gl_zl = global_xyz_l(2);

  gl_xu = global_xyz_u(0);
  gl_yu = global_xyz_u(1);
  gl_zu = global_xyz_u(2);

  GRID_X_SIZE = max_x_id;
  GRID_Y_SIZE = max_y_id;
  GRID_Z_SIZE = max_z_id;
  GLYZ_SIZE   = GRID_Y_SIZE * GRID_Z_SIZE;
  GLXYZ_SIZE  = GRID_X_SIZE * GLYZ_SIZE;

  resolution = _resolution;
  inv_resolution = 1.0 / _resolution;

  data = new uint8_t[GLXYZ_SIZE];
  memset(data, 0, GLXYZ_SIZE * sizeof(uint8_t));

  data_raw = new uint8_t[GLXYZ_SIZE];
  memset(data_raw, 0, GLXYZ_SIZE * sizeof(uint8_t));

  Map_Node = new MappingNodePtr **[GRID_X_SIZE];
  for (int i = 0; i < GRID_X_SIZE; i++) {
    Map_Node[i] = new MappingNodePtr *[GRID_Y_SIZE];
    for (int j = 0; j < GRID_Y_SIZE; j++) {
      Map_Node[i][j] = new MappingNodePtr[GRID_Z_SIZE];
      for (int k = 0; k < GRID_Z_SIZE; k++) {
        Vector3i tmpIdx(i, j, k);
        Vector3d pos = gridIndex2coord(tmpIdx);
        Map_Node[i][j][k] = new MappingNode(tmpIdx, pos);
      }
    }
  }
}

void Astarpath::resetGrid(MappingNodePtr ptr) {
  ptr->id = 0;
  ptr->Father = NULL;
  ptr->g_score = inf;
  ptr->f_score = inf;
}

void Astarpath::resetUsedGrids() {
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++)
        resetGrid(Map_Node[i][j][k]);
}

void Astarpath::set_barrier(const double coord_x, const double coord_y,
                            const double coord_z)
{
  if (coord_x < gl_xl || coord_y < gl_yl || coord_z < gl_zl ||
      coord_x >= gl_xu || coord_y >= gl_yu || coord_z >= gl_zu)
    return;

  int idx_x = static_cast<int>((coord_x - gl_xl) * inv_resolution);
  int idx_y = static_cast<int>((coord_y - gl_yl) * inv_resolution);
  int idx_z = static_cast<int>((coord_z - gl_zl) * inv_resolution);

  if (idx_x < 0 || idx_x >= GRID_X_SIZE ||
      idx_y < 0 || idx_y >= GRID_Y_SIZE ||
      idx_z < 0 || idx_z >= GRID_Z_SIZE)
    return;

  // raw mark
  data_raw[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] = 1;

  // margin from existing param
  static bool   inited = false;
  static double margin_m = 0.0;
  if (!inited) {
    ros::param::param<double>("/trajectory_generator_node/map/margin", margin_m, 0.0);
    inited = true;
  }

  // IMPORTANT: keep inflation modest, but do NOT rely on this alone.
  // We will add hard-clearance checks at query time.
  double infl_xy_base = 0.35;
  double infl_z_base  = 0.25;
  ros::param::param("~astar_inflation_xy_base", infl_xy_base, 0.35);
  ros::param::param("~astar_inflation_z_base",  infl_z_base,  0.25);

  const double infl_xy_m = std::max(0.0, infl_xy_base + margin_m);
  const double infl_z_m  = std::max(0.0, infl_z_base  + 0.5 * margin_m);

  const int infl_xy = std::max(1, (int)std::ceil(infl_xy_m / resolution));
  const int infl_z  = std::max(0, (int)std::ceil(infl_z_m  / resolution));

  for (int dx = -infl_xy; dx <= infl_xy; ++dx) {
    for (int dy = -infl_xy; dy <= infl_xy; ++dy) {
      const int nx = idx_x + dx;
      const int ny = idx_y + dy;
      if (nx < 0 || nx >= GRID_X_SIZE || ny < 0 || ny >= GRID_Y_SIZE) continue;

      if (dx*dx + dy*dy > infl_xy*infl_xy) continue;

      for (int dz = -infl_z; dz <= infl_z; ++dz) {
        const int nz = idx_z + dz;
        if (nz < 0 || nz >= GRID_Z_SIZE) continue;
        data[nx * GLYZ_SIZE + ny * GRID_Z_SIZE + nz] = 1;
      }
    }
  }
}

vector<Vector3d> Astarpath::getVisitedNodes() {
  vector<Vector3d> visited_nodes;
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++) {
        if (Map_Node[i][j][k]->id == -1)
          visited_nodes.push_back(Map_Node[i][j][k]->coord);
      }
  ROS_WARN("visited_nodes size : %d", (int)visited_nodes.size());
  return visited_nodes;
}

Vector3d Astarpath::gridIndex2coord(const Vector3i &index) {
  Vector3d pt;
  pt(0) = ((double)index(0) + 0.5) * resolution + gl_xl;
  pt(1) = ((double)index(1) + 0.5) * resolution + gl_yl;
  pt(2) = ((double)index(2) + 0.5) * resolution + gl_zl;
  return pt;
}

Vector3i Astarpath::coord2gridIndex(const Vector3d &pt) {
  Vector3i idx;
  idx << min(max(int((pt(0) - gl_xl) * inv_resolution), 0), GRID_X_SIZE - 1),
      min(max(int((pt(1) - gl_yl) * inv_resolution), 0), GRID_Y_SIZE - 1),
      min(max(int((pt(2) - gl_zl) * inv_resolution), 0), GRID_Z_SIZE - 1);
  return idx;
}

Vector3i Astarpath::c2i(const Vector3d &pt) {
  Vector3i idx;
  idx << min(max(int((pt(0) - gl_xl) * inv_resolution), 0), GRID_X_SIZE - 1),
      min(max(int((pt(1) - gl_yl) * inv_resolution), 0), GRID_Y_SIZE - 1),
      min(max(int((pt(2) - gl_zl) * inv_resolution), 0), GRID_Z_SIZE - 1);
  return idx;
}

Eigen::Vector3d Astarpath::coordRounding(const Eigen::Vector3d &coord) {
  return gridIndex2coord(coord2gridIndex(coord));
}

inline bool Astarpath::isOccupied(const Eigen::Vector3i &index) const {
  return isOccupied(index(0), index(1), index(2));
}

bool Astarpath::is_occupy(const Eigen::Vector3i &index) {
  return isOccupied(index(0), index(1), index(2));
}

bool Astarpath::is_occupy_raw(const Eigen::Vector3i &index) {
  int idx_x = index(0);
  int idx_y = index(1);
  int idx_z = index(2);
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data_raw[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] == 1));
}

inline bool Astarpath::isFree(const Eigen::Vector3i &index) const {
  return isFree(index(0), index(1), index(2));
}

inline bool Astarpath::isOccupied(const int &idx_x, const int &idx_y,
                                  const int &idx_z) const {
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] == 1));
}

inline bool Astarpath::isFree(const int &idx_x, const int &idx_y,
                              const int &idx_z) const {
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] < 1));
}

// ============================================================
// Clearance helpers - 保留但简化
// ============================================================

double Astarpath::nearestObsDistM(const Vector3i &idx, int max_r_cells) const
{
  if (isOccupied(idx)) return 0.0;

  const int cx = idx(0), cy = idx(1), cz = idx(2);
  int best_r2 = (max_r_cells + 1) * (max_r_cells + 1) + 1;

  for (int dx = -max_r_cells; dx <= max_r_cells; ++dx) {
    int nx = cx + dx;
    if (nx < 0 || nx >= GRID_X_SIZE) continue;
    for (int dy = -max_r_cells; dy <= max_r_cells; ++dy) {
      int ny = cy + dy;
      if (ny < 0 || ny >= GRID_Y_SIZE) continue;
      for (int dz = -max_r_cells; dz <= max_r_cells; ++dz) {
        int nz = cz + dz;
        if (nz < 0 || nz >= GRID_Z_SIZE) continue;

        const int r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= best_r2) continue;
        if (isOccupied(nx, ny, nz)) {
          best_r2 = r2;
        }
      }
    }
  }

  if (best_r2 > max_r_cells * max_r_cells) {
    return (max_r_cells + 1) * resolution;
  }
  return std::sqrt((double)best_r2) * resolution;
}

bool Astarpath::isTooCloseHard(const Vector3i &idx,
                               int hard_xy_cells,
                               int hard_z_cells) const
{
  const int cx = idx(0), cy = idx(1), cz = idx(2);
  for (int dx = -hard_xy_cells; dx <= hard_xy_cells; ++dx) {
    int nx = cx + dx;
    if (nx < 0 || nx >= GRID_X_SIZE) continue;
    for (int dy = -hard_xy_cells; dy <= hard_xy_cells; ++dy) {
      int ny = cy + dy;
      if (ny < 0 || ny >= GRID_Y_SIZE) continue;
      if (dx*dx + dy*dy > hard_xy_cells*hard_xy_cells) continue;

      for (int dz = -hard_z_cells; dz <= hard_z_cells; ++dz) {
        int nz = cz + dz;
        if (nz < 0 || nz >= GRID_Z_SIZE) continue;
        if (isOccupied(nx, ny, nz)) return true;
      }
    }
  }
  return false;
}

double Astarpath::softClearancePenalty(const Vector3i &idx,
                                       double soft_range_m,
                                       int soft_scan_cells) const
{
  if (soft_range_m <= 1e-6) return 0.0;
  const double d = nearestObsDistM(idx, soft_scan_cells);
  if (d >= soft_range_m) return 0.0;
  const double x = (soft_range_m - d) / soft_range_m;
  return x * x; // quadratic
}

// ============================================================
// Successors (26-neighborhood) - 简化版本，参考 refer-1
// ============================================================

inline void Astarpath::AstarGetSucc(MappingNodePtr currentPtr,
                                    vector<MappingNodePtr> &neighborPtrSets,
                                    vector<double> &edgeCostSets)
{
  neighborPtrSets.clear();
  edgeCostSets.clear();

  Vector3i Idx_neighbor;
  for (int dx = -1; dx < 2; dx++) {
    for (int dy = -1; dy < 2; dy++) {
      for (int dz = -1; dz < 2; dz++) {
        if (dx == 0 && dy == 0 && dz == 0)
          continue;

        Idx_neighbor(0) = (currentPtr->index)(0) + dx;
        Idx_neighbor(1) = (currentPtr->index)(1) + dy;
        Idx_neighbor(2) = (currentPtr->index)(2) + dz;

        if (Idx_neighbor(0) < 0 || Idx_neighbor(0) >= GRID_X_SIZE ||
            Idx_neighbor(1) < 0 || Idx_neighbor(1) >= GRID_Y_SIZE ||
            Idx_neighbor(2) < 0 || Idx_neighbor(2) >= GRID_Z_SIZE) {
          continue;
        }

        neighborPtrSets.push_back(
            Map_Node[Idx_neighbor(0)][Idx_neighbor(1)][Idx_neighbor(2)]);
        edgeCostSets.push_back(sqrt(dx * dx + dy * dy + dz * dz));
      }
    }
  }
}

// 3D diagonal-distance heuristic + tie-breaker - 参考 refer-1 实现
double Astarpath::getHeu(MappingNodePtr node1, MappingNodePtr node2) {
  // 计算各轴索引差的绝对值
  double dx = std::abs(node1->index(0) - node2->index(0));
  double dy = std::abs(node1->index(1) - node2->index(1));
  double dz = std::abs(node1->index(2) - node2->index(2));

  double heu = 0.0;
  
  // 使用 3D 对角线距离 (Diagonal Distance)
  double min_delta = std::min({dx, dy, dz});
  double max_delta = std::max({dx, dy, dz});
  double mid_delta = dx + dy + dz - min_delta - max_delta;

  // 预计算的权重：
  // (sqrt(3) - sqrt(2)) ≈ 0.317837
  // (sqrt(2) - 1)       ≈ 0.414213
  heu = 0.317837 * min_delta + 0.414213 * mid_delta + max_delta;

  // Tie Breaker: 微小地打破对称性，倾向于深度优先，加快收敛
  double tie_breaker = 1.0 + 1.0 / 2500.0; 
  
  return heu * tie_breaker;
}

// ============================================================
// A* Search - 简化版本，参考 refer-1 实现
// ============================================================

bool Astarpath::AstarSearch(Vector3d start_pt, Vector3d end_pt)
{
  ros::Time time_1 = ros::Time::now();

  // 重置所有网格节点
  resetUsedGrids();
  Openset.clear();
  terminatePtr = NULL;

  // start_point 和 end_point 索引
  Vector3i start_idx = coord2gridIndex(start_pt);
  Vector3i end_idx = coord2gridIndex(end_pt);
  goalIdx = end_idx;

  // start_point 和 end_point 的位置（对齐到网格中心）
  start_pt = gridIndex2coord(start_idx);
  end_pt = gridIndex2coord(end_idx);

  // 初始化起点和终点节点指针
  MappingNodePtr startPtr = Map_Node[start_idx(0)][start_idx(1)][start_idx(2)];
  MappingNodePtr endPtr = Map_Node[end_idx(0)][end_idx(1)][end_idx(2)];

  MappingNodePtr currentPtr = NULL;
  MappingNodePtr neighborPtr = NULL;

  // 将起点加入 Open Set
  startPtr->g_score = 0;
  startPtr->f_score = getHeu(startPtr, endPtr);
  startPtr->id = 1;
  startPtr->coord = start_pt;
  startPtr->Father = NULL;
  Openset.insert(make_pair(startPtr->f_score, startPtr));

  double tentative_g_score;
  vector<MappingNodePtr> neighborPtrSets;
  vector<double> edgeCostSets;

  while (!Openset.empty()) {
    // 1. 弹出 f 值最小的节点
    currentPtr = Openset.begin()->second;
    Openset.erase(Openset.begin());
    currentPtr->id = -1;  // 标记为已访问（加入 closed set）

    // 2. 判断是否到达终点
    if (currentPtr->index == goalIdx) {
      terminatePtr = currentPtr;
      ROS_INFO("[A*] Goal found!");
      return true;
    }

    // 3. 扩展当前节点
    AstarGetSucc(currentPtr, neighborPtrSets, edgeCostSets);

    for (unsigned int i = 0; i < neighborPtrSets.size(); i++) {
      neighborPtr = neighborPtrSets[i];

      // 跳过已在 closed set 中的节点
      if (neighborPtr->id == -1) {
        continue;
      }

      tentative_g_score = currentPtr->g_score + edgeCostSets[i];

      // 跳过被占据的节点
      if (isOccupied(neighborPtr->index)) {
        continue;
      }

      if (neighborPtr->id == 0) {
        // 节点未访问过，添加到 open set
        neighborPtr->g_score = tentative_g_score;
        neighborPtr->f_score = tentative_g_score + getHeu(neighborPtr, endPtr);
        neighborPtr->Father = currentPtr;
        neighborPtr->id = 1;
        neighborPtr->coord = gridIndex2coord(neighborPtr->index);
        Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
      } else if (neighborPtr->id == 1) {
        // 节点已在 open set 中，检查是否需要更新
        if (neighborPtr->g_score > tentative_g_score) {
          neighborPtr->g_score = tentative_g_score;
          neighborPtr->Father = currentPtr;
          neighborPtr->f_score = tentative_g_score + getHeu(neighborPtr, endPtr);
          Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
        }
      }
    }
  }

  ros::Time time_2 = ros::Time::now();
  if ((time_2 - time_1).toSec() > 0.1)
    ROS_WARN("Time consume in Astar path finding is %f", (time_2 - time_1).toSec());

  ROS_WARN("[A*] No path found!");
  return false;
}

vector<Vector3d> Astarpath::getPath() {
  vector<Vector3d> path;
  vector<MappingNodePtr> front_path;

  // 从终点回溯到起点
  MappingNodePtr cur = terminatePtr;
  while (cur != NULL) {
    cur->coord = gridIndex2coord(cur->index);
    front_path.push_back(cur);
    cur = cur->Father;
  }

  // 将 front_path 中的节点坐标反转后添加到 path 中
  // front_path 是从终点到起点的顺序，需要反转为从起点到终点
  for (int i = (int)front_path.size() - 1; i >= 0; i--) {
    path.push_back(front_path[i]->coord);
  }

  return path;
}

// ============================================================
// Path post-processing - 简化版本，参考 refer-1 实现
// ============================================================

// 找到最近的自由格子
bool Astarpath::findNearestFree(const Vector3i &seed, Vector3i &out_free) {
  if (!isOccupied(seed)) {
    out_free = seed;
    return true;
  }

  const int maxR = 10;
  for (int r = 1; r <= maxR; ++r) {
    for (int dx = -r; dx <= r; ++dx) {
      for (int dy = -r; dy <= r; ++dy) {
        for (int dz = -r; dz <= r; ++dz) {
          if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r) continue;

          Vector3i q = seed + Vector3i(dx, dy, dz);
          if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
              q(1) < 0 || q(1) >= GRID_Y_SIZE ||
              q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;

          if (!isOccupied(q)) {
            out_free = q;
            return true;
          }
        }
      }
    }
  }
  return false;
}

// Line-of-Sight 检测：检查两点之间是否有障碍物
bool Astarpath::lineOfSight(const Vector3d& start, const Vector3d& end) {
  Vector3d direction = end - start;
  double distance = direction.norm();
  
  if (distance < 1e-6) return true;
  
  // 使用较小的步长进行检测，确保不会穿过障碍物
  double step_size = resolution * 0.5;
  int num_steps = static_cast<int>(distance / step_size) + 1;
  
  for (int i = 0; i <= num_steps; i++) {
    double t = static_cast<double>(i) / num_steps;
    Vector3d point = start + t * (end - start);
    Vector3i idx = coord2gridIndex(point);
    
    if (isOccupied(idx)) {
      return false;  // 有障碍物，不可通行
    }
  }
  return true;  // 无障碍物，可直接通行
}

std::vector<Vector3d> Astarpath::pathSimplify(const vector<Vector3d> &path,
                                              double path_resolution)
{
  // 如果路径点太少，直接返回
  if (path.size() <= 2) {
    return path;
  }
  
  // 使用 Line-of-Sight 进行路径简化
  vector<Vector3d> simplified_path;
  simplified_path.push_back(path[0]);  // 保留起点
  
  size_t current = 0;
  
  while (current < path.size() - 1) {
    // 尝试找到能直接到达的最远点
    size_t farthest = current + 1;
    
    for (size_t i = path.size() - 1; i > current + 1; i--) {
      if (lineOfSight(path[current], path[i])) {
        farthest = i;
        break;  // 找到最远的可直达点
      }
    }
    
    simplified_path.push_back(path[farthest]);
    current = farthest;
  }
  
  ROS_INFO("[pathSimplify] in=%d out=%d", (int)path.size(), (int)simplified_path.size());
  
  return simplified_path;
}

// ============================================================
// Keep your downstream helpers unchanged (but safeCheck upgraded)
// ============================================================

double Astarpath::perpendicularDistance(const Eigen::Vector3d point_insert,
                                        const Eigen::Vector3d point_st,
                                        const Eigen::Vector3d point_end)
{
  Vector3d line1 = point_end - point_st;
  const double n = line1.norm();
  if (n < 1e-9) return 0.0;
  Vector3d line2 = point_insert - point_st;
  return double(line2.cross(line1).norm() / n);
}

Vector3d Astarpath::getPosPoly(MatrixXd polyCoeff, int k, double t) {
  Vector3d ret;
  int _poly_num1D = (int)polyCoeff.cols() / 3;
  for (int dim = 0; dim < 3; dim++) {
    VectorXd coeff = (polyCoeff.row(k)).segment(dim * _poly_num1D, _poly_num1D);
    VectorXd time = VectorXd::Zero(_poly_num1D);

    for (int j = 0; j < _poly_num1D; j++)
      if (j == 0)
        time(j) = 1.0;
      else
        time(j) = pow(t, j);

    ret(dim) = coeff.dot(time);
  }
  return ret;
}

int Astarpath::safeCheck(MatrixXd polyCoeff, VectorXd time) {
  int unsafe_segment = -1;  // -1 表示整个轨迹是安全的

  // 使用保守的采样步长
  double delta_t = resolution / 1.0;
  double t = delta_t;
  Vector3d advancePos;

  for (int i = 0; i < polyCoeff.rows(); i++) {
    while (t < time(i)) {
      advancePos = getPosPoly(polyCoeff, i, t);
      if (isOccupied(coord2gridIndex(advancePos))) {
        unsafe_segment = i;
        break;
      }
      t += delta_t;
    }
    if (unsafe_segment != -1) {
      break;
    } else {
      t = delta_t;
    }
  }
  return unsafe_segment;
}

void Astarpath::resetOccupy() {
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++) {
        data[i * GLYZ_SIZE + j * GRID_Z_SIZE + k] = 0;
        data_raw[i * GLYZ_SIZE + j * GRID_Y_SIZE + k] = 0;
      }
}
