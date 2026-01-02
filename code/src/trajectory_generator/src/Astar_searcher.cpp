#include "Astar_searcher.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <queue>
#include <vector>

#include <ros/ros.h>

using namespace std;
using namespace Eigen;

// ============================================================
// Tunables (with ROS params override)
// ============================================================

// Weighted A*: f = g + w*h
static constexpr double kWAstarWDefault = 1.02;
// heuristic tie-breaker
static constexpr double kTieBreaker = 1.0 + 1e-4;
// discourage unnecessary vertical moves
static constexpr double kZMovePenalty = 1.25;
// additional penalty when the move introduces vertical component after horizontal
static constexpr double kZTurnPenaltyWDefault = 0.6;
// generic turning penalty weight
static constexpr double kTurnPenaltyWDefault  = 0.35;
// prefer staying near nominal altitude (start z)
static constexpr double kZDeviationWDefault   = 0.15;
// RViz goal z=0 fallback
static constexpr double kMinGoalZUp = 1e-3;

// start/goal snap search radius (cells)
static constexpr int kNearestFreeMaxRCells = 12;

// LoS sampling
static inline double segStep(double res) { return std::max(0.05, 0.5 * res); }

// forward cone probing (used only for adaptive split; keep)
static constexpr double kForwardConeHalfAngleDeg = 40.0;
static constexpr double kForwardLookaheadM       = 20.0;
static constexpr int    kForwardRaysYaw          = 3;
static constexpr int    kForwardRaysPitch        = 2;
static constexpr double kForwardProbeStepM       = 0.20;

// ============================================================
// Continuous endpoint anchoring
// ============================================================
namespace {
static Eigen::Vector3d g_start_cont(0, 0, 0);
static Eigen::Vector3d g_goal_cont(0, 0, 0);
static bool g_have_cont = false;
static bool g_reached_goal = false;
} // namespace

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

  // IMPORTANT:
  // - 这里的膨胀只用于把“点云离散化”后的栅格障碍连成片；
  // - 真正的安全距离主要由 A* 的 hard-clearance / safeCheck 来保证。
  // 过大的 set_barrier 膨胀会把可通过的“缝”堵死，导致后期 A* time budget hit/乱规划。
  double infl_xy_base = 0.0;
  double infl_z_base  = 0.0;
  ros::param::param("~astar_inflation_xy_base", infl_xy_base, 0.0);
  ros::param::param("~astar_inflation_z_base",  infl_z_base,  0.0);

  const double infl_xy_m = std::max(0.0, infl_xy_base + margin_m);
  const double infl_z_m  = std::max(0.0, infl_z_base  + 0.5 * margin_m);

  const int infl_xy = std::max(0, (int)std::ceil(infl_xy_m / resolution));
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
// Clearance helpers (hard/soft)
// ============================================================

namespace {

// clamp helper
inline double clampd(double v, double a, double b) {
  return std::min(std::max(v, a), b);
}

} // namespace

// compute nearest obstacle distance (meters) within a capped neighborhood.
// if none found within max_r_cells => return (max_r_cells + 1) * resolution
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
// Successors (26-neighborhood + NO corner cutting + hard clearance)
// ============================================================

inline void Astarpath::AstarGetSucc(MappingNodePtr currentPtr,
                                    vector<MappingNodePtr> &neighborPtrSets,
                                    vector<double> &edgeCostSets)
{
  neighborPtrSets.clear();
  edgeCostSets.clear();

  // hard clearance params
  int hard_xy_cells = 1, hard_z_cells = 0;
  ros::param::param("~astar_hard_clearance_xy", hard_xy_cells, 1);
  ros::param::param("~astar_hard_clearance_z",  hard_z_cells,  0);

  const int x = currentPtr->index(0);
  const int y = currentPtr->index(1);
  const int z = currentPtr->index(2);

  // “逃逸模式”：若当前点处于 hard-clearance 区域内，允许扩展一些同样贴近障碍的点，
  // 但要求它们能显著增大与障碍的距离（逐步爬出），否则会出现 openset exhausted/no progress 卡死。
  const bool cur_too_close = isTooCloseHard(currentPtr->index, hard_xy_cells, hard_z_cells);
  const double cur_clr = nearestObsDistM(currentPtr->index, 6);

  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;

        int nx = x + dx, ny = y + dy, nz = z + dz;
        if (nx < 0 || nx >= GRID_X_SIZE || ny < 0 || ny >= GRID_Y_SIZE ||
            nz < 0 || nz >= GRID_Z_SIZE)
          continue;

        Vector3i nidx(nx, ny, nz);
        if (isOccupied(nidx)) continue;
        // allow goal cell even if it violates hard-clearance (but never if occupied)
        if (nidx != goalIdx && isTooCloseHard(nidx, hard_xy_cells, hard_z_cells)) {
          // escape: allow only if it increases clearance vs current
          if (!(cur_too_close && (nearestObsDistM(nidx, 6) > cur_clr + 1e-3))) continue;
        }

        // no corner cutting (also with clearance)
        Vector3i ax(x + dx, y, z);
        Vector3i bx(x, y + dy, z);
        Vector3i cx(x, y, z + dz);

        // 在逃逸模式下，角点/边的 clearance 检查会把所有可行邻居剪没；
        // 因此这里对“占据”仍严格禁止，但 clearance 只在非逃逸模式下强制。
        if (dx != 0 && isOccupied(ax)) continue;
        if (dy != 0 && isOccupied(bx)) continue;
        if (dz != 0 && isOccupied(cx)) continue;
        if (!cur_too_close) {
          if (dx != 0 && isTooCloseHard(ax, hard_xy_cells, hard_z_cells)) continue;
          if (dy != 0 && isTooCloseHard(bx, hard_xy_cells, hard_z_cells)) continue;
          if (dz != 0 && isTooCloseHard(cx, hard_xy_cells, hard_z_cells)) continue;
        }

        if (dx != 0 && dy != 0) {
          Vector3i cxy(x + dx, y + dy, z);
          if (isOccupied(cxy)) continue;
          if (!cur_too_close && isTooCloseHard(cxy, hard_xy_cells, hard_z_cells)) continue;
        }
        if (dx != 0 && dz != 0) {
          Vector3i cxz(x + dx, y, z + dz);
          if (isOccupied(cxz)) continue;
          if (!cur_too_close && isTooCloseHard(cxz, hard_xy_cells, hard_z_cells)) continue;
        }
        if (dy != 0 && dz != 0) {
          Vector3i cyz(x, y + dy, z + dz);
          if (isOccupied(cyz)) continue;
          if (!cur_too_close && isTooCloseHard(cyz, hard_xy_cells, hard_z_cells)) continue;
        }

        neighborPtrSets.push_back(Map_Node[nx][ny][nz]);

        double base = std::sqrt(double(dx * dx + dy * dy + dz * dz));
        if (dz != 0) base *= kZMovePenalty;
        edgeCostSets.push_back(base);
      }
    }
  }
}

// 3D diagonal-distance heuristic + tie-breaker
double Astarpath::getHeu(MappingNodePtr node1, MappingNodePtr node2) {
  int dx = std::abs(node1->index(0) - node2->index(0));
  int dy = std::abs(node1->index(1) - node2->index(1));
  int dz = std::abs(node1->index(2) - node2->index(2));

  int d1 = std::min(dx, std::min(dy, dz));
  int d3 = std::max(dx, std::max(dy, dz));
  int d2 = dx + dy + dz - d1 - d3;

  const double D  = 1.0;
  const double D2 = std::sqrt(2.0);
  const double D3 = std::sqrt(3.0);

  double heu = (double)d1 * D3 + (double)(d2 - d1) * D2 + (double)(d3 - d2) * D;
  return kTieBreaker * heu;
}

// ============================================================
// A* Search (robust + angle/clearance penalties + safe fallback)
// ============================================================

bool Astarpath::AstarSearch(Vector3d start_pt, Vector3d end_pt)
{
  ros::Time t0 = ros::Time::now();

  resetUsedGrids();
  Openset.clear();
  terminatePtr = NULL;

  // params
  static bool inited = false;
  static double max_search_time = 0.20;        // seconds
  static double analytic_connect_dist = 8.0;   // meters
  static double w_astar = kWAstarWDefault;
  static double turn_w  = kTurnPenaltyWDefault;
  static double zturn_w = kZTurnPenaltyWDefault;
  static double zdev_w  = kZDeviationWDefault;

  // soft clearance
  static double soft_range_m = 0.60;
  static int    soft_scan_cells = 6;
  static double soft_w = 0.80;

  // hard clearance neighborhood
  static int hard_xy_cells = 1;
  static int hard_z_cells  = 0;

  if (!inited) {
    ros::param::param("~astar_max_search_time", max_search_time, 0.20);
    ros::param::param("~astar_analytic_connect_dist", analytic_connect_dist, 8.0);
    ros::param::param("~astar_w", w_astar, kWAstarWDefault);

    ros::param::param("~astar_turn_penalty_w", turn_w, kTurnPenaltyWDefault);
    ros::param::param("~astar_zturn_penalty_w", zturn_w, kZTurnPenaltyWDefault);
    ros::param::param("~astar_zdev_penalty_w", zdev_w, kZDeviationWDefault);

    ros::param::param("~astar_soft_clearance_range_m", soft_range_m, 0.60);
    ros::param::param("~astar_soft_clearance_scan_cells", soft_scan_cells, 6);
    ros::param::param("~astar_soft_clearance_weight", soft_w, 0.80);

    ros::param::param("~astar_hard_clearance_xy", hard_xy_cells, 1);
    ros::param::param("~astar_hard_clearance_z",  hard_z_cells,  0);

    inited = true;
  } else {
    // allow runtime tuning
    ros::param::get("~astar_max_search_time", max_search_time);
    ros::param::get("~astar_w", w_astar);
    ros::param::get("~astar_turn_penalty_w", turn_w);
    ros::param::get("~astar_zturn_penalty_w", zturn_w);
    ros::param::get("~astar_zdev_penalty_w", zdev_w);
    ros::param::get("~astar_soft_clearance_range_m", soft_range_m);
    ros::param::get("~astar_soft_clearance_scan_cells", soft_scan_cells);
    ros::param::get("~astar_soft_clearance_weight", soft_w);
    ros::param::get("~astar_hard_clearance_xy", hard_xy_cells);
    ros::param::get("~astar_hard_clearance_z", hard_z_cells);
  }

  // RViz may send z=0; keep current altitude
  if (end_pt(2) <= gl_zl + kMinGoalZUp) end_pt(2) = start_pt(2);

  // record continuous endpoints
  g_start_cont = start_pt;
  g_goal_cont  = end_pt;
  g_have_cont  = true;
  g_reached_goal = false;

  auto clampIntoMap = [&](Vector3d &p) {
    p(0) = std::min(std::max(p(0), gl_xl + 1e-6), gl_xu - 1e-6);
    p(1) = std::min(std::max(p(1), gl_yl + 1e-6), gl_yu - 1e-6);
    p(2) = std::min(std::max(p(2), gl_zl + 1e-6), gl_zu - 1e-6);
  };
  clampIntoMap(start_pt);
  clampIntoMap(end_pt);

  Vector3i start_idx = coord2gridIndex(start_pt);
  Vector3i end_idx   = coord2gridIndex(end_pt);
  goalIdx = end_idx;

  // LoS / inside
  const double los_step = segStep(resolution);

  auto insideMap = [&](const Vector3d& p) -> bool {
    return (p(0) >= gl_xl && p(0) < gl_xu &&
            p(1) >= gl_yl && p(1) < gl_yu &&
            p(2) >= gl_zl && p(2) < gl_zu);
  };

  auto pointSafe = [&](const Vector3d& p) -> bool {
    if (!insideMap(p)) return false;
    Vector3i idx = coord2gridIndex(p);
    if (isOccupied(idx)) return false;
    if (isTooCloseHard(idx, hard_xy_cells, hard_z_cells)) return false;
    return true;
  };

  // endpoint-safe: allow start/goal to be close to obstacles, but never inside obstacles/outside map
  auto pointFree = [&](const Vector3d& p) -> bool {
    if (!insideMap(p)) return false;
    Vector3i idx = coord2gridIndex(p);
    if (isOccupied(idx)) return false;
    return true;
  };

  auto segmentSafe = [&](const Vector3d& a, const Vector3d& b) -> bool {
    const double L = (b - a).norm();
    const int N = std::max(1, (int)std::ceil(L / los_step));
    for (int i = 0; i <= N; ++i) {
      const double t = (double)i / (double)N;
      const Vector3d p = a + t * (b - a);
      const bool endpoint = (i == 0 || i == N);
      if (endpoint) {
        if (!pointFree(p)) return false;
      } else {
        if (!pointSafe(p)) return false;
      }
    }
    return true;
  };

  // ==========================================================
  // Better nearest-free: choose the “best” by clearance + dz preference
  // ==========================================================
  auto nearestFreeBest = [&](const Vector3i& seed, Vector3i &out_free) -> bool {
    // if already safe
    if (!isOccupied(seed) && !isTooCloseHard(seed, hard_xy_cells, hard_z_cells)) {
      out_free = seed;
      return true;
    }

    const double seed_z = gridIndex2coord(seed)(2);

    bool found = false;
    double bestScore = 1e100;
    Vector3i bestIdx = seed;

    const int R = kNearestFreeMaxRCells;
    for (int r = 1; r <= R; ++r) {
      for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
          for (int dz = -r; dz <= r; ++dz) {
            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r) continue;

            Vector3i q = seed + Vector3i(dx, dy, dz);
            if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
                q(1) < 0 || q(1) >= GRID_Y_SIZE ||
                q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;

            if (isOccupied(q)) continue;
            if (isTooCloseHard(q, hard_xy_cells, hard_z_cells)) continue;

            const double dcell = std::sqrt((double)(dx*dx + dy*dy + dz*dz));
            const double qz = gridIndex2coord(q)(2);
            const double dzm = std::fabs(qz - seed_z);

            // clearance term (prefer large)
            const double clr = nearestObsDistM(q, soft_scan_cells); // capped
            const double invc = 1.0 / std::max(0.05, clr);

            // score: small distance, small dz, large clearance
            const double score = 1.0 * dcell + 2.0 * (dzm / std::max(1e-6, resolution)) + 2.5 * invc;

            if (score < bestScore) {
              bestScore = score;
              bestIdx = q;
              found = true;
            }
          }
        }
      }
      if (found) break;
    }

    if (!found) return false;
    out_free = bestIdx;
    return true;
  };

  // snap start / goal
  // 只在“真正落在占据栅格内”时才 snap，避免仅因 hard-clearance 导致起点跳变（倒飞/抖动）
  if (isOccupied(start_idx)) {
    Vector3i sfree;
    if (nearestFreeBest(start_idx, sfree)) {
      start_idx = sfree;
      start_pt  = gridIndex2coord(start_idx);
      g_start_cont = start_pt;
      ROS_WARN("[A*] Start occupied, snapped to best nearby free cell.");
    } else {
      ROS_ERROR("[A*] Start occupied and no nearby free cell found.");
      return false;
    }
  } else if (isTooCloseHard(start_idx, hard_xy_cells, hard_z_cells)) {
    ROS_WARN_THROTTLE(0.5, "[A*] Start too close to obstacle (hard-clearance), keep continuous start and plan away.");
  }

  if (isOccupied(end_idx)) {
    Vector3i gfree;
    if (nearestFreeBest(end_idx, gfree)) {
      end_idx = gfree;
      end_pt  = gridIndex2coord(end_idx);
      g_goal_cont = end_pt;
      goalIdx = end_idx;
      ROS_WARN("[A*] Goal occupied, snapped to best nearby free cell.");
    } else {
      ROS_ERROR("[A*] Goal occupied and no nearby free cell found.");
      return false;
    }
  } else if (isTooCloseHard(end_idx, hard_xy_cells, hard_z_cells)) {
    // goal 允许违反 hard-clearance（只要不在障碍内），便于靠近目标；AstarGetSucc 已对 goalIdx 放行
    ROS_WARN_THROTTLE(0.5, "[A*] Goal too close to obstacle (hard-clearance), allow goal but enforce clearance on path interior.");
  }

  MappingNodePtr startPtr = Map_Node[start_idx(0)][start_idx(1)][start_idx(2)];
  MappingNodePtr endPtr   = Map_Node[end_idx(0)][end_idx(1)][end_idx(2)];

  startPtr->coord = gridIndex2coord(start_idx);
  endPtr->coord   = gridIndex2coord(end_idx);

  if (start_idx == end_idx) {
    terminatePtr = startPtr;
    g_reached_goal = true;
    return true;
  }

  // adapt time budget in dense area: if start clearance small, allow more time
  {
    const double start_clr = nearestObsDistM(start_idx, soft_scan_cells);
    if (start_clr < 0.8) {
      max_search_time = std::min(0.50, max_search_time * 2.0);
    }
  }

  const double nominal_z = startPtr->coord(2);

  startPtr->g_score = 0.0;
  startPtr->f_score = startPtr->g_score + w_astar * getHeu(startPtr, endPtr);
  startPtr->id = 1;
  startPtr->Father = NULL;

  Openset.insert(make_pair(startPtr->f_score, startPtr));

  vector<MappingNodePtr> neighborPtrSets;
  vector<double> edgeCostSets;

  // best-so-far for diagnostic only (do NOT blindly execute it)
  MappingNodePtr bestPtr = startPtr;
  double best_h = getHeu(startPtr, endPtr);

  auto computeMoveDir = [&](MappingNodePtr a, MappingNodePtr b) -> Vector3d {
    Vector3d d = (gridIndex2coord(b->index) - gridIndex2coord(a->index));
    double n = d.norm();
    if (n < 1e-6) return Vector3d::Zero();
    return d / n;
  };

  // NOTE: 原先的 retreat 逻辑会导致无人机“往回飞/停在空旷处卡死”，已移除。

  while (!Openset.empty()) {
    // time budget - 延长搜索时间，确保尽可能找到完整路径
    const double elapsed = (ros::Time::now() - t0).toSec();
    if (elapsed > max_search_time * 2.0) {
      // 只有在搜索时间非常长时才中断
      // 优先使用 bestPtr（最接近目标的点）作为“部分路径终点”
      const double clr = nearestObsDistM(bestPtr->index, soft_scan_cells);
      
      if (bestPtr->index == goalIdx) {
        terminatePtr = bestPtr;
        g_reached_goal = true;
        return true;
      }

      // 若毫无进展（bestPtr 仍为 start），直接返回失败
      if (bestPtr == startPtr) {
        ROS_WARN("[A*] time budget hit but no progress (clr=%.2f).", clr);
        return false;
      }

      // 返回“部分路径”（到 bestPtr），由上层状态机继续重规划直至到达目标
      terminatePtr = bestPtr;
      g_reached_goal = false;
      ROS_WARN("[A*] time budget hit. Return partial path to best-so-far (clr=%.2f, h=%.2f).", clr, best_h);
      return true;
    }

    auto it = Openset.begin();
    const double popped_f = it->first;
    MappingNodePtr currentPtr = it->second;
    Openset.erase(it);

    if (currentPtr->id != 1) continue;
    if (popped_f > currentPtr->f_score + 1e-6) continue;

    const double hcur = getHeu(currentPtr, endPtr);
    if (hcur < best_h) { best_h = hcur; bestPtr = currentPtr; }

    if (currentPtr->index == goalIdx) {
      terminatePtr = currentPtr;
      g_reached_goal = true;
      return true;
    }

    // analytic connect near goal
    const double dist_goal = (endPtr->coord - currentPtr->coord).norm();
    if (dist_goal <= analytic_connect_dist && segmentSafe(currentPtr->coord, endPtr->coord)) {
      endPtr->Father  = currentPtr;
      endPtr->g_score = currentPtr->g_score + dist_goal;
      endPtr->f_score = endPtr->g_score;
      endPtr->id = 1;
      terminatePtr = endPtr;
      g_reached_goal = true;
      return true;
    }

    currentPtr->id = -1;

    AstarGetSucc(currentPtr, neighborPtrSets, edgeCostSets);

    for (unsigned int i = 0; i < neighborPtrSets.size(); i++) {
      MappingNodePtr neighborPtr = neighborPtrSets[i];
      if (neighborPtr->id == -1) continue;

      // base move cost
      double g_inc = edgeCostSets[i];

      // soft clearance penalty
      g_inc += soft_w * softClearancePenalty(neighborPtr->index, soft_range_m, soft_scan_cells);

      // altitude deviation penalty (keep z stable unless needed)
      g_inc += zdev_w * std::fabs(gridIndex2coord(neighborPtr->index)(2) - nominal_z);

      // turning penalty (angle-aware)
      if (currentPtr->Father != NULL) {
        Vector3d d_prev = computeMoveDir(currentPtr->Father, currentPtr);
        Vector3d d_new  = computeMoveDir(currentPtr, neighborPtr);
        if (d_prev.norm() > 1e-6 && d_new.norm() > 1e-6) {
          double cosang = clampd(d_prev.dot(d_new), -1.0, 1.0);
          double turn_cost = (1.0 - cosang); // 0..2
          g_inc += turn_w * turn_cost;

          // extra penalty if “horizontal then vertical” or “vertical then horizontal”
          const double prev_v = std::fabs(d_prev(2));
          const double new_v  = std::fabs(d_new(2));
          g_inc += zturn_w * std::fabs(new_v - prev_v);
        }
      }

      double tentative_g_score = currentPtr->g_score + g_inc;

      if (neighborPtr->id == 0) {
        neighborPtr->Father  = currentPtr;
        neighborPtr->g_score = tentative_g_score;
        neighborPtr->f_score = tentative_g_score + w_astar * getHeu(neighborPtr, endPtr);
        neighborPtr->id = 1;
        neighborPtr->coord = gridIndex2coord(neighborPtr->index);
        Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
      } else if (neighborPtr->id == 1) {
        if (tentative_g_score + 1e-9 < neighborPtr->g_score) {
          neighborPtr->Father  = currentPtr;
          neighborPtr->g_score = tentative_g_score;
          neighborPtr->f_score = tentative_g_score + w_astar * getHeu(neighborPtr, endPtr);
          neighborPtr->coord = gridIndex2coord(neighborPtr->index);
          Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
        }
      }
    }
  }

  // openset exhausted: 使用 bestPtr，确保向目标方向前进
  // 不使用 retreat，避免无人机往回飞；返回部分路径由上层继续重规划
  if (bestPtr != startPtr) {
    terminatePtr = bestPtr;
    g_reached_goal = false;
    ROS_WARN("[A*] openset exhausted. Using best-so-far toward goal (h=%.2f).", best_h);
    return true;
  }
  
  // 如果完全没有进展，返回失败
  ROS_ERROR("[A*] openset exhausted with no progress.");
  return false;
}

vector<Vector3d> Astarpath::getPath() {
  vector<Vector3d> path;
  if (terminatePtr == NULL) return path;

  vector<MappingNodePtr> front_path;
  MappingNodePtr cur = terminatePtr;
  while (cur != NULL) {
    cur->coord = gridIndex2coord(cur->index);
    front_path.push_back(cur);
    cur = cur->Father;
  }

  path.reserve(front_path.size());
  for (int i = (int)front_path.size() - 1; i >= 0; --i)
    path.push_back(front_path[i]->coord);

  // anchor endpoints - 关键修复：始终保留起点和目标点
  if (g_have_cont && !path.empty()) {
    path.front() = g_start_cont;
    // 仅当真正到达 goalIdx 时才用连续目标点覆盖末端
    if (g_reached_goal) path.back() = g_goal_cont;
  }

  ROS_INFO_THROTTLE(1.0, "[getPath] size=%d reached_goal=%d",
                    (int)path.size(), (int)g_reached_goal);

  return path;
}

// ============================================================
// Path post-processing (STRICT safety, no “force insert colliding point”)
// ============================================================

std::vector<Vector3d> Astarpath::pathSimplify(const vector<Vector3d> &path,
                                              double path_resolution)
{
  if (path.size() < 2) return path;

  // params (runtime)
  int hard_xy_cells = 1, hard_z_cells = 0;
  ros::param::param("~astar_hard_clearance_xy", hard_xy_cells, 1);
  ros::param::param("~astar_hard_clearance_z",  hard_z_cells,  0);

  double max_climb_deg = 20.0; // limit near-vertical segments
  ros::param::param("~astar_max_climb_deg", max_climb_deg, 20.0);
  const double tan_max_climb = std::tan(max_climb_deg * M_PI / 180.0);

  const double los_step = segStep(resolution);

  auto insideMap = [&](const Vector3d& p) -> bool {
    return (p(0) >= gl_xl && p(0) < gl_xu &&
            p(1) >= gl_yl && p(1) < gl_yu &&
            p(2) >= gl_zl && p(2) < gl_zu);
  };

  auto clampIntoMap = [&](Vector3d &p) {
    p(0) = std::min(std::max(p(0), gl_xl + 1e-6), gl_xu - 1e-6);
    p(1) = std::min(std::max(p(1), gl_yl + 1e-6), gl_yu - 1e-6);
    p(2) = std::min(std::max(p(2), gl_zl + 1e-6), gl_zu - 1e-6);
  };

  auto pointSafe = [&](const Vector3d& p) -> bool {
    if (!insideMap(p)) return false;
    Vector3i idx = coord2gridIndex(p);
    if (isOccupied(idx)) return false;
    if (isTooCloseHard(idx, hard_xy_cells, hard_z_cells)) return false;
    return true;
  };

  // endpoint-safe：端点允许贴近障碍（但不能在障碍内/越界）
  auto pointFree = [&](const Vector3d& p) -> bool {
    if (!insideMap(p)) return false;
    Vector3i idx = coord2gridIndex(p);
    if (isOccupied(idx)) return false;
    return true;
  };

  auto segmentSafe = [&](const Vector3d& a, const Vector3d& b) -> bool {
    const double L = (b - a).norm();
    const int N = std::max(1, (int)std::ceil(L / los_step));
    for (int i = 0; i <= N; ++i) {
      const double t = (double)i / (double)N;
      const Vector3d p = a + t * (b - a);
      const bool endpoint = (i == 0 || i == N);
      if (endpoint) {
        if (!pointFree(p)) return false;
      } else {
        if (!pointSafe(p)) return false;
      }
    }
    return true;
  };

  // smarter sanitize: local best by clearance
  auto sanitizePoint = [&](Vector3d p) -> Vector3d {
    clampIntoMap(p);
    if (pointSafe(p)) return p;

    Vector3i idx = coord2gridIndex(p);
    Vector3i best = idx;
    double bestClr = -1.0;

    for (int dx = -2; dx <= 2; ++dx)
      for (int dy = -2; dy <= 2; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          Vector3i q = idx + Vector3i(dx, dy, dz);
          if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
              q(1) < 0 || q(1) >= GRID_Y_SIZE ||
              q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;
          if (isOccupied(q) || isTooCloseHard(q, hard_xy_cells, hard_z_cells)) continue;
          const double clr = nearestObsDistM(q, 6);
          if (clr > bestClr) { bestClr = clr; best = q; }
        }

    return gridIndex2coord(best);
  };

  // 1) de-dup
  std::vector<Vector3d> clean;
  clean.reserve(path.size());
  clean.push_back(path.front());
  for (size_t i = 1; i < path.size(); ++i) {
    if ((path[i] - clean.back()).norm() > 1e-4) clean.push_back(path[i]);
  }
  if (clean.size() < 2) return clean;

  // anchor endpoints
  // - 起点：优先保持连续坐标，只有落在障碍内才 sanitize（避免仅因 clearance 导致起点跳变）
  // - 终点：只有 A* 真正到达 goalIdx 时才用连续目标点覆盖
  if (g_have_cont) {
    Vector3d s = g_start_cont;
    clampIntoMap(s);
    if (!pointFree(s)) s = sanitizePoint(s);  // occupied/outside only
    clean.front() = s;

    if (g_reached_goal) {
      Vector3d g = g_goal_cont;
      clampIntoMap(g);
      if (!pointFree(g)) g = sanitizePoint(g);
      clean.back() = g;
    } else {
      clean.back() = sanitizePoint(clean.back());
    }
  } else {
    clean.front() = sanitizePoint(clean.front());
    clean.back()  = sanitizePoint(clean.back());
  }
  
  // 2) greedy LoS prune (strict)
  std::vector<Vector3d> pruned;
  pruned.reserve(clean.size());
  pruned.push_back(clean.front());
  size_t i = 0;
  while (i + 1 < clean.size()) {
    size_t best = i + 1;
    for (size_t j = clean.size() - 1; j > i; --j) {
      if (segmentSafe(clean[i], clean[j])) { best = j; break; }
    }
    pruned.push_back(clean[best]);
    i = best;
  }

  // 3) light corner fillet (only if all involved segments safe)
  const double fillet_ratio = 0.25;
  const double min_corner = std::max(0.8, 2.0 * path_resolution);

  std::vector<Vector3d> fillet;
  fillet.reserve(pruned.size() * 2);
  fillet.push_back(pruned.front());
  for (size_t k = 1; k + 1 < pruned.size(); ++k) {
    const Vector3d a = fillet.back();
    const Vector3d b = pruned[k];
    const Vector3d c = pruned[k + 1];

    const Vector3d ab = b - a;
    const Vector3d bc = c - b;
    const double Lab = ab.norm();
    const double Lbc = bc.norm();
    if (Lab < 1e-6 || Lbc < 1e-6) continue;

    const Vector3d u = ab / Lab;
    const Vector3d v = bc / Lbc;
    const double cosang = std::max(-1.0, std::min(1.0, u.dot(v)));
    const double ang = std::acos(cosang);

    if (ang > (25.0 * M_PI / 180.0) && (Lab > min_corner) && (Lbc > min_corner)) {
      const double d1 = std::max(min_corner * 0.5, fillet_ratio * Lab);
      const double d2 = std::max(min_corner * 0.5, fillet_ratio * Lbc);
      Vector3d b1 = sanitizePoint(b - u * std::min(d1, 0.45 * Lab));
      Vector3d b2 = sanitizePoint(b + v * std::min(d2, 0.45 * Lbc));

      if (segmentSafe(fillet.back(), b1) &&
          segmentSafe(b1, b2) &&
          segmentSafe(b2, c)) {
        fillet.push_back(b1);
        fillet.push_back(b2);
        continue;
      }
    }
    fillet.push_back(b);
  }
  fillet.push_back(pruned.back());

  // forward-cone obstacle probing (use strict pointSafe)
  auto forwardObstacleDist = [&](const Vector3d& p0, const Vector3d& dir_unit) -> double {
    Vector3d d = dir_unit;
    if (d.norm() < 1e-6) return 0.0;

    Vector3d up(0, 0, 1);
    if (std::fabs(d.dot(up)) > 0.95) up = Vector3d(0, 1, 0);
    Vector3d right = d.cross(up);
    if (right.norm() < 1e-6) right = Vector3d(1, 0, 0);
    right.normalize();
    Vector3d up2 = right.cross(d).normalized();

    const double th = kForwardConeHalfAngleDeg * M_PI / 180.0;
    const double tth = std::tan(th);

    std::vector<Vector3d> dirs;
    dirs.reserve((2 * kForwardRaysYaw + 1) * (2 * kForwardRaysPitch + 1));
    for (int iy = -kForwardRaysYaw; iy <= kForwardRaysYaw; ++iy) {
      for (int ip = -kForwardRaysPitch; ip <= kForwardRaysPitch; ++ip) {
        Vector3d dd = d
          + (double)iy * (tth / std::max(1, kForwardRaysYaw)) * right
          + (double)ip * (tth / std::max(1, kForwardRaysPitch)) * up2;
        dd.normalize();
        dirs.push_back(dd);
      }
    }

    double best = kForwardLookaheadM + 1.0;
    for (const auto& rd : dirs) {
      for (double s = kForwardProbeStepM; s <= kForwardLookaheadM; s += kForwardProbeStepM) {
        Vector3d q = p0 + s * rd;
        if (!insideMap(q)) { best = std::min(best, s); break; }
        if (!pointSafe(q)) { best = std::min(best, s); break; }
      }
    }
    return best;
  };

  // adaptive split lengths
  const double base_open = std::min(6.0, std::max(3.5, 10.0 * path_resolution));
  const double base_mid  = std::min(3.0, std::max(2.0,  6.0 * path_resolution));
  const double base_near = std::min(1.5, std::max(1.0,  3.5 * path_resolution));
  const double kMinSplitLenM = std::max(0.8, 2.5 * path_resolution);

  std::vector<Vector3d> out;
  out.reserve(fillet.size() * 2);
  out.push_back(fillet.front());

  // detour search: try to find a safe intermediate point around obstacle
  auto findDetour = [&](const Vector3d& a, const Vector3d& b, Vector3d& det) -> bool {
    Vector3d mid = 0.5 * (a + b);
    Vector3d d = (b - a);
    double L = d.norm();
    if (L < 1e-6) return false;
    d /= L;

    Vector3d perp(-d.y(), d.x(), 0.0);
    if (perp.norm() < 1e-6) perp = Vector3d(1, 0, 0);
    perp.normalize();

    // try offsets (left/right) and slight vertical offsets
    double bestClr = -1.0;
    Vector3d bestP;

    for (int step = 1; step <= 8; ++step) {
      double r = step * std::max(0.6, 2.0 * resolution);
      for (int sgn : {-1, 1}) {
        for (int vz = -2; vz <= 2; ++vz) {
          Vector3d cand = mid + (double)sgn * r * perp + (double)vz * resolution * Vector3d(0,0,1);
          cand = sanitizePoint(cand);
          if (!pointSafe(cand)) continue;
          if (!segmentSafe(a, cand)) continue;
          if (!segmentSafe(cand, b)) continue;

          Vector3i cidx = coord2gridIndex(cand);
          double clr = nearestObsDistM(cidx, 6);
          if (clr > bestClr) {
            bestClr = clr;
            bestP = cand;
          }
        }
      }
      if (bestClr > 0.0) break;
    }

    if (bestClr <= 0.0) return false;
    det = bestP;
    return true;
  };

  // enforce climb angle: if nearly vertical, try detour
  auto enforceClimb = [&](const Vector3d& a, const Vector3d& b, Vector3d& maybe_det) -> bool {
    Vector3d d = b - a;
    double dz = std::fabs(d.z());
    double dxy = std::sqrt(d.x()*d.x() + d.y()*d.y());
    if (dxy < 1e-3 && dz > 0.3) {
      // near-vertical: must detour
      return findDetour(a, b, maybe_det);
    }
    if (dxy > 1e-3 && (dz / dxy) > tan_max_climb) {
      // too steep: try detour as well
      return findDetour(a, b, maybe_det);
    }
    return false;
  };

  // strict pushChecked: NEVER insert points if the segment is unsafe
  std::function<bool(const Vector3d&, const Vector3d&, int)> pushSeg;
  pushSeg = [&](const Vector3d& a, const Vector3d& b, int depth) -> bool {
    if (segmentSafe(a, b)) {
      if ((b - out.back()).norm() > 1e-4) out.push_back(b);
      return true;
    }

    // before splitting, see if it's a steep segment; attempt detour
    Vector3d det;
    if (enforceClimb(a, b, det)) {
      if (!pushSeg(a, det, depth + 1)) return false;
      return pushSeg(det, b, depth + 1);
    }

    // try a geometric detour even for non-steep segments
    if (findDetour(a, b, det)) {
      if (!pushSeg(a, det, depth + 1)) return false;
      return pushSeg(det, b, depth + 1);
    }

    if (depth >= 8) {
      // FAIL: do NOT force insert colliding points
      return false;
    }

    Vector3d mid = sanitizePoint(0.5 * (a + b));
    if (!pointSafe(mid)) return false;

    if (!pushSeg(a, mid, depth + 1)) return false;
    return pushSeg(mid, b, depth + 1);
  };

  auto pushChecked = [&](const Vector3d& target) -> bool {
    Vector3d p = sanitizePoint(target);
    if (!pointSafe(p)) return false;
    return pushSeg(out.back(), p, 0);
  };

  bool push_failed = false;
  for (size_t s = 0; s + 1 < fillet.size(); ++s) {
    const Vector3d a0 = out.back();
    const Vector3d b0 = fillet[s + 1];
    Vector3d d = b0 - a0;
    const double L = d.norm();
    if (L < 1e-6) continue;
    d /= L;

    const double obs = forwardObstacleDist(a0, d);

    double max_len = base_mid;
    if (obs <= 1.0) max_len = base_near;
    else if (obs <= 3.0) max_len = base_mid;
    else max_len = base_open;

    max_len = std::max(max_len, kMinSplitLenM);

    int N = std::max(1, (int)std::ceil(L / max_len));
    while (N > 1 && (L / (double)N) < kMinSplitLenM) --N;

    for (int k = 1; k <= N; ++k) {
      const double t = (double)k / (double)N;
      Vector3d p = a0 + t * (b0 - a0);

      if (!pushChecked(p)) {
        // stop at last safe prefix
        push_failed = true;
        ROS_WARN("[pathSimplify] pushChecked failed. Fallback to raw A* path.");
        break;
      }
    }
    if (push_failed) break;
  }

  // final de-dup
  std::vector<Vector3d> final_out;
  final_out.reserve(out.size());
  final_out.push_back(out.front());
  for (size_t ii = 1; ii < out.size(); ++ii) {
    if ((out[ii] - final_out.back()).norm() > 1e-4) final_out.push_back(out[ii]);
  }

  // 如果 push 过程失败或输出点太少，则回退到更保守的 clean（基本等同原始 A* 网格路径，避免 out=1 导致规划失败/卡死）
  if (push_failed || final_out.size() < 2) {
    ROS_WARN("[pathSimplify] fallback: push_failed=%d out=%d -> return clean=%d",
             (int)push_failed, (int)final_out.size(), (int)clean.size());
    return clean;
  }

  // 若确实到达了 goalIdx，则用连续目标点覆盖末端（更精确）
  if (g_have_cont && g_reached_goal && !final_out.empty()) {
    final_out.back() = sanitizePoint(g_goal_cont);
  }

  ROS_INFO("[pathSimplify] in=%d clean=%d pruned=%d fillet=%d out=%d",
           (int)path.size(), (int)clean.size(), (int)pruned.size(),
           (int)fillet.size(), (int)final_out.size());

  return final_out;
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
  int unsafe_segment = -1;

  // runtime params
  int hard_xy_cells = 1, hard_z_cells = 0;
  ros::param::param("~astar_hard_clearance_xy", hard_xy_cells, 1);
  ros::param::param("~astar_hard_clearance_z",  hard_z_cells,  0);

  // finer sampling：更密一些以降低“穿过细障碍但未被采样到”的概率
  // 经验值：让采样步长对应的位移不超过 ~0.1m（在 2~3m/s 量级下）
  const double delta_t = std::max(std::min(resolution * 0.25, 0.06), 0.02);

  auto insideMap = [&](const Vector3d& p) -> bool {
    return (p(0) >= gl_xl && p(0) < gl_xu &&
            p(1) >= gl_yl && p(1) < gl_yu &&
            p(2) >= gl_zl && p(2) < gl_zu);
  };

  auto pointSafe = [&](const Vector3d& p) -> bool {
    if (!insideMap(p)) return false;
    Vector3i idx = coord2gridIndex(p);
    if (isOccupied(idx)) return false;
    if (isTooCloseHard(idx, hard_xy_cells, hard_z_cells)) return false;
    return true;
  };

  double t = delta_t;
  Vector3d advancePos;
  for (int i = 0; i < polyCoeff.rows(); i++) {
    while (t <= time(i) + 1e-9) {
      advancePos = getPosPoly(polyCoeff, i, t);
      if (!pointSafe(advancePos)) {
        unsafe_segment = i;
        break;
      }
      t += delta_t;
    }
    if (unsafe_segment != -1) break;
    t = delta_t;
  }
  return unsafe_segment;
}

void Astarpath::resetOccupy() {
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++) {
        data[i * GLYZ_SIZE + j * GRID_Z_SIZE + k] = 0;
        data_raw[i * GLYZ_SIZE + j * GRID_Z_SIZE + k] = 0;
      }
}
