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
// Tunables
// ============================================================

// Weighted A*: f = g + w*h
static constexpr double kWAstarW      = 1.05;
// heuristic tie-breaker
static constexpr double kTieBreaker   = 1.0 + 1e-4;
// discourage unnecessary vertical moves
static constexpr double kZMovePenalty = 1.20;

// RViz goal z=0 fallback
static constexpr double kMinGoalZUp   = 1e-3;

// start/goal snap to nearest free cell radius (cells)
static constexpr int    kNearestFreeMaxRCells = 12;

// LoS sampling
static inline double segStep(double res) { return std::max(0.05, 0.5 * res); }

// Forward cone probing (still useful, but we will additionally use clearance)
static constexpr double kForwardConeHalfAngleDeg = 30.0; // ±30°
static constexpr double kForwardLookaheadM       = 6.0;
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
// Grid/map init & bookkeeping (match header API)
// ============================================================

void Astarpath::begin_grid_map(double _resolution, Vector3d global_xyz_l,
                               Vector3d global_xyz_u, int max_x_id,
                               int max_y_id, int max_z_id) {
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
                            const double coord_z) {
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

  // compatibility with your existing param
  static bool   inited = false;
  static double margin_m = 0.0;
  static double extra_infl_m = 0.10; // NEW: conservative boost to reduce end-segment "擦边"
  if (!inited) {
    ros::param::param<double>("/trajectory_generator_node/map/margin", margin_m, 0.0);
    ros::param::param<double>("~astar_extra_inflation_m", extra_infl_m, 0.10);
    inited = true;
  }

  // Increase baseline inflation (important for polynomial smoothing safety)
  const double infl_xy_m = std::max(0.0, 0.38 + margin_m + extra_infl_m);
  const double infl_z_m  = std::max(0.0, 0.30 + 0.5 * margin_m + 0.5 * extra_infl_m);

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

Vector3i Astarpath::c2i(const Vector3d &pt) { return coord2gridIndex(pt); }

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
// Successors (26-neighborhood + NO corner cutting)
// + soft clearance penalty to keep away from obstacles
// ============================================================

inline void Astarpath::AstarGetSucc(MappingNodePtr currentPtr,
                                    vector<MappingNodePtr> &neighborPtrSets,
                                    vector<double> &edgeCostSets) {
  neighborPtrSets.clear();
  edgeCostSets.clear();

  // soft clearance penalty parameters
  static bool inited = false;
  static int  clear_scan_r = 3;       // cells
  static double clear_w = 0.25;       // weight
  if (!inited) {
    ros::param::param("~astar_clear_scan_r", clear_scan_r, 3);
    ros::param::param("~astar_clear_weight", clear_w, 0.25);
    inited = true;
  }

  auto approxClearanceCells = [&](const Vector3i& idx) -> int {
    if (isOccupied(idx)) return 0;
    for (int r = 1; r <= clear_scan_r; ++r) {
      for (int dx = -r; dx <= r; ++dx)
        for (int dy = -r; dy <= r; ++dy)
          for (int dz = -r; dz <= r; ++dz) {
            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r) continue;
            Vector3i q = idx + Vector3i(dx, dy, dz);
            if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
                q(1) < 0 || q(1) >= GRID_Y_SIZE ||
                q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;
            if (isOccupied(q)) return r;
          }
    }
    return clear_scan_r + 1;
  };

  const int x = currentPtr->index(0);
  const int y = currentPtr->index(1);
  const int z = currentPtr->index(2);

  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;

        int nx = x + dx, ny = y + dy, nz = z + dz;
        if (nx < 0 || nx >= GRID_X_SIZE || ny < 0 || ny >= GRID_Y_SIZE ||
            nz < 0 || nz >= GRID_Z_SIZE)
          continue;

        if (isOccupied(nx, ny, nz)) continue;

        // no corner cutting
        int ax = x + dx, ay = y, az = z;
        int bx = x, by = y + dy, bz = z;
        int cx = x, cy = y, cz = z + dz;

        if (dx != 0 && isOccupied(ax, ay, az)) continue;
        if (dy != 0 && isOccupied(bx, by, bz)) continue;
        if (dz != 0 && isOccupied(cx, cy, cz)) continue;

        if (dx != 0 && dy != 0 && isOccupied(x + dx, y + dy, z)) continue;
        if (dx != 0 && dz != 0 && isOccupied(x + dx, y, z + dz)) continue;
        if (dy != 0 && dz != 0 && isOccupied(x, y + dy, z + dz)) continue;

        MappingNodePtr nb = Map_Node[nx][ny][nz];
        neighborPtrSets.push_back(nb);

        double base = std::sqrt(double(dx * dx + dy * dy + dz * dz));
        if (dz != 0) base *= kZMovePenalty;

        // clearance penalty (soft safety margin)
        const int c = approxClearanceCells(nb->index); // larger is better
        const double invc = 1.0 / (double)std::max(1, c); // 1.. small
        const double penalty = clear_w * invc;

        edgeCostSets.push_back(base + penalty);
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
// A* Search (robust: snap start/goal, time budget, best-so-far fallback)
// + LoS checks with safety radius (buffer) to protect smoothing
// ============================================================

bool Astarpath::AstarSearch(Vector3d start_pt, Vector3d end_pt) {
  ros::Time t0 = ros::Time::now();

  resetUsedGrids();
  Openset.clear();
  terminatePtr = NULL;

  static bool inited = false;
  static double max_search_time = 0.12;        // seconds (slightly increased for final clutter)
  static double analytic_connect_dist = 7.0;   // meters
  static double los_clearance_m = 0.15;        // NEW: LoS safety buffer (meters)
  if (!inited) {
    ros::param::param("~astar_max_search_time", max_search_time, 0.12);
    ros::param::param("~astar_analytic_connect_dist", analytic_connect_dist, 7.0);
    ros::param::param("~astar_los_clearance_m", los_clearance_m, 0.15);
    inited = true;
  }

  if (end_pt(2) <= gl_zl + kMinGoalZUp) end_pt(2) = start_pt(2);

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

  const int r_safe = std::max(0, (int)std::ceil(los_clearance_m / resolution));

  auto insideMap = [&](const Vector3d& p) -> bool {
    return (p(0) >= gl_xl && p(0) < gl_xu &&
            p(1) >= gl_yl && p(1) < gl_yu &&
            p(2) >= gl_zl && p(2) < gl_zu);
  };

  auto nearOccupied = [&](const Vector3i& idx, int r) -> bool {
    if (isOccupied(idx)) return true;
    if (r <= 0) return false;
    for (int dx = -r; dx <= r; ++dx)
      for (int dy = -r; dy <= r; ++dy)
        for (int dz = -r; dz <= r; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          Vector3i q = idx + Vector3i(dx, dy, dz);
          if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
              q(1) < 0 || q(1) >= GRID_Y_SIZE ||
              q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;
          if (isOccupied(q)) return true;
        }
    return false;
  };

  auto approxClearanceCells = [&](const Vector3i& idx, int rmax) -> int {
    if (isOccupied(idx)) return 0;
    for (int r = 1; r <= rmax; ++r) {
      for (int dx = -r; dx <= r; ++dx)
        for (int dy = -r; dy <= r; ++dy)
          for (int dz = -r; dz <= r; ++dz) {
            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r) continue;
            Vector3i q = idx + Vector3i(dx, dy, dz);
            if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
                q(1) < 0 || q(1) >= GRID_Y_SIZE ||
                q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;
            if (isOccupied(q)) return r;
          }
    }
    return rmax + 1;
  };

  // choose a free cell near seed, preferring larger clearance (important at goal/end segment)
  auto bestFreeNearby = [&](const Vector3i& seed, Vector3i &out_free) -> bool {
    if (!isOccupied(seed) && !nearOccupied(seed, r_safe)) { out_free = seed; return true; }

    bool found = false;
    double best_cost = 1e9;
    Vector3i best = seed;

    const int rmax = kNearestFreeMaxRCells;
    const int clear_scan = std::max(3, r_safe + 2);

    for (int r = 1; r <= rmax; ++r) {
      for (int dx = -r; dx <= r; ++dx)
        for (int dy = -r; dy <= r; ++dy)
          for (int dz = -r; dz <= r; ++dz) {
            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r) continue;
            Vector3i q = seed + Vector3i(dx, dy, dz);
            if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
                q(1) < 0 || q(1) >= GRID_Y_SIZE ||
                q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;
            if (isOccupied(q)) continue;
            if (nearOccupied(q, r_safe)) continue; // keep some buffer

            const double dist = std::sqrt(double(dx*dx + dy*dy + dz*dz));
            const int c = approxClearanceCells(q, clear_scan);
            const double clear_term = 1.0 / (double)std::max(1, c); // smaller is better
            const double cost = dist + 1.5 * clear_term; // prefer clearance near obstacles

            if (cost < best_cost) {
              best_cost = cost;
              best = q;
              found = true;
            }
          }
      if (found) break;
    }

    if (found) out_free = best;
    return found;
  };

  // If start/goal are inside/too close to obstacles, snap to a safer nearby cell
  if (isOccupied(start_idx) || nearOccupied(start_idx, r_safe)) {
    Vector3i sfree;
    if (bestFreeNearby(start_idx, sfree)) {
      start_idx = sfree;
      start_pt  = gridIndex2coord(start_idx);
      g_start_cont = start_pt;
      ROS_WARN("[A*] Start unsafe/occupied, snapped to safer free cell.");
    } else {
      ROS_ERROR("[A*] Start unsafe and no nearby safe cell found.");
      return false;
    }
  }

  if (isOccupied(end_idx) || nearOccupied(end_idx, r_safe)) {
    Vector3i gfree;
    if (bestFreeNearby(end_idx, gfree)) {
      end_idx = gfree;
      end_pt  = gridIndex2coord(end_idx);
      g_goal_cont = end_pt;
      goalIdx = end_idx;
      ROS_WARN("[A*] Goal unsafe/occupied, snapped to safer free cell.");
    } else {
      ROS_ERROR("[A*] Goal unsafe and no nearby safe cell found.");
      return false;
    }
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

  const double los_step = segStep(resolution);

  // LoS with safety radius (key change vs plain occupancy LoS)
  auto segmentSafe = [&](const Vector3d& a, const Vector3d& b) -> bool {
    const double L = (b - a).norm();
    const int N = std::max(1, (int)std::ceil(L / los_step));
    for (int i = 0; i <= N; ++i) {
      const double tt = (double)i / (double)N;
      const Vector3d p = a + tt * (b - a);
      if (!insideMap(p)) return false;
      const Vector3i id = coord2gridIndex(p);
      if (nearOccupied(id, r_safe)) return false;
    }
    return true;
  };

  startPtr->g_score = 0.0;
  startPtr->f_score = startPtr->g_score + kWAstarW * getHeu(startPtr, endPtr);
  startPtr->id = 1;
  startPtr->Father = NULL;

  Openset.insert(make_pair(startPtr->f_score, startPtr));

  vector<MappingNodePtr> neighborPtrSets;
  vector<double> edgeCostSets;

  MappingNodePtr bestPtr = startPtr;
  double best_h = getHeu(startPtr, endPtr);

  while (!Openset.empty()) {
    if ((ros::Time::now() - t0).toSec() > max_search_time) {
      terminatePtr = bestPtr;
      g_reached_goal = (bestPtr && bestPtr->index == goalIdx);
      ROS_WARN("[A*] time budget hit (%.3fs). Return best-so-far (h=%.2f).",
               max_search_time, best_h);
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

    // analytic connect near goal ONLY if the whole segment is "safe with radius"
    const double dist_goal = (endPtr->coord - currentPtr->coord).norm();
    if (dist_goal <= analytic_connect_dist &&
        segmentSafe(currentPtr->coord, endPtr->coord)) {
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
      if (isOccupied(neighborPtr->index)) continue;

      double tentative_g_score = currentPtr->g_score + edgeCostSets[i];

      if (neighborPtr->id == 0) {
        neighborPtr->Father  = currentPtr;
        neighborPtr->g_score = tentative_g_score;
        neighborPtr->f_score = tentative_g_score + kWAstarW * getHeu(neighborPtr, endPtr);
        neighborPtr->id = 1;
        neighborPtr->coord = gridIndex2coord(neighborPtr->index);
        Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
      } else if (neighborPtr->id == 1) {
        if (tentative_g_score + 1e-9 < neighborPtr->g_score) {
          neighborPtr->Father  = currentPtr;
          neighborPtr->g_score = tentative_g_score;
          neighborPtr->f_score = tentative_g_score + kWAstarW * getHeu(neighborPtr, endPtr);
          neighborPtr->coord = gridIndex2coord(neighborPtr->index);
          Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
        }
      }
    }
  }

  terminatePtr = bestPtr;
  g_reached_goal = (bestPtr && bestPtr->index == goalIdx);
  ROS_WARN("[A*] openset exhausted. Return best-so-far (h=%.2f).", best_h);
  return true;
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

  if (g_have_cont && !path.empty()) {
    path.front() = g_start_cont;
    if (g_reached_goal) path.back() = g_goal_cont;
  }
  return path;
}

// ============================================================
// Path post-processing:
//   - de-dup
//   - greedy prune using segmentSafe (with radius buffer)
//   - fillet
//   - adaptive split by forward probe AND local clearance
//   - safety repair: if a segment can't be made safe by bisection,
//                    try lateral offset midpoints to detour around cylinders
// ============================================================

std::vector<Vector3d> Astarpath::pathSimplify(const vector<Vector3d> &path,
                                              double path_resolution) {
  if (path.size() < 2) return path;

  static bool inited = false;
  static double los_clearance_m = 0.15;
  if (!inited) {
    ros::param::param("~astar_los_clearance_m", los_clearance_m, 0.15);
    inited = true;
  }
  const int r_safe = std::max(0, (int)std::ceil(los_clearance_m / resolution));
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

  auto nearOccupied = [&](const Vector3i& idx, int r) -> bool {
    if (isOccupied(idx)) return true;
    if (r <= 0) return false;
    for (int dx = -r; dx <= r; ++dx)
      for (int dy = -r; dy <= r; ++dy)
        for (int dz = -r; dz <= r; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          Vector3i q = idx + Vector3i(dx, dy, dz);
          if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
              q(1) < 0 || q(1) >= GRID_Y_SIZE ||
              q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;
          if (isOccupied(q)) return true;
        }
    return false;
  };

  auto segmentSafe = [&](const Vector3d& a, const Vector3d& b) -> bool {
    const double L = (b - a).norm();
    const int N = std::max(1, (int)std::ceil(L / los_step));
    for (int i = 0; i <= N; ++i) {
      const double t = (double)i / (double)N;
      const Vector3d p = a + t * (b - a);
      if (!insideMap(p)) return false;
      const Vector3i id = coord2gridIndex(p);
      if (nearOccupied(id, r_safe)) return false;
    }
    return true;
  };

  auto approxClearanceCells = [&](const Vector3i& idx, int rmax) -> int {
    if (isOccupied(idx)) return 0;
    for (int r = 1; r <= rmax; ++r) {
      for (int dx = -r; dx <= r; ++dx)
        for (int dy = -r; dy <= r; ++dy)
          for (int dz = -r; dz <= r; ++dz) {
            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r) continue;
            Vector3i q = idx + Vector3i(dx, dy, dz);
            if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
                q(1) < 0 || q(1) >= GRID_Y_SIZE ||
                q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;
            if (isOccupied(q)) return r;
          }
    }
    return rmax + 1;
  };

  auto improveClearance = [&](Vector3d p) -> Vector3d {
    clampIntoMap(p);
    Vector3i idx = coord2gridIndex(p);
    if (!nearOccupied(idx, r_safe)) return p;

    // search small neighborhood for a safer point
    Vector3i best = idx;
    int best_c = 0;
    double best_d = 1e9;

    const int scan = std::max(2, r_safe + 1);
    for (int dx = -scan; dx <= scan; ++dx)
      for (int dy = -scan; dy <= scan; ++dy)
        for (int dz = -scan; dz <= scan; ++dz) {
          Vector3i q = idx + Vector3i(dx, dy, dz);
          if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
              q(1) < 0 || q(1) >= GRID_Y_SIZE ||
              q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;
          if (isOccupied(q)) continue;
          if (nearOccupied(q, r_safe)) continue;

          int c = approxClearanceCells(q, std::max(3, r_safe + 2));
          double d = Vector3d(dx,dy,dz).norm();
          if (c > best_c || (c == best_c && d < best_d)) {
            best_c = c; best_d = d; best = q;
          }
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

  // anchor endpoints and improve clearance (prevents end-segment squeeze)
  if (g_have_cont) {
    clean.front() = improveClearance(g_start_cont);
    if (g_reached_goal) clean.back() = improveClearance(g_goal_cont);
    else clean.back() = improveClearance(clean.back());
  } else {
    clean.front() = improveClearance(clean.front());
    clean.back()  = improveClearance(clean.back());
  }

  // 2) greedy prune using segmentSafe (NOT plain LoS)
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

  // 3) corner fillet (light, but keep safety)
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

    if (ang > (25.0 * std::acos(-1.0) / 180.0) &&
        (Lab > min_corner) && (Lbc > min_corner)) {
      const double d1 = std::max(min_corner * 0.5, fillet_ratio * Lab);
      const double d2 = std::max(min_corner * 0.5, fillet_ratio * Lbc);
      Vector3d b1 = improveClearance(b - u * std::min(d1, 0.45 * Lab));
      Vector3d b2 = improveClearance(b + v * std::min(d2, 0.45 * Lbc));

      if (segmentSafe(fillet.back(), b1) &&
          segmentSafe(b1, b2) &&
          segmentSafe(b2, c)) {
        fillet.push_back(b1);
        fillet.push_back(b2);
        continue;
      }
    }
    fillet.push_back(improveClearance(b));
  }
  fillet.push_back(pruned.back());

  // forward-cone obstacle probing
  auto forwardObstacleDist = [&](const Vector3d& p0, const Vector3d& dir_unit) -> double {
    Vector3d d = dir_unit;
    if (d.norm() < 1e-6) return 0.0;

    Vector3d up(0, 0, 1);
    if (std::fabs(d.dot(up)) > 0.95) up = Vector3d(0, 1, 0);
    Vector3d right = d.cross(up);
    if (right.norm() < 1e-6) right = Vector3d(1, 0, 0);
    right.normalize();
    Vector3d up2 = right.cross(d).normalized();

    const double th = kForwardConeHalfAngleDeg * std::acos(-1.0) / 180.0;
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
        if (nearOccupied(coord2gridIndex(q), r_safe)) { best = std::min(best, s); break; }
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
  out.reserve(fillet.size() * 3);
  out.push_back(fillet.front());

  // try to find a detour midpoint when segment cannot satisfy segmentSafe
  auto findDetourMid = [&](const Vector3d& a, const Vector3d& b) -> bool {
    Vector3d d = b - a;
    const double L = d.norm();
    if (L < 1e-6) return false;
    d /= L;

    Vector3d up(0,0,1);
    if (std::fabs(d.dot(up)) > 0.95) up = Vector3d(0,1,0);
    Vector3d right = d.cross(up).normalized();
    Vector3d up2 = right.cross(d).normalized();

    const Vector3d mid0 = 0.5 * (a + b);

    // candidate offsets (meters)
    const double radii[] = {0.6, 1.0, 1.4};
    const double angles_deg[] = {0, 45, 90, 135, 180, 225, 270, 315};

    for (double r : radii) {
      for (double angd : angles_deg) {
        double ang = angd * std::acos(-1.0) / 180.0;
        Vector3d off = std::cos(ang) * right + std::sin(ang) * up2;
        Vector3d mid = improveClearance(mid0 + r * off);
        if (segmentSafe(a, mid) && segmentSafe(mid, b)) {
          if ((mid - out.back()).norm() > 1e-4) out.push_back(mid);
          return true;
        }
      }
      // also try vertical nudges
      Vector3d midu = improveClearance(mid0 + r * up2);
      if (segmentSafe(a, midu) && segmentSafe(midu, b)) {
        if ((midu - out.back()).norm() > 1e-4) out.push_back(midu);
        return true;
      }
      Vector3d midd = improveClearance(mid0 - r * up2);
      if (segmentSafe(a, midd) && segmentSafe(midd, b)) {
        if ((midd - out.back()).norm() > 1e-4) out.push_back(midd);
        return true;
      }
    }
    return false;
  };

  // push target with safety repair
  auto pushChecked = [&](const Vector3d& target) {
    Vector3d p = improveClearance(target);

    struct Seg { Vector3d a,b; int depth; };
    std::vector<Seg> st;
    st.push_back({out.back(), p, 0});

    while (!st.empty()) {
      Seg seg = st.back(); st.pop_back();

      if (segmentSafe(seg.a, seg.b)) {
        if ((seg.b - out.back()).norm() > 1e-4) out.push_back(seg.b);
        continue;
      }

      // if bisection depth is high, try detour midpoints (critical for cylinder obstacles)
      if (seg.depth >= 6) {
        if (findDetourMid(seg.a, seg.b)) {
          // after inserting mid, we will continue with seg.b next via recursion-like split
          st.push_back({out.back(), seg.b, seg.depth + 1});
          continue;
        }
      }

      if (seg.depth >= 10) {
        // final fallback: accept split midpoints (will at least avoid huge jumps)
        Vector3d mid = improveClearance(0.5 * (seg.a + seg.b));
        if ((mid - out.back()).norm() > 1e-4) out.push_back(mid);
        if ((seg.b - out.back()).norm() > 1e-4) out.push_back(seg.b);
        continue;
      }

      Vector3d mid = improveClearance(0.5 * (seg.a + seg.b));
      st.push_back({mid, seg.b, seg.depth + 1});
      st.push_back({seg.a, mid, seg.depth + 1});
    }
  };

  for (size_t s = 0; s + 1 < fillet.size(); ++s) {
    const Vector3d a0 = out.back();
    const Vector3d b0 = fillet[s + 1];
    Vector3d d = b0 - a0;
    const double L = d.norm();
    if (L < 1e-6) continue;
    d /= L;

    const double obs = forwardObstacleDist(a0, d);

    // also adapt by local clearance at a0 (side obstacles)
    const int ccell = approxClearanceCells(coord2gridIndex(a0), std::max(4, r_safe + 2));
    const double clearance_m = std::max(0.0, (double)ccell * resolution);

    double max_len = base_mid;
    if (obs <= 1.0) max_len = base_near;
    else if (obs <= 3.0) max_len = base_mid;
    else max_len = base_open;

    if (clearance_m < 0.8) max_len = std::min(max_len, base_near);
    else if (clearance_m < 1.2) max_len = std::min(max_len, base_mid);

    max_len = std::max(max_len, kMinSplitLenM);

    int N = std::max(1, (int)std::ceil(L / max_len));
    while (N > 1 && (L / (double)N) < kMinSplitLenM) --N;

    for (int k = 1; k <= N; ++k) {
      const double t = (double)k / (double)N;
      Vector3d p = a0 + t * (b0 - a0);
      pushChecked(p);
    }
  }

  // final de-dup
  std::vector<Vector3d> final_out;
  final_out.reserve(out.size());
  final_out.push_back(out.front());
  for (size_t ii = 1; ii < out.size(); ++ii) {
    if ((out[ii] - final_out.back()).norm() > 1e-4) final_out.push_back(out[ii]);
  }

  ROS_WARN("[pathSimplify] in=%d clean=%d pruned=%d fillet=%d out=%d (r_safe=%d)",
           (int)path.size(), (int)clean.size(), (int)pruned.size(), (int)fillet.size(),
           (int)final_out.size(), r_safe);

  return final_out;
}

// ============================================================
// helpers used by your downstream modules (unchanged)
// ============================================================

double Astarpath::perpendicularDistance(const Eigen::Vector3d point_insert,
                                        const Eigen::Vector3d point_st,
                                        const Eigen::Vector3d point_end) {
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

  // slightly finer sampling makes collision detection consistent
  const double delta_t = std::max(std::min(resolution * 0.5, 0.10), 0.03);

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
