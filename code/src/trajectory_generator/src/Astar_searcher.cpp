#include "Astar_searcher.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include <ros/ros.h>

using namespace std;
using namespace Eigen;

// -------------------------------
// Tunables (keep minimal, but effective)
// -------------------------------
static constexpr double kWAstarW      = 1.05;       // Weighted A*: f = g + w*h
static constexpr double kTieBreaker   = 1.0 + 1e-4; // small >1 tie-breaker
static constexpr double kZMovePenalty = 1.20;       // discourage unnecessary vertical moves
static constexpr double kMinGoalZUp   = 1e-3;       // treat near-zero as "invalid ground goal"

// obstacle inflation (configuration-space safety) in meters
static constexpr double kInflationBaseM  = 0.30;    // base safety buffer
static constexpr double kInflationZBaseM = 0.25;    // mild Z inflation

// LoS sampling step (meters)
static inline double segStep(double res) {
  return std::max(0.05, 0.5 * res);
}

// -------------------------------
// NEW: forward-cone sparsification knobs
// -------------------------------
static constexpr double kForwardConeHalfAngleDeg = 30.0; // only forward ~30deg affects density
static constexpr double kForwardRayCastAngleDeg  = 20.0; // sample rays within cone (approx)
static constexpr double kForwardLookaheadM       = 6.0;  // how far to look ahead for obstacles
static constexpr double kMinSplitLenM            = 1.2;  // never split below this (avoid micro segments)

// NEW: if start/goal is in inflated occupied cell, snap to nearest free within this radius (cells)
static constexpr int    kNearestFreeMaxRCells    = 10;

// -------------------------------
// NEW: store continuous (non-grid-rounded) start/goal for endpoint anchoring
// (Fixes: "RMSE small but not reaching RViz goal", since rounding to grid center causes mismatch)
// -------------------------------
namespace {
static Eigen::Vector3d g_start_cont(0, 0, 0);
static Eigen::Vector3d g_goal_cont(0, 0, 0);
static bool            g_have_cont = false;
} // namespace

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
  GLYZ_SIZE = GRID_Y_SIZE * GRID_Z_SIZE;
  GLXYZ_SIZE = GRID_X_SIZE * GLYZ_SIZE;

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

  // read extra margin once (if param exists); otherwise 0
  static bool   inited = false;
  static double margin_m = 0.0;
  if (!inited) {
    ros::param::param<double>("/trajectory_generator_node/map/margin", margin_m, 0.0);
    inited = true;
  }

  const double infl_xy_m = std::max(0.0, kInflationBaseM + margin_m);
  const double infl_z_m  = std::max(0.0, kInflationZBaseM + 0.5 * margin_m);

  const int infl_xy = std::max(1, (int)std::ceil(infl_xy_m / resolution));
  const int infl_z  = std::max(0, (int)std::ceil(infl_z_m  / resolution));

  for (int dx = -infl_xy; dx <= infl_xy; ++dx) {
    for (int dy = -infl_xy; dy <= infl_xy; ++dy) {
      const int nx = idx_x + dx;
      const int ny = idx_y + dy;
      if (nx < 0 || nx >= GRID_X_SIZE || ny < 0 || ny >= GRID_Y_SIZE) continue;

      // disk-ish inflation in XY
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

// 26-neighborhood successor generation + NO corner cutting
inline void Astarpath::AstarGetSucc(MappingNodePtr currentPtr,
                                    vector<MappingNodePtr> &neighborPtrSets,
                                    vector<double> &edgeCostSets) {
  neighborPtrSets.clear();
  edgeCostSets.clear();

  const int cx = currentPtr->index(0);
  const int cy = currentPtr->index(1);
  const int cz = currentPtr->index(2);

  Vector3i nb;
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        if (dx == 0 && dy == 0 && dz == 0) continue;

        int nx = cx + dx;
        int ny = cy + dy;
        int nz = cz + dz;

        if (nx < 0 || nx >= GRID_X_SIZE ||
            ny < 0 || ny >= GRID_Y_SIZE ||
            nz < 0 || nz >= GRID_Z_SIZE)
          continue;

        // no corner cutting
        if (dx != 0 && isOccupied(cx + dx, cy, cz)) continue;
        if (dy != 0 && isOccupied(cx, cy + dy, cz)) continue;
        if (dz != 0 && isOccupied(cx, cy, cz + dz)) continue;

        if (dx != 0 && dy != 0) {
          if (isOccupied(cx + dx, cy, cz) || isOccupied(cx, cy + dy, cz)) continue;
        }
        if (dx != 0 && dz != 0) {
          if (isOccupied(cx + dx, cy, cz) || isOccupied(cx, cy, cz + dz)) continue;
        }
        if (dy != 0 && dz != 0) {
          if (isOccupied(cx, cy + dy, cz) || isOccupied(cx, cy, cz + dz)) continue;
        }

        nb << nx, ny, nz;
        neighborPtrSets.push_back(Map_Node[nx][ny][nz]);

        double base = std::sqrt(double(dx * dx + dy * dy + dz * dz));
        if (dz != 0) base *= kZMovePenalty;
        edgeCostSets.push_back(base);
      }
    }
  }
}

// 3D diagonal distance heuristic + tie-breaker
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

bool Astarpath::AstarSearch(Vector3d start_pt, Vector3d end_pt) {
  ros::Time time_1 = ros::Time::now();

  resetUsedGrids();
  Openset.clear();
  terminatePtr = NULL;

  // RViz may send z=0; keep flight altitude
  if (end_pt(2) <= gl_zl + kMinGoalZUp) {
    end_pt(2) = start_pt(2);
  }

  // Clamp continuous endpoints into map (do NOT round to grid here)
  auto clampd = [&](double v, double lo, double hi) -> double {
    return std::max(lo, std::min(hi, v));
  };
  auto clampIntoMap = [&](Vector3d p) -> Vector3d {
    p(0) = clampd(p(0), gl_xl + 1e-3, gl_xu - 1e-3);
    p(1) = clampd(p(1), gl_yl + 1e-3, gl_yu - 1e-3);
    p(2) = clampd(p(2), gl_zl + 1e-3, gl_zu - 1e-3);
    return p;
  };

  start_pt = clampIntoMap(start_pt);
  end_pt   = clampIntoMap(end_pt);

  // Store continuous endpoints for later anchoring (Fix: "not reaching RViz goal")
  g_start_cont = start_pt;
  g_goal_cont  = end_pt;
  g_have_cont  = true;

  Vector3i start_idx = coord2gridIndex(start_pt);
  Vector3i end_idx   = coord2gridIndex(end_pt);
  goalIdx = end_idx;

  // If start/goal cell becomes occupied after inflation, snap to nearest free
  auto findNearestFree = [&](const Vector3i& center, int max_r, Vector3i& out_idx) -> bool {
    if (!isOccupied(center)) { out_idx = center; return true; }
    double best_d2 = 1e18;
    bool found = false;

    for (int r = 1; r <= max_r; ++r) {
      for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
          for (int dz = -r; dz <= r; ++dz) {
            // only check the "shell" for speed
            if (std::abs(dx) != r && std::abs(dy) != r && std::abs(dz) != r) continue;

            Vector3i cand = center + Vector3i(dx, dy, dz);
            if (cand(0) < 0 || cand(0) >= GRID_X_SIZE ||
                cand(1) < 0 || cand(1) >= GRID_Y_SIZE ||
                cand(2) < 0 || cand(2) >= GRID_Z_SIZE) continue;

            if (!isOccupied(cand)) {
              const double d2 = double(dx*dx + dy*dy + dz*dz);
              if (d2 < best_d2) {
                best_d2 = d2;
                out_idx = cand;
                found = true;
              }
            }
          }
        }
      }
      if (found) return true;
    }
    return false;
  };

  if (isOccupied(start_idx)) {
    Vector3i s2;
    if (findNearestFree(start_idx, kNearestFreeMaxRCells, s2)) {
      ROS_WARN("[A*] start in inflated obstacle, snap to nearest free cell.");
      start_idx = s2;
      g_start_cont = gridIndex2coord(start_idx); // keep consistent with executable endpoint
    } else {
      ROS_WARN("[A*] start in obstacle cell, abort.");
      return false;
    }
  }
  if (isOccupied(end_idx)) {
    Vector3i g2;
    if (findNearestFree(end_idx, kNearestFreeMaxRCells, g2)) {
      ROS_WARN("[A*] goal in inflated obstacle, snap to nearest free cell.");
      end_idx = g2;
      goalIdx = end_idx;
      g_goal_cont = gridIndex2coord(end_idx); // target becomes reachable goal
    } else {
      ROS_WARN("[A*] goal in obstacle cell, abort.");
      return false;
    }
  }

  // Planning nodes use grid centers (stable)
  Vector3d start_grid = gridIndex2coord(start_idx);
  Vector3d goal_grid  = gridIndex2coord(end_idx);

  MappingNodePtr startPtr = Map_Node[start_idx(0)][start_idx(1)][start_idx(2)];
  MappingNodePtr endPtr   = Map_Node[end_idx(0)][end_idx(1)][end_idx(2)];

  startPtr->coord = start_grid;
  endPtr->coord   = goal_grid;

  startPtr->g_score = 0.0;
  startPtr->f_score = startPtr->g_score + kWAstarW * getHeu(startPtr, endPtr);
  startPtr->id = 1;
  startPtr->Father = NULL;

  Openset.insert(make_pair(startPtr->f_score, startPtr));

  // NEW: local LoS check for "analytic expansion" near the goal (Lazy-Theta* style)
  const double los_step = segStep(resolution);
  auto insideMap = [&](const Vector3d& p) -> bool {
    return (p(0) >= gl_xl && p(0) < gl_xu &&
            p(1) >= gl_yl && p(1) < gl_yu &&
            p(2) >= gl_zl && p(2) < gl_zu);
  };
  auto segmentFreeInflated = [&](const Vector3d& a, const Vector3d& b) -> bool {
    const double L = (b - a).norm();
    const int N = std::max(1, (int)std::ceil(L / los_step));
    for (int i = 0; i <= N; ++i) {
      const double t = (double)i / (double)N;
      const Vector3d p = a + t * (b - a);
      if (!insideMap(p)) return false;
      if (isOccupied(coord2gridIndex(p))) return false;
    }
    return true;
  };

  vector<MappingNodePtr> neighborPtrSets;
  vector<double> edgeCostSets;

  while (!Openset.empty()) {
    auto it = Openset.begin();
    MappingNodePtr currentPtr = it->second;
    Openset.erase(it);

    if (currentPtr->id == -1) continue;

    if (currentPtr->index == goalIdx) {
      terminatePtr = currentPtr;

      ros::Time time_2 = ros::Time::now();
      if ((time_2 - time_1).toSec() > 0.1)
        ROS_WARN("Time consume in Astar path finding is %f",
                 (time_2 - time_1).toSec());
      return true;
    }

    // NEW: if close enough to goal and there is direct LoS, connect and finish early
    // (reduces expansions and yields straighter terminal segment)
    const double dist_to_goal = (gridIndex2coord(currentPtr->index) - goal_grid).norm();
    if (dist_to_goal < 6.0) {
      if (segmentFreeInflated(gridIndex2coord(currentPtr->index), goal_grid)) {
        endPtr->Father = currentPtr;
        terminatePtr = endPtr;

        ros::Time time_2 = ros::Time::now();
        if ((time_2 - time_1).toSec() > 0.1)
          ROS_WARN("Time consume in Astar path finding is %f",
                   (time_2 - time_1).toSec());
        return true;
      }
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
        if (tentative_g_score < neighborPtr->g_score) {
          neighborPtr->Father  = currentPtr;
          neighborPtr->g_score = tentative_g_score;
          neighborPtr->f_score = tentative_g_score + kWAstarW * getHeu(neighborPtr, endPtr);
          neighborPtr->coord = gridIndex2coord(neighborPtr->index);
          Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
        }
      }
    }
  }

  ros::Time time_2 = ros::Time::now();
  if ((time_2 - time_1).toSec() > 0.1)
    ROS_WARN("Time consume in Astar path finding is %f",
             (time_2 - time_1).toSec());
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

  // NEW: anchor endpoints to continuous start/goal (avoid grid-center mismatch)
  if (g_have_cont && !path.empty()) {
    path.front() = g_start_cont;
    path.back()  = g_goal_cont;
  }

  return path;
}

// ------------------------------------------------------------
// Path post-processing:
// 1) LoS prune (aggressive; only collision-free, not clearance-based)
// 2) Light fillet at sharp corners (only if safe)
// 3) NEW: forward-cone-aware sparsification (open => very sparse; only forward 30deg + close obstacles densify)
// This directly addresses: "too many intermediate nodes => vel=0 at each => slow in open area"
// ------------------------------------------------------------
std::vector<Vector3d> Astarpath::pathSimplify(const vector<Vector3d> &path,
                                              double path_resolution) {
  if (path.empty()) return {};
  if (path.size() == 1) return path;

  auto clampd = [&](double v, double lo, double hi) -> double {
    return std::max(lo, std::min(hi, v));
  };
  auto clampIntoMap = [&](Vector3d p) -> Vector3d {
    p(0) = clampd(p(0), gl_xl + 1e-3, gl_xu - 1e-3);
    p(1) = clampd(p(1), gl_yl + 1e-3, gl_yu - 1e-3);
    p(2) = clampd(p(2), gl_zl + 1e-3, gl_zu - 1e-3);
    return p;
  };
  auto insideMap = [&](const Vector3d& p) -> bool {
    return (p(0) >= gl_xl && p(0) < gl_xu &&
            p(1) >= gl_yl && p(1) < gl_yu &&
            p(2) >= gl_zl && p(2) < gl_zu);
  };

  // 0) de-duplicate consecutive points
  std::vector<Vector3d> clean;
  clean.reserve(path.size());
  clean.push_back(path.front());
  for (size_t i = 1; i < path.size(); ++i) {
    if ((path[i] - clean.back()).norm() > 1e-4) clean.push_back(path[i]);
  }
  if (clean.size() <= 2) {
    if (g_have_cont && clean.size() >= 1) clean.front() = clampIntoMap(g_start_cont);
    if (g_have_cont && clean.size() >= 2) clean.back()  = clampIntoMap(g_goal_cont);
    return clean;
  }

  // NEW: anchor endpoints before pruning
  if (g_have_cont) {
    clean.front() = clampIntoMap(g_start_cont);
    clean.back()  = clampIntoMap(g_goal_cont);
  } else {
    clean.front() = clampIntoMap(clean.front());
    clean.back()  = clampIntoMap(clean.back());
  }

  const double los_step = segStep(resolution);

  auto segmentFreeInflated = [&](const Vector3d& a, const Vector3d& b) -> bool {
    const double L = (b - a).norm();
    const int N = std::max(1, (int)std::ceil(L / los_step));
    for (int i = 0; i <= N; ++i) {
      const double t = (double)i / (double)N;
      const Vector3d p = a + t * (b - a);
      if (!insideMap(p)) return false;
      if (isOccupied(coord2gridIndex(p))) return false;
    }
    return true;
  };

  // 1) LoS greedy pruning (Theta*/Lazy-Theta* style shortcutting)
  std::vector<Vector3d> pruned;
  pruned.reserve(clean.size());
  size_t i = 0;
  pruned.push_back(clean[0]);
  while (i < clean.size() - 1) {
    size_t j = clean.size() - 1;
    for (; j > i + 1; --j) {
      if (segmentFreeInflated(clean[i], clean[j])) break;
    }
    pruned.push_back(clean[j]);
    i = j;
  }

  // 2) Light fillet around sharp turns (avoid unnecessary stop-go at corners)
  std::vector<Vector3d> fillet;
  fillet.reserve(pruned.size() * 2);
  fillet.push_back(pruned.front());

  const double fillet_max = std::max(0.80, 4.0 * resolution);
  const double angle_thr  = 0.40; // rad (~23deg)

  for (size_t k = 1; k + 1 < pruned.size(); ++k) {
    const Vector3d A = pruned[k - 1];
    const Vector3d B = pruned[k];
    const Vector3d C = pruned[k + 1];

    const Vector3d v1 = (B - A);
    const Vector3d v2 = (C - B);
    const double L1 = v1.norm();
    const double L2 = v2.norm();

    if (L1 < 1e-6 || L2 < 1e-6) {
      fillet.push_back(B);
      continue;
    }

    const Vector3d u1 = v1 / L1;
    const Vector3d u2 = v2 / L2;
    const double cosang = clampd(u1.dot(u2), -1.0, 1.0);
    const double ang = std::acos(cosang);

    if (ang < angle_thr) {
      fillet.push_back(B);
      continue;
    }

    const double d = std::min(fillet_max, 0.25 * std::min(L1, L2));
    if (d < 2.0 * resolution) {
      fillet.push_back(B);
      continue;
    }

    const Vector3d B1 = B - u1 * d;
    const Vector3d B2 = B + u2 * d;

    if (segmentFreeInflated(A, B1) && segmentFreeInflated(B1, B2) && segmentFreeInflated(B2, C)) {
      fillet.push_back(B1);
      fillet.push_back(B2);
    } else {
      fillet.push_back(B);
    }
  }
  fillet.push_back(pruned.back());

  // 3) NEW: forward-cone-aware sparsification (this is the 핵심 for speed)
  //    Only obstacles ahead (≈30°) and close enough cause densification; side obstacles do NOT.
  const double cone_half_rad = kForwardConeHalfAngleDeg * M_PI / 180.0;
  const double ray_ang_rad   = kForwardRayCastAngleDeg  * M_PI / 180.0;

  auto rayHitDistRaw = [&](const Vector3d& p0, const Vector3d& dir_in, double max_d) -> double {
    Vector3d dir = dir_in;
    const double n = dir.norm();
    if (n < 1e-6) return max_d;
    dir /= n;

    const double step = std::max(0.05, 0.5 * resolution);
    const int N = std::max(1, (int)std::ceil(max_d / step));

    for (int i = 1; i <= N; ++i) {
      const double s = i * step;
      Vector3d p = p0 + dir * s;
      if (!insideMap(p)) return max_d;
      if (is_occupy_raw(coord2gridIndex(p))) {
        return s;
      }
    }
    return max_d;
  };

  auto forwardClearanceApprox = [&](const Vector3d& p, const Vector3d& dir) -> double {
    Vector3d d = dir;
    const double n = d.norm();
    if (n < 1e-6) return kForwardLookaheadM;
    d /= n;

    // build a stable local frame (right, up2)
    Vector3d up(0, 0, 1);
    if (std::abs(d.dot(up)) > 0.95) up = Vector3d(0, 1, 0);
    Vector3d right = d.cross(up).normalized();
    Vector3d up2   = right.cross(d).normalized();

    // rays inside forward cone (center + yaw± + pitch±)
    double best = kForwardLookaheadM;
    best = std::min(best, rayHitDistRaw(p, d, kForwardLookaheadM));
    best = std::min(best, rayHitDistRaw(p, (d * std::cos(ray_ang_rad) + right * std::sin(ray_ang_rad)).normalized(), kForwardLookaheadM));
    best = std::min(best, rayHitDistRaw(p, (d * std::cos(ray_ang_rad) - right * std::sin(ray_ang_rad)).normalized(), kForwardLookaheadM));
    best = std::min(best, rayHitDistRaw(p, (d * std::cos(ray_ang_rad) + up2   * std::sin(ray_ang_rad)).normalized(), kForwardLookaheadM));
    best = std::min(best, rayHitDistRaw(p, (d * std::cos(ray_ang_rad) - up2   * std::sin(ray_ang_rad)).normalized(), kForwardLookaheadM));
    return best;
  };

  auto chooseMaxLenByForward = [&](double fwd_clear_m) -> double {
    // Larger clearance => fewer waypoints => faster (because Vel/Acc are forced to 0 at waypoints in your traj generator)
    // The thresholds are tuned for your "open vs near obstacle" behavior.
    const double base_scale = std::max(1.0, path_resolution);

    if (fwd_clear_m >= 4.5) return 12.0 * base_scale;
    if (fwd_clear_m >= 3.0) return 8.0  * base_scale;
    if (fwd_clear_m >= 2.0) return 5.0  * base_scale;
    if (fwd_clear_m >= 1.3) return 3.0  * base_scale;
    return 1.5 * base_scale;
  };

  // local nudge for INTERNAL points only (endpoints must remain anchored)
  auto nudgeToFreeInflated = [&](const Vector3d& p_in) -> Vector3d {
    Vector3d p = clampIntoMap(p_in);
    Vector3i idx = coord2gridIndex(p);
    if (!isOccupied(idx)) return p;

    // very small local search (keep it cheap and stable)
    const int r = 2;
    double best_d2 = 1e18;
    bool found = false;
    Vector3i best = idx;

    for (int dx = -r; dx <= r; ++dx)
      for (int dy = -r; dy <= r; ++dy)
        for (int dz = -r; dz <= r; ++dz) {
          Vector3i c = idx + Vector3i(dx, dy, dz);
          if (c(0) < 0 || c(0) >= GRID_X_SIZE ||
              c(1) < 0 || c(1) >= GRID_Y_SIZE ||
              c(2) < 0 || c(2) >= GRID_Z_SIZE) continue;
          if (!isOccupied(c)) {
            const double d2 = double(dx*dx + dy*dy + dz*dz);
            if (d2 < best_d2) {
              best_d2 = d2;
              best = c;
              found = true;
            }
          }
        }

    if (found) return gridIndex2coord(best);
    return p; // fallback (should be rare)
  };

  std::vector<Vector3d> out;
  out.reserve(fillet.size() * 2);
  out.push_back(fillet.front());

  int total_splits = 0;

  for (size_t s = 0; s + 1 < fillet.size(); ++s) {
    const Vector3d a = fillet[s];
    const Vector3d b = fillet[s + 1];
    const Vector3d dir = (b - a);
    const double L = dir.norm();

    if (L < 1e-6) continue;

    // Forward clearance sampled at a + mid (approximates the 30deg forward influence)
    const Vector3d mid = a + 0.5 * (b - a);
    const double fwd_a   = forwardClearanceApprox(a, dir);
    const double fwd_mid = forwardClearanceApprox(mid, dir);
    const double fwd_clear = std::min(fwd_a, fwd_mid);

    double max_len = chooseMaxLenByForward(fwd_clear);
    max_len = std::max(max_len, kMinSplitLenM);

    // If the segment is collision-free, we still may split only when forward clearance is small.
    // This is intentionally a "speed control by waypoint count" mechanism (matches your traj generator behavior).
    int N = std::max(1, (int)std::ceil(L / max_len));

    // Safety: avoid pathological over-splitting
    N = std::min(N, 10);

    total_splits += (N - 1);

    for (int k = 1; k <= N; ++k) {
      const double t = (double)k / (double)N;
      Vector3d p = a + t * (b - a);

      const bool is_last_point = (s + 1 == fillet.size() - 1) && (k == N);
      const bool is_first_point = (s == 0) && (k == 1);

      // Keep endpoints anchored exactly (fix "end not reached")
      if (is_first_point) p = fillet[1]; // keep continuity; start already in out[0]
      if (is_last_point)  p = fillet.back();

      // Internal points: nudge if discretization makes them fall into occupied inflated voxel
      if (!is_last_point) {
        p = nudgeToFreeInflated(p);
      }

      if ((p - out.back()).norm() > 1e-4) out.push_back(p);
    }
  }

  // Final anchor (hard guarantee)
  if (g_have_cont && !out.empty()) {
    out.front() = clampIntoMap(g_start_cont);
    out.back()  = clampIntoMap(g_goal_cont);
  }

  ROS_WARN("[pathSimplify] in=%d clean=%d pruned=%d fillet=%d out=%d total_splits=%d",
           (int)path.size(), (int)clean.size(), (int)pruned.size(), (int)fillet.size(), (int)out.size(),
           total_splits);

  return out;
}

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

  // finer sampling reduces "overshoot through obstacle" misses
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
