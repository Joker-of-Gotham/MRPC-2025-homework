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

// minimum clearance constraint for shortcut segments (meters)
// if segment samples have clearance < this => reject shortcut (prevents narrow gaps)
static constexpr double kMinClearanceM = 0.45;

// clearance probing (local, cheap) in cells
static constexpr int    kClearProbeRCells = 6;      // probe radius in cells
static constexpr double kClearOpenCapM    = 6.0;    // if no obstacle within probe => treat as open

// LoS sampling step (meters)
static inline double segStep(double res) {
  return std::max(0.05, 0.5 * res);
}

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

      // disk-ish inflation in XY (optional but helps)
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

// 3D diagonal distance heuristic (admissible baseline) + tie-breaker
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

  Vector3i start_idx = coord2gridIndex(start_pt);
  Vector3i end_idx   = coord2gridIndex(end_pt);
  goalIdx = end_idx;

  start_pt = gridIndex2coord(start_idx);
  end_pt   = gridIndex2coord(end_idx);

  MappingNodePtr startPtr = Map_Node[start_idx(0)][start_idx(1)][start_idx(2)];
  MappingNodePtr endPtr   = Map_Node[end_idx(0)][end_idx(1)][end_idx(2)];

  startPtr->coord = start_pt;
  endPtr->coord   = end_pt;

  if (isOccupied(startPtr->index)) {
    ROS_WARN("[A*] start in obstacle cell, abort.");
    return false;
  }
  if (isOccupied(endPtr->index)) {
    ROS_WARN("[A*] goal in obstacle cell, abort.");
    return false;
  }

  startPtr->g_score = 0.0;
  startPtr->f_score = startPtr->g_score + kWAstarW * getHeu(startPtr, endPtr);
  startPtr->id = 1;
  startPtr->Father = NULL;

  Openset.insert(make_pair(startPtr->f_score, startPtr));

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

  return path;
}

// ------------------------------------------------------------
// Path post-processing: LoS prune + fillet + clearance-adaptive sparsify
// Goal: fewer waypoints in open space, keep safety in narrow gaps, smoother corners.
// ------------------------------------------------------------
std::vector<Vector3d> Astarpath::pathSimplify(const vector<Vector3d> &path,
                                              double path_resolution) {
  if (path.empty()) return {};
  if (path.size() == 1) return path;

  // 0) de-duplicate consecutive points
  std::vector<Vector3d> clean;
  clean.reserve(path.size());
  clean.push_back(path.front());
  for (size_t i = 1; i < path.size(); ++i) {
    if ((path[i] - clean.back()).norm() > 1e-4) clean.push_back(path[i]);
  }
  if (clean.size() <= 2) return clean;

  const double los_step = segStep(resolution);

  auto insideMap = [&](const Vector3d& p) -> bool {
    return (p(0) >= gl_xl && p(0) < gl_xu &&
            p(1) >= gl_yl && p(1) < gl_yu &&
            p(2) >= gl_zl && p(2) < gl_zu);
  };

  // approx clearance at point: nearest occupied within probe radius; if none => open cap
  auto approxClearance = [&](const Vector3d& p) -> double {
    if (!insideMap(p)) return 0.0;
    const Vector3i idx = coord2gridIndex(p);

    bool found = false;
    double best = 1e9;

    for (int dx = -kClearProbeRCells; dx <= kClearProbeRCells; ++dx) {
      for (int dy = -kClearProbeRCells; dy <= kClearProbeRCells; ++dy) {
        for (int dz = -kClearProbeRCells; dz <= kClearProbeRCells; ++dz) {
          Vector3i q = idx + Vector3i(dx, dy, dz);
          if (q(0) < 0 || q(0) >= GRID_X_SIZE ||
              q(1) < 0 || q(1) >= GRID_Y_SIZE ||
              q(2) < 0 || q(2) >= GRID_Z_SIZE) continue;
          if (isOccupied(q)) {
            found = true;
            const double d = std::sqrt(double(dx*dx + dy*dy + dz*dz)) * resolution;
            if (d < best) best = d;
          }
        }
      }
    }
    if (!found) return kClearOpenCapM;
    return best;
  };

  // segment validity: collision-free AND keep minimum clearance
  auto segmentValid = [&](const Vector3d& a, const Vector3d& b) -> bool {
    const double L = (b - a).norm();
    const int N = std::max(1, (int)std::ceil(L / los_step));
    for (int i = 0; i <= N; ++i) {
      const double t = (double)i / (double)N;
      const Vector3d p = a + t * (b - a);

      if (!insideMap(p)) return false;
      if (isOccupied(coord2gridIndex(p))) return false;

      const double clr = approxClearance(p);
      if (clr < kMinClearanceM) return false;
    }
    return true;
  };

  // 1) LoS greedy pruning (A* post-smoothing / Theta*-style shortcutting)
  std::vector<Vector3d> pruned;
  pruned.reserve(clean.size());
  size_t i = 0;
  pruned.push_back(clean[0]);
  while (i < clean.size() - 1) {
    size_t j = clean.size() - 1;
    for (; j > i + 1; --j) {
      if (segmentValid(clean[i], clean[j])) break;
    }
    pruned.push_back(clean[j]);
    i = j;
  }
  if (pruned.size() <= 2) {
    // still OK; will be sparsified below
  }

  // 2) Corner fillet (very light): insert two points around sharp corners if safe
  std::vector<Vector3d> fillet;
  fillet.reserve(pruned.size() * 2);
  fillet.push_back(pruned.front());

  auto clampd = [&](double v, double lo, double hi) -> double {
    return std::max(lo, std::min(hi, v));
  };

  const double fillet_max = std::max(0.60, 3.0 * resolution);
  const double angle_thr  = 0.35; // rad (~20deg) : only fillet meaningful turns

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

    // nearly straight -> keep
    if (ang < angle_thr) {
      fillet.push_back(B);
      continue;
    }

    const double d = std::min(fillet_max, 0.30 * std::min(L1, L2));
    if (d < 2.0 * resolution) {
      fillet.push_back(B);
      continue;
    }

    const Vector3d B1 = B - u1 * d;
    const Vector3d B2 = B + u2 * d;

    // accept fillet only if safe
    if (segmentValid(A, B1) && segmentValid(B1, B2) && segmentValid(B2, C)) {
      fillet.push_back(B1);
      fillet.push_back(B2);
    } else {
      fillet.push_back(B);
    }
  }
  fillet.push_back(pruned.back());

  // 3) Clearance-adaptive max-segment sparsification (open => long segments, near => short)
  auto segmentMinClearance = [&](const Vector3d& a, const Vector3d& b) -> double {
    const double L = (b - a).norm();
    const double step = std::max(0.40, 2.0 * resolution);
    const int N = std::max(1, (int)std::ceil(L / step));
    double minc = kClearOpenCapM;
    for (int i = 0; i <= N; ++i) {
      const double t = (double)i / (double)N;
      const Vector3d p = a + t * (b - a);
      minc = std::min(minc, approxClearance(p));
    }
    return minc;
  };

  // set three regimes by clearance
  const double max_len_open = std::max(4.0, 12.0 * path_resolution);
  const double max_len_mid  = std::max(2.0,  6.0 * path_resolution);
  const double max_len_near = std::max(1.0,  3.0 * path_resolution);

  std::vector<Vector3d> out;
  out.reserve(fillet.size() * 2);
  out.push_back(fillet.front());

  double maxSegObserved = 0.0;

  const int local_search_r = 1;

  for (size_t s = 0; s + 1 < fillet.size(); ++s) {
    const Vector3d a = fillet[s];
    const Vector3d b = fillet[s + 1];
    const double L = (b - a).norm();
    maxSegObserved = std::max(maxSegObserved, L);

    const double minc = segmentMinClearance(a, b);

    double max_len = max_len_mid;
    if (minc >= 2.0) max_len = max_len_open;
    else if (minc >= 1.0) max_len = max_len_mid;
    else max_len = max_len_near;

    // split only if needed
    const int N = std::max(1, (int)std::ceil(L / max_len));

    for (int k = 1; k <= N; ++k) {
      const double t = (double)k / (double)N;
      Vector3d p = a + t * (b - a);

      // local nudge if rounding hits occupied
      if (!insideMap(p)) {
        // fallback: clamp into map bounds
        p(0) = clampd(p(0), gl_xl + 1e-3, gl_xu - 1e-3);
        p(1) = clampd(p(1), gl_yl + 1e-3, gl_yu - 1e-3);
        p(2) = clampd(p(2), gl_zl + 1e-3, gl_zu - 1e-3);
      }

      Vector3i idx = coord2gridIndex(p);
      if (isOccupied(idx)) {
        bool found = false;
        for (int dx = -local_search_r; dx <= local_search_r && !found; ++dx)
          for (int dy = -local_search_r; dy <= local_search_r && !found; ++dy)
            for (int dz = -local_search_r; dz <= local_search_r && !found; ++dz) {
              Vector3i cand = idx + Vector3i(dx, dy, dz);
              if (cand(0) < 0 || cand(0) >= GRID_X_SIZE ||
                  cand(1) < 0 || cand(1) >= GRID_Y_SIZE ||
                  cand(2) < 0 || cand(2) >= GRID_Z_SIZE) continue;
              if (!isOccupied(cand)) {
                idx = cand;
                found = true;
              }
            }
        p = gridIndex2coord(idx);
      }

      // avoid re-introducing duplicates
      if ((p - out.back()).norm() > 1e-4) out.push_back(p);
    }
  }

  ROS_WARN("[pathSimplify] in=%d clean=%d pruned=%d fillet=%d out=%d maxSegObs=%.3f (open/mid/near max=%.2f/%.2f/%.2f)",
           (int)path.size(), (int)clean.size(), (int)pruned.size(), (int)fillet.size(), (int)out.size(),
           maxSegObserved, max_len_open, max_len_mid, max_len_near);

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
