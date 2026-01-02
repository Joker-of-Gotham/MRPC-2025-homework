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
// Tunables
// ============================================================

// Weighted A*: f = g + w*h (w > 1 makes search faster but less optimal)
static constexpr double kWAstarW = 1.05;
// heuristic tie-breaker to break symmetry
static constexpr double kTieBreaker = 1.0 + 1e-4;
// Line-of-Sight sampling step (finer = safer)
static constexpr double kLosSampleRatio = 0.3;  // relative to resolution

// Obstacle inflation radii (in cells)
static constexpr int kInflationXY = 2;
static constexpr int kInflationZ = 1;

// Maximum search time (seconds)
static constexpr double kMaxSearchTime = 0.5;

// Stored goal for path reconstruction
static Eigen::Vector3d g_goal_pt(0, 0, 0);
static Eigen::Vector3d g_start_pt(0, 0, 0);

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

  // raw mark (original obstacle position)
  data_raw[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] = 1;

  // Inflate obstacles to ensure safe clearance
  const int infl_xy = kInflationXY;
  const int infl_z  = kInflationZ;

  for (int dx = -infl_xy; dx <= infl_xy; ++dx) {
    for (int dy = -infl_xy; dy <= infl_xy; ++dy) {
      const int nx = idx_x + dx;
      const int ny = idx_y + dy;
      if (nx < 0 || nx >= GRID_X_SIZE || ny < 0 || ny >= GRID_Y_SIZE) continue;

      // Circular inflation for XY
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
  return coord2gridIndex(pt);
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
// Clearance helpers
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
  return x * x;
}

// ============================================================
// A* Successors with corner-cutting prevention
// ============================================================

inline void Astarpath::AstarGetSucc(MappingNodePtr currentPtr,
                                    vector<MappingNodePtr> &neighborPtrSets,
                                    vector<double> &edgeCostSets)
{
  neighborPtrSets.clear();
  edgeCostSets.clear();

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

        Vector3i nidx(nx, ny, nz);
        if (isOccupied(nidx)) continue;

        // Prevent corner cutting: check intermediate cells
        if (dx != 0) {
          Vector3i ax(x + dx, y, z);
          if (isOccupied(ax)) continue;
        }
        if (dy != 0) {
          Vector3i by(x, y + dy, z);
          if (isOccupied(by)) continue;
        }
        if (dz != 0) {
          Vector3i cz(x, y, z + dz);
          if (isOccupied(cz)) continue;
        }

        // Check diagonal cells in each plane
        if (dx != 0 && dy != 0) {
          Vector3i cxy(x + dx, y + dy, z);
          if (isOccupied(cxy)) continue;
        }
        if (dx != 0 && dz != 0) {
          Vector3i cxz(x + dx, y, z + dz);
          if (isOccupied(cxz)) continue;
        }
        if (dy != 0 && dz != 0) {
          Vector3i cyz(x, y + dy, z + dz);
          if (isOccupied(cyz)) continue;
        }

        neighborPtrSets.push_back(Map_Node[nx][ny][nz]);
        edgeCostSets.push_back(std::sqrt(double(dx*dx + dy*dy + dz*dz)));
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
// Find nearest free cell to a given cell
// ============================================================

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

// ============================================================
// A* Search - Clean and Robust
// ============================================================

bool Astarpath::AstarSearch(Vector3d start_pt, Vector3d end_pt)
{
  ros::Time t0 = ros::Time::now();

  // Reset grid state
  resetUsedGrids();
  Openset.clear();
  terminatePtr = NULL;

  // Store goal for path reconstruction
  g_start_pt = start_pt;
  g_goal_pt = end_pt;

  // Clamp into map bounds
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

  // Handle occupied start/goal by finding nearest free
  if (isOccupied(start_idx)) {
    Vector3i free_idx;
    if (findNearestFree(start_idx, free_idx)) {
      start_idx = free_idx;
      start_pt = gridIndex2coord(start_idx);
      ROS_WARN("[A*] Start occupied, snapped to nearby free cell.");
    } else {
      ROS_ERROR("[A*] Start occupied and no nearby free cell.");
      return false;
    }
  }

  if (isOccupied(end_idx)) {
    Vector3i free_idx;
    if (findNearestFree(end_idx, free_idx)) {
      end_idx = free_idx;
      end_pt = gridIndex2coord(end_idx);
      goalIdx = end_idx;
      ROS_WARN("[A*] Goal occupied, snapped to nearby free cell.");
    } else {
      ROS_ERROR("[A*] Goal occupied and no nearby free cell.");
      return false;
    }
  }

  MappingNodePtr startPtr = Map_Node[start_idx(0)][start_idx(1)][start_idx(2)];
  MappingNodePtr endPtr   = Map_Node[end_idx(0)][end_idx(1)][end_idx(2)];

  startPtr->coord = gridIndex2coord(start_idx);
  endPtr->coord   = gridIndex2coord(end_idx);

  // Trivial case
  if (start_idx == end_idx) {
    terminatePtr = startPtr;
    return true;
  }

  startPtr->g_score = 0.0;
  startPtr->f_score = kWAstarW * getHeu(startPtr, endPtr);
  startPtr->id = 1;
  startPtr->Father = NULL;
  Openset.insert(make_pair(startPtr->f_score, startPtr));

  vector<MappingNodePtr> neighborPtrSets;
  vector<double> edgeCostSets;

  while (!Openset.empty()) {
    // Time budget check
    if ((ros::Time::now() - t0).toSec() > kMaxSearchTime) {
      ROS_WARN("[A*] Time budget exceeded.");
      return false;
    }

    auto it = Openset.begin();
    MappingNodePtr currentPtr = it->second;
    Openset.erase(it);

    if (currentPtr->id != 1) continue;

    // Check if goal reached
    if (currentPtr->index == goalIdx) {
      terminatePtr = currentPtr;
      ROS_INFO("[A*] Goal found!");
      return true;
    }

    currentPtr->id = -1;  // Mark as closed

    AstarGetSucc(currentPtr, neighborPtrSets, edgeCostSets);

    for (unsigned int i = 0; i < neighborPtrSets.size(); i++) {
      MappingNodePtr neighborPtr = neighborPtrSets[i];
      if (neighborPtr->id == -1) continue;

      double tentative_g = currentPtr->g_score + edgeCostSets[i];

      if (neighborPtr->id == 0) {
        // New node
        neighborPtr->Father  = currentPtr;
        neighborPtr->g_score = tentative_g;
        neighborPtr->f_score = tentative_g + kWAstarW * getHeu(neighborPtr, endPtr);
        neighborPtr->id = 1;
        neighborPtr->coord = gridIndex2coord(neighborPtr->index);
        Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
      } else if (neighborPtr->id == 1) {
        // Already in open set, check if this path is better
        if (tentative_g < neighborPtr->g_score - 1e-9) {
          neighborPtr->Father  = currentPtr;
          neighborPtr->g_score = tentative_g;
          neighborPtr->f_score = tentative_g + kWAstarW * getHeu(neighborPtr, endPtr);
          neighborPtr->coord = gridIndex2coord(neighborPtr->index);
          Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
        }
      }
    }
  }

  ROS_WARN("[A*] No path found (open set exhausted).");
  return false;
}

// ============================================================
// Path retrieval
// ============================================================

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

// ============================================================
// Line-of-Sight check
// ============================================================

bool Astarpath::lineOfSight(const Vector3d& start, const Vector3d& end) {
  Vector3d direction = end - start;
  double distance = direction.norm();

  if (distance < 1e-6) return true;

  // Use fine sampling
  double step_size = resolution * kLosSampleRatio;
  int num_steps = static_cast<int>(distance / step_size) + 1;

  for (int i = 0; i <= num_steps; i++) {
    double t = static_cast<double>(i) / num_steps;
    Vector3d point = start + t * (end - start);
    Vector3i idx = coord2gridIndex(point);

    if (isOccupied(idx)) {
      return false;
    }
  }
  return true;
}

// ============================================================
// Path simplification with Line-of-Sight pruning
// ============================================================

std::vector<Vector3d> Astarpath::pathSimplify(const vector<Vector3d> &path,
                                              double path_resolution)
{
  if (path.size() <= 2) {
    return path;
  }

  vector<Vector3d> simplified_path;
  simplified_path.push_back(path[0]);  // Keep start

  size_t current = 0;

  while (current < path.size() - 1) {
    // Find the farthest point we can reach directly
    size_t farthest = current + 1;

    for (size_t i = path.size() - 1; i > current + 1; i--) {
      if (lineOfSight(path[current], path[i])) {
        farthest = i;
        break;
      }
    }

    simplified_path.push_back(path[farthest]);
    current = farthest;
  }

  ROS_INFO("[pathSimplify] in=%d out=%d", (int)path.size(), (int)simplified_path.size());

  return simplified_path;
}

// ============================================================
// Helper functions
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

// ============================================================
// Trajectory safety check with fine sampling
// ============================================================

int Astarpath::safeCheck(MatrixXd polyCoeff, VectorXd time) {
  int unsafe_segment = -1;

  // Use very fine sampling to catch any collision
  const double delta_t = std::max(0.02, resolution * 0.3);

  for (int i = 0; i < polyCoeff.rows(); i++) {
    double t = 0.0;
    while (t <= time(i)) {
      Vector3d pos = getPosPoly(polyCoeff, i, t);
      Vector3i idx = coord2gridIndex(pos);

      // Check if in map bounds
      if (pos(0) < gl_xl || pos(0) >= gl_xu ||
          pos(1) < gl_yl || pos(1) >= gl_yu ||
          pos(2) < gl_zl || pos(2) >= gl_zu) {
        unsafe_segment = i;
        break;
      }

      if (isOccupied(idx)) {
        unsafe_segment = i;
        break;
      }

      t += delta_t;
    }
    if (unsafe_segment != -1) break;
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
