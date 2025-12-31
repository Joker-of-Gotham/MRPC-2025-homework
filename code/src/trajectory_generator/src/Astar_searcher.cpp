#include "Astar_searcher.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <limits>

using namespace std;
using namespace Eigen;

namespace
{
// Conservative safety buffer (meters) for obstacle inflation used in grid marking and line checking.
// Keep modest; too large will make narrow passages impossible.
constexpr double kSafetyInflationM = 0.35;

// Line-of-sight collision checking step factor.
// step = max(resolution * kLoSStepFactor, kMinLoSStepM)
constexpr double kLoSStepFactor = 0.5;
constexpr double kMinLoSStepM   = 0.05;

// A* heuristic tie-breaker (slightly > 1 encourages progress toward goal and reduces search "zig-zag")
constexpr double kTieBreaker = 1.0 + 1e-3;

// If start/goal lies in occupied cell (common after collision), search nearest free cell within this radius (cells)
constexpr int kNearestFreeMaxR = 6;
} // namespace

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

  resolution     = _resolution;
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

void Astarpath::resetGrid(MappingNodePtr ptr)
{
  ptr->id      = 0;
  ptr->Father  = NULL;
  ptr->g_score = inf;
  ptr->f_score = inf;
}

void Astarpath::resetUsedGrids()
{
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++)
        resetGrid(Map_Node[i][j][k]);
}

/**
 * Mark obstacle cell(s).
 *
 * Key fix vs. the baseline:
 * - Inflate obstacles in XY (and slightly in Z) to avoid "aliasing holes" when map resolution is coarse.
 * - Fill downwards to ground within the inflated column to avoid sparse Z-sampling of vertical pillars
 *   causing free "gaps" (typical in point-cloud occupancy).
 */
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

  const int inflate_xy = std::max(1, (int)std::ceil(kSafetyInflationM / resolution));
  const int inflate_z  = std::max(1, (int)std::ceil(0.20 / resolution)); // mild Z inflation

  // Mark raw cell
  data_raw[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] = 1;

  // Inflate (disk in XY) + mild Z thickening and fill-down to ground
  for (int dx = -inflate_xy; dx <= inflate_xy; ++dx) {
    for (int dy = -inflate_xy; dy <= inflate_xy; ++dy) {
      if (dx * dx + dy * dy > inflate_xy * inflate_xy)
        continue;

      int nx = idx_x + dx;
      int ny = idx_y + dy;
      if (nx < 0 || nx >= GRID_X_SIZE || ny < 0 || ny >= GRID_Y_SIZE)
        continue;

      // fill to ground to avoid Z "holes" from sparse point sampling of columns
      int z_top = std::min(GRID_Z_SIZE - 1, idx_z + inflate_z);
      for (int nz = 0; nz <= z_top; ++nz) {
        data[nx * GLYZ_SIZE + ny * GRID_Z_SIZE + nz] = 1;
      }
    }
  }
}

vector<Vector3d> Astarpath::getVisitedNodes()
{
  vector<Vector3d> visited_nodes;
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++) {
        // visualize nodes in close list only
        if (Map_Node[i][j][k]->id == -1)
          visited_nodes.push_back(Map_Node[i][j][k]->coord);
      }

  ROS_WARN("visited_nodes size : %d", (int)visited_nodes.size());
  return visited_nodes;
}

Vector3d Astarpath::gridIndex2coord(const Vector3i &index)
{
  Vector3d pt;
  pt(0) = ((double)index(0) + 0.5) * resolution + gl_xl;
  pt(1) = ((double)index(1) + 0.5) * resolution + gl_yl;
  pt(2) = ((double)index(2) + 0.5) * resolution + gl_zl;
  return pt;
}

Vector3i Astarpath::coord2gridIndex(const Vector3d &pt)
{
  Vector3i idx;
  idx << min(max(int((pt(0) - gl_xl) * inv_resolution), 0), GRID_X_SIZE - 1),
      min(max(int((pt(1) - gl_yl) * inv_resolution), 0), GRID_Y_SIZE - 1),
      min(max(int((pt(2) - gl_zl) * inv_resolution), 0), GRID_Z_SIZE - 1);
  return idx;
}

Vector3i Astarpath::c2i(const Vector3d &pt)
{
  Vector3i idx;
  idx << min(max(int((pt(0) - gl_xl) * inv_resolution), 0), GRID_X_SIZE - 1),
      min(max(int((pt(1) - gl_yl) * inv_resolution), 0), GRID_Y_SIZE - 1),
      min(max(int((pt(2) - gl_zl) * inv_resolution), 0), GRID_Z_SIZE - 1);
  return idx;
}

Eigen::Vector3d Astarpath::coordRounding(const Eigen::Vector3d &coord)
{
  return gridIndex2coord(coord2gridIndex(coord));
}

inline bool Astarpath::isOccupied(const Eigen::Vector3i &index) const
{
  return isOccupied(index(0), index(1), index(2));
}

bool Astarpath::is_occupy(const Eigen::Vector3i &index)
{
  return isOccupied(index(0), index(1), index(2));
}

bool Astarpath::is_occupy_raw(const Eigen::Vector3i &index)
{
  int idx_x = index(0);
  int idx_y = index(1);
  int idx_z = index(2);
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data_raw[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] == 1));
}

inline bool Astarpath::isFree(const Eigen::Vector3i &index) const
{
  return isFree(index(0), index(1), index(2));
}

inline bool Astarpath::isOccupied(const int &idx_x, const int &idx_y,
                                  const int &idx_z) const
{
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] == 1));
}

inline bool Astarpath::isFree(const int &idx_x, const int &idx_y,
                              const int &idx_z) const
{
  return (idx_x >= 0 && idx_x < GRID_X_SIZE && idx_y >= 0 && idx_y < GRID_Y_SIZE &&
          idx_z >= 0 && idx_z < GRID_Z_SIZE &&
          (data[idx_x * GLYZ_SIZE + idx_y * GRID_Z_SIZE + idx_z] < 1));
}

/**
 * Successor generation with "no corner cutting":
 * For diagonal moves, require that the relevant axis-adjacent cells are free.
 * This prevents paths that squeeze through the corner between obstacles (common cause of collisions).
 */
inline void Astarpath::AstarGetSucc(MappingNodePtr currentPtr,
                                    vector<MappingNodePtr> &neighborPtrSets,
                                    vector<double> &edgeCostSets)
{
  neighborPtrSets.clear();
  edgeCostSets.clear();

  Vector3i Idx_neighbor;
  const int cx = currentPtr->index(0);
  const int cy = currentPtr->index(1);
  const int cz = currentPtr->index(2);

  for (int dx = -1; dx < 2; dx++) {
    for (int dy = -1; dy < 2; dy++) {
      for (int dz = -1; dz < 2; dz++) {

        if (dx == 0 && dy == 0 && dz == 0)
          continue;

        int nx = cx + dx;
        int ny = cy + dy;
        int nz = cz + dz;

        if (nx < 0 || nx >= GRID_X_SIZE ||
            ny < 0 || ny >= GRID_Y_SIZE ||
            nz < 0 || nz >= GRID_Z_SIZE)
          continue;

        // no corner cutting: if moving diagonally, ensure intermediate axis neighbors are free
        if (dx != 0 && isOccupied(cx + dx, cy, cz)) continue;
        if (dy != 0 && isOccupied(cx, cy + dy, cz)) continue;
        if (dz != 0 && isOccupied(cx, cy, cz + dz)) continue;

        // Also block 2-axis diagonals if either axis neighbor is occupied
        if (dx != 0 && dy != 0) {
          if (isOccupied(cx + dx, cy, cz) || isOccupied(cx, cy + dy, cz)) continue;
        }
        if (dx != 0 && dz != 0) {
          if (isOccupied(cx + dx, cy, cz) || isOccupied(cx, cy, cz + dz)) continue;
        }
        if (dy != 0 && dz != 0) {
          if (isOccupied(cx, cy + dy, cz) || isOccupied(cx, cy, cz + dz)) continue;
        }

        Idx_neighbor << nx, ny, nz;

        neighborPtrSets.push_back(Map_Node[nx][ny][nz]);
        edgeCostSets.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
      }
    }
  }
}

/**
 * STEP 1.1: Heuristic function
 * Use Euclidean distance in index-space (consistent with edge costs) plus a tiny tie-breaker.
 */
double Astarpath::getHeu(MappingNodePtr node1, MappingNodePtr node2)
{
  int dx = node1->index(0) - node2->index(0);
  int dy = node1->index(1) - node2->index(1);
  int dz = node1->index(2) - node2->index(2);
  double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
  return kTieBreaker * dist;
}

/**
 * STEP 1.2: A* main loop
 */
bool Astarpath::AstarSearch(Vector3d start_pt, Vector3d end_pt)
{
  ros::Time time_1 = ros::Time::now();

  // Reset node states for each query (critical for correctness across replans)
  resetUsedGrids();
  Openset.clear();
  terminatePtr = NULL;

  // If the goal comes from RViz 2D goal tools, z is often 0.0. For a flying robot this
  // frequently causes planning on an unintended Z-slice. Default to planning at current altitude.
  if (end_pt(2) <= gl_zl + 1e-3) {
    end_pt(2) = start_pt(2);
  }

  // Indices
  Vector3i start_idx = coord2gridIndex(start_pt);
  Vector3i end_idx   = coord2gridIndex(end_pt);

  auto nearestFree = [&](const Eigen::Vector3i &idx_in) -> Eigen::Vector3i {
    if (!isOccupied(idx_in))
      return idx_in;

    for (int r = 1; r <= kNearestFreeMaxR; ++r) {
      for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
          for (int dz = -r; dz <= r; ++dz) {
            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r)
              continue;
            Eigen::Vector3i cand = idx_in + Eigen::Vector3i(dx, dy, dz);
            if (cand(0) < 0 || cand(0) >= GRID_X_SIZE ||
                cand(1) < 0 || cand(1) >= GRID_Y_SIZE ||
                cand(2) < 0 || cand(2) >= GRID_Z_SIZE)
              continue;
            if (!isOccupied(cand))
              return cand;
          }
        }
      }
    }
    return idx_in;
  };

  // Snap to nearest free if necessary (helps when replanning after collision)
  start_idx = nearestFree(start_idx);
  end_idx   = nearestFree(end_idx);

  goalIdx = end_idx;

  // Use grid nodes directly (avoid allocating new nodes and ensure reset() applies)
  MappingNodePtr startPtr = Map_Node[start_idx(0)][start_idx(1)][start_idx(2)];
  MappingNodePtr endPtr   = Map_Node[end_idx(0)][end_idx(1)][end_idx(2)];

  // Set start/goal coordinates at grid centers
  startPtr->coord = gridIndex2coord(start_idx);
  endPtr->coord   = gridIndex2coord(end_idx);

  // Initialize start
  startPtr->g_score = 0.0;
  startPtr->f_score = getHeu(startPtr, endPtr);
  startPtr->id      = 1; // in open set
  startPtr->Father  = NULL;

  Openset.insert(make_pair(startPtr->f_score, startPtr));

  double tentative_g_score;
  vector<MappingNodePtr> neighborPtrSets;
  vector<double> edgeCostSets;

  while (!Openset.empty()) {

    // 1) Pop node with smallest f
    auto it = Openset.begin();
    MappingNodePtr currentPtr = it->second;
    Openset.erase(it);

    // Skip outdated entries (same pointer may be inserted multiple times)
    if (currentPtr->id == -1)
      continue;

    // Mark closed
    currentPtr->id = -1;

    // 2) Goal check
    if (currentPtr->index == goalIdx) {
      terminatePtr = currentPtr;

      ros::Time time_2 = ros::Time::now();
      if ((time_2 - time_1).toSec() > 0.1)
        ROS_WARN("Time consume in Astar path finding is %f",
                 (time_2 - time_1).toSec());
      return true;
    }

    // 3) Expand
    AstarGetSucc(currentPtr, neighborPtrSets, edgeCostSets);

    for (unsigned int i = 0; i < neighborPtrSets.size(); i++) {
      MappingNodePtr neighborPtr = neighborPtrSets[i];

      if (neighborPtr->id == -1)
        continue;

      if (isOccupied(neighborPtr->index))
        continue;

      tentative_g_score = currentPtr->g_score + edgeCostSets[i];

      // 4) Unvisited -> push to open
      if (neighborPtr->id == 0) {
        neighborPtr->Father  = currentPtr;
        neighborPtr->g_score = tentative_g_score;
        neighborPtr->f_score = tentative_g_score + getHeu(neighborPtr, endPtr);
        neighborPtr->id      = 1;

        Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
        continue;
      }
      // In open: relax
      else if (neighborPtr->id == 1) {
        if (tentative_g_score < neighborPtr->g_score) {
          neighborPtr->g_score = tentative_g_score;
          neighborPtr->f_score = tentative_g_score + getHeu(neighborPtr, endPtr);
          neighborPtr->Father  = currentPtr;

          Openset.insert(make_pair(neighborPtr->f_score, neighborPtr));
        }
        continue;
      }
    }
  }

  ros::Time time_2 = ros::Time::now();
  if ((time_2 - time_1).toSec() > 0.1)
    ROS_WARN("Time consume in Astar path finding is %f",
             (time_2 - time_1).toSec());

  ROS_WARN("[A*] search failed, no path found.");
  return false;
}

/**
 * STEP 1.3: Trace back the path
 */
vector<Vector3d> Astarpath::getPath()
{
  vector<Vector3d> path;
  vector<MappingNodePtr> front_path;

  if (terminatePtr == NULL) {
    ROS_WARN("[A*] getPath called but terminatePtr is NULL.");
    return path;
  }

  // Backtrack
  MappingNodePtr cur = terminatePtr;
  while (cur != NULL) {
    cur->coord = gridIndex2coord(cur->index);
    front_path.push_back(cur);
    cur = cur->Father;
  }

  // Reverse to start -> goal
  for (int i = (int)front_path.size() - 1; i >= 0; --i) {
    path.push_back(front_path[i]->coord);
  }

  return path;
}

/**
 * Collision-aware path simplification and adaptive resampling.
 *
 * This replaces "pure geometry" simplification (e.g., RDP) that may delete
 * critical corner points and cause the straightened segments (and subsequent
 * polynomial smoothing) to cut through obstacles.
 *
 * Output behavior:
 * - In open space: fewer waypoints (faster execution)
 * - Near obstacles / narrow passages: denser waypoints (improve tracking and safety)
 */
std::vector<Vector3d> Astarpath::pathSimplify(const vector<Vector3d> &path,
                                             double path_resolution)
{
  std::vector<Vector3d> resultPath;

  if (path.empty())
    return resultPath;
  if (path.size() == 1) {
    resultPath.push_back(path.front());
    return resultPath;
  }

  const double los_step = std::max(resolution * kLoSStepFactor, kMinLoSStepM);
  const int near_r = std::max(1, (int)std::ceil(0.20 / resolution)); // 1 cell neighborhood for "near obstacle"

  auto isNearObstacle = [&](const Eigen::Vector3i &idx) -> bool {
    for (int dx = -near_r; dx <= near_r; ++dx)
      for (int dy = -near_r; dy <= near_r; ++dy)
        for (int dz = 0; dz <= 0; ++dz) { // only in-plane "near" check for pillars
          int x = idx(0) + dx, y = idx(1) + dy, z = idx(2) + dz;
          if (isOccupied(x, y, z))
            return true;
        }
    return false;
  };

  auto segmentFree = [&](const Vector3d &a, const Vector3d &b) -> bool {
    const double L = (b - a).norm();
    const int steps = std::max(1, (int)std::ceil(L / los_step));
    for (int i = 0; i <= steps; ++i) {
      const double t = (double)i / (double)steps;
      Vector3d p = a + t * (b - a);
      Vector3i idx = coord2gridIndex(p);
      if (isOccupied(idx))
        return false;
    }
    return true;
  };

  auto segmentNearObstacle = [&](const Vector3d &a, const Vector3d &b) -> bool {
    const double L = (b - a).norm();
    const int steps = std::max(1, (int)std::ceil(L / std::max(resolution, 0.05)));
    for (int i = 0; i <= steps; ++i) {
      const double t = (double)i / (double)steps;
      Vector3d p = a + t * (b - a);
      Vector3i idx = coord2gridIndex(p);
      if (isNearObstacle(idx))
        return true;
    }
    return false;
  };

  // 0) remove consecutive duplicates
  std::vector<Vector3d> in;
  in.reserve(path.size());
  in.push_back(path.front());
  for (size_t i = 1; i < path.size(); ++i) {
    if ((path[i] - in.back()).norm() > 1e-6)
      in.push_back(path[i]);
  }
  if (in.size() == 1) {
    resultPath.push_back(in.front());
    return resultPath;
  }

  // 1) "String-pulling" pruning: jump to the farthest visible waypoint
  std::vector<Vector3d> pruned;
  pruned.reserve(in.size());
  size_t i = 0;
  pruned.push_back(in[0]);
  while (i < in.size() - 1) {
    size_t j = in.size() - 1;
    for (; j > i + 1; --j) {
      if (segmentFree(in[i], in[j]))
        break;
    }
    pruned.push_back(in[j]);
    i = j;
  }

  // 2) adaptive resampling
  resultPath.clear();
  resultPath.reserve(pruned.size() * 4);
  resultPath.push_back(pruned.front());

  // Base spacing: use input "path_resolution" but allow adaptive adjustment.
  const double base_step = std::max(0.10, path_resolution);
  const double open_step = std::max(0.80, 4.0 * base_step); // reduce segments in open areas

  for (size_t s = 0; s + 1 < pruned.size(); ++s) {
    const Vector3d a = pruned[s];
    const Vector3d b = pruned[s + 1];

    const bool nearObs = segmentNearObstacle(a, b);
    const double step = nearObs ? base_step : open_step;

    const double L = (b - a).norm();
    const int N = std::max(1, (int)std::ceil(L / step));

    for (int k = 1; k <= N; ++k) {
      const double t = (double)k / (double)N;
      Vector3d p = a + t * (b - a);

      // As a final guard, if this densified point falls into occupied due to rounding,
      // snap to nearest free cell center.
      Vector3i idx = coord2gridIndex(p);
      if (isOccupied(idx)) {
        bool found = false;
        for (int r = 1; r <= kNearestFreeMaxR && !found; ++r) {
          for (int dx = -r; dx <= r && !found; ++dx)
            for (int dy = -r; dy <= r && !found; ++dy)
              for (int dz = -r; dz <= r && !found; ++dz) {
                if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r)
                  continue;
                Vector3i cand = idx + Vector3i(dx, dy, dz);
                if (cand(0) < 0 || cand(0) >= GRID_X_SIZE ||
                    cand(1) < 0 || cand(1) >= GRID_Y_SIZE ||
                    cand(2) < 0 || cand(2) >= GRID_Z_SIZE)
                  continue;
                if (!isOccupied(cand)) {
                  idx = cand;
                  found = true;
                  break;
                }
              }
        }
        p = gridIndex2coord(idx);
      }

      resultPath.push_back(p);
    }
  }

  return resultPath;
}

double Astarpath::perpendicularDistance(const Eigen::Vector3d point_insert,
                                        const Eigen::Vector3d point_st,
                                        const Eigen::Vector3d point_end)
{
  Vector3d line1 = point_end - point_st;
  Vector3d line2 = point_insert - point_st;
  return double(line2.cross(line1).norm() / line1.norm());
}

Vector3d Astarpath::getPosPoly(MatrixXd polyCoeff, int k, double t)
{
  Vector3d ret;
  int _poly_num1D = (int)polyCoeff.cols() / 3;
  for (int dim = 0; dim < 3; dim++) {
    VectorXd coeff = (polyCoeff.row(k)).segment(dim * _poly_num1D, _poly_num1D);
    VectorXd time  = VectorXd::Zero(_poly_num1D);

    for (int j = 0; j < _poly_num1D; j++)
      if (j == 0)
        time(j) = 1.0;
      else
        time(j) = pow(t, j);

    ret(dim) = coeff.dot(time);
  }
  return ret;
}

/**
 * Safety check for the polynomial trajectory.
 * Strengthened vs baseline:
 * - Smaller sampling step (resolution/2, with a floor), reducing missed collisions
 * - Uses the inflated occupancy 'data' (and we inflated more conservatively in set_barrier)
 */
int Astarpath::safeCheck(MatrixXd polyCoeff, VectorXd time)
{
  int unsafe_segment = -1; // -1 -> the whole trajectory is safe

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
    if (unsafe_segment != -1) {
      break;
    } else {
      t = delta_t;
    }
  }
  return unsafe_segment;
}

void Astarpath::resetOccupy()
{
  for (int i = 0; i < GRID_X_SIZE; i++)
    for (int j = 0; j < GRID_Y_SIZE; j++)
      for (int k = 0; k < GRID_Z_SIZE; k++) {
        data[i * GLYZ_SIZE + j * GRID_Z_SIZE + k]     = 0;
        data_raw[i * GLYZ_SIZE + j * GRID_Z_SIZE + k] = 0;
      }
}
