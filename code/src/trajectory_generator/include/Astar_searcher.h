#ifndef _ASTART_SEARCHER_H
#define _ASTART_SEARCHER_H

#include <iostream>
#include <vector>
#include <cstdint>
#include <ros/ros.h>
#include <ros/console.h>
#include <Eigen/Eigen>
#include "backward.hpp"
#include "node.h"

class Astarpath
{	
	private:

	protected:
		uint8_t * data;
		uint8_t * data_raw;
		MappingNodePtr *** Map_Node;
		Eigen::Vector3i goalIdx;
		int GRID_X_SIZE, GRID_Y_SIZE, GRID_Z_SIZE;
		int GLXYZ_SIZE, GLYZ_SIZE;

		double resolution, inv_resolution;
		double gl_xl, gl_yl, gl_zl;
		double gl_xu, gl_yu, gl_zu;

		MappingNodePtr terminatePtr;
		std::multimap<double, MappingNodePtr> Openset;

		// 性能关键：A* 内层扩展（AstarGetSucc）绝不能频繁访问参数服务器（ros::param）。
		// 这些参数在每次 AstarSearch 开头缓存一次，供 AstarGetSucc/Theta* 直接读取。
		int hard_xy_cells_ = 1;
		int hard_z_cells_  = 0;

		// 仅重置“本次搜索用到”的节点，避免每次 O(map_size) 全图 reset 导致卡顿
		std::vector<MappingNodePtr> used_nodes_;

		// 近障碍距离场（Chessboard/L∞，保守下界）：用于快速 clearance 查询，替代大量邻域暴力扫描
		// dist_chess_[addr] = 到最近占据格的步数（26邻域每步=1），并 cap 到 dist_max_cells_+1
		std::vector<uint16_t> dist_chess_;
		int dist_max_cells_ = 0;
		bool dist_inited_ = false;
		std::vector<int> pending_obs_addr_;  // 新增占据格（monotonic add）用于增量更新

		double getHeu(MappingNodePtr node1, MappingNodePtr node2);
		void AstarGetSucc(MappingNodePtr currentPtr, std::vector<MappingNodePtr> & neighborPtrSets, std::vector<double> & edgeCostSets);		
		Eigen::Vector3d gridIndex2coord(const Eigen::Vector3i & index) const;
		Eigen::Vector3i coord2gridIndex(const Eigen::Vector3d & pt) const;
		bool isOccupied(const int & idx_x, const int & idx_y, const int & idx_z) const;
		bool isOccupied(const Eigen::Vector3i & index) const;
		bool isFree(const int & idx_x, const int & idx_y, const int & idx_z) const;
		bool isFree(const Eigen::Vector3i & index) const;
    	
		
		

	public:
		Astarpath(){};
		~Astarpath(){};
		bool AstarSearch(Eigen::Vector3d start_pt, Eigen::Vector3d end_pt);
		void resetGrid(MappingNodePtr ptr);
		void resetUsedGrids();
		bool is_occupy(const Eigen::Vector3i & index);
		bool is_occupy_raw(const Eigen::Vector3i & index);
		Eigen::Vector3i c2i(const Eigen::Vector3d & pt);

		void begin_grid_map(double _resolution, Eigen::Vector3d global_xyz_l, Eigen::Vector3d global_xyz_u, int max_x_id, int max_y_id, int max_z_id);
		void set_barrier(const double coord_x, const double coord_y, const double coord_z);

		Eigen::Vector3d coordRounding(const Eigen::Vector3d & coord);
		std::vector<Eigen::Vector3d> getPath();
		std::vector<Eigen::Vector3d> getVisitedNodes();
		std::vector<Eigen::Vector3d> pathSimplify(const std::vector<Eigen::Vector3d> &path, const double path_resolution);
		Eigen::Vector3d getPosPoly( Eigen::MatrixXd polyCoeff, int k, double t );
		int safeCheck( Eigen::MatrixXd polyCoeff, Eigen::VectorXd time);
		double perpendicularDistance(const Eigen::Vector3d point_insert,const Eigen::Vector3d point_st,const Eigen::Vector3d point_end);
        void resetOccupy();
		double nearestObsDistM(const Eigen::Vector3i &idx, int max_r_cells) const;
		bool   isTooCloseHard(const Eigen::Vector3i &idx, int hard_xy_cells, int hard_z_cells) const;
		double softClearancePenalty(const Eigen::Vector3i &idx, double soft_range_m, int soft_scan_cells) const;
		bool   lineOfSight(const Eigen::Vector3i &a, const Eigen::Vector3i &b, int hard_xy_cells, int hard_z_cells) const;
		bool   findNearestFree(const Eigen::Vector3i &seed, Eigen::Vector3i &out_free);

		// 更新/增量更新距离场（用于 soft-clearance / collision check）
		void ensureDistanceField(int max_cells);

};


#endif