#pragma once

// bag of words
#include "DBoW2.h"		// defines QueryResults

// opencv
#include <cv.h>
#include <highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

// #include "srba-stereo-slam.h"
#include <srba.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/poses/CPose3DRotVec.h>
#include <mrpt/utils/TStereoCamera.h>
#include <mrpt/vision/types.h>
#include <mrpt/utils/CConfigFile.h>

#include "srba-stereo-slam_common.h"

using namespace mrpt;
using namespace mrpt::poses;

typedef std::vector<cv::KeyPoint> 	TKeyPointList;
typedef std::vector<cv::DMatch> 	TDMatchList;

enum LCResult { lcr_NO_LC, lcr_FOUND_LC, lcr_BAD_DATA, lcr_NOT_ENOUGH_DATA }; //!< Enum for defining the result of loop closure search

/*********************************************
STRUCT: Data association information
**********************************************/
typedef struct t_kf_da_info {
	size_t								kf_idx;				//!< Index of the 'other' KF
	size_t								tracked_matches;	//!< Number of tracked matches with the 'other' KF
	vector< pair< size_t, double > >	tracking_info;		//!< Pair of 'other_match_idx' -> 'mean_distance'. Vector size == number of current matches

	t_kf_da_info() : 
		kf_idx(INVALID_KF_ID), 
		tracked_matches(0), 
		tracking_info( vector< pair< size_t, double > >() ) 
	{}

} t_kf_da_info;
typedef vector<t_kf_da_info> TVectorKfsDaInfo;			// Vector of DA information, one for each KEYFRAME compared with the current one

/*********************************************
STRUCT: Loop closure information
**********************************************/
typedef struct TLoopClosureInfo {
	vector<TKeyFrameID> similar_kfs;		//!< A vector containing the ids of the KFs similar to this one
	TKeyFrameID			lc_id;				//!< In case there is loop closure, here is the target KF's ID
	vector<CPose3D>		similar_kfs_poses;	//!< A vector containing the poses of the similar KFs wrt this one

	TLoopClosureInfo() : 
		similar_kfs( vector<TKeyFrameID>() ), 
		lc_id(INVALID_KF_ID), 
		similar_kfs_poses(vector<CPose3D>()) 
	{}
} TLoopClosureInfo;

/*********************************************
STRUCT: Application options
**********************************************/
typedef struct TGeneralOptions
{
	enum captureSource { csRawlog, csImgDir };	
	captureSource cap_src;						//!< [int] (def:1) -- Image source: [0] Rawlog file ; [1] Image directory

	int		from_step, 							//!< [int] (def:0) -- Number of the first frame to process
			to_step, 							//!< [int] (def:0 -unlimited-) -- Number of the last frame to process
			save_at_iteration, 					//!< [int] (def:0) -- Iteration where to save the state (TO DO)
			max_num_kfs,						//!< [int] (def:0 -unlimited-) -- Maximum number of KFs to be inserted in the system (app will finish when reached)
			start_index, 						//!< [int] (def:0) -- Input configuration (Image dir): starting image
			end_index,							//!< [int] (def:0 -unlimited-) -- Input configuration (Image dir): ending image
			verbose_level;						//!< [int] (def:0) -- Verbose level: [0] None ; [1] Important ; [2] More info
			
	bool	debug,								//!< [bool] (def:false) -- Store and show some debugging information
			show3D,								//!< [bool] (def:false) -- Show information GUI
			enable_logger,						//!< [bool] (def:false) -- Enable time logger for certain operations (for debugging). Time info will be shown at the program's end.
			load_state_from_file,				//!< [bool] (def:false) -- Load application state from file (TO DO)
			save_state_to_file,					//!< [bool] (def:false) -- Save application state to file (TO DO)
			pause_after_show_op,				//!< [bool] (def:false) -- Pause application after showing parameters
			pause_at_each_iteration;			//!< [bool] (def:false) -- Pause application after each iteration
			
	string	out_dir,							//!< [string] (def:'') -- Application output folder
			rawlog_file,						//!< [string] (def:'') -- Rawlog file path
			state_file,							//!< [string] (def:'') -- File where to save/load the application state (TO DO)
			image_dir_url,						//!< [string] (def:'') -- Image folder path
			left_format,						//!< [string] (def:'') -- Left image filename format
			right_format;						//!< [string] (def:'') -- Right image filename format

	/** Default constructor */
	TGeneralOptions() : 
		cap_src(csImgDir), 
		from_step(0), 
		to_step(0), 
		save_at_iteration(0), 
		max_num_kfs(0), 
		start_index(0), 
		end_index(0), 
		verbose_level(0),
		debug(false), 
		show3D(false), 
		enable_logger(false),
		load_state_from_file(false), 
		save_state_to_file(false), 
		pause_after_show_op(false),
		pause_at_each_iteration(false), 
		out_dir(""), 
		rawlog_file(""), 
		state_file(""), 
		image_dir_url(""), 
		left_format(""), 
		right_format("")
	{}

	/** Load data from config file */
	void loadFromConfigFile( const mrpt::utils::CConfigFile & config )
	{
		MRPT_LOAD_CONFIG_VAR(pause_after_show_op,bool,config,"APP_OPTIONS")

		MRPT_LOAD_CONFIG_VAR(out_dir,string,config,"APP_OPTIONS")
		MRPT_LOAD_CONFIG_VAR(debug,bool,config,"APP_OPTIONS")
		MRPT_LOAD_CONFIG_VAR(show3D,bool,config,"APP_OPTIONS")
		MRPT_LOAD_CONFIG_VAR(enable_logger,bool,config,"APP_OPTIONS")

		MRPT_LOAD_CONFIG_VAR(verbose_level,int,config,"APP_OPTIONS")
		MRPT_LOAD_CONFIG_VAR(pause_at_each_iteration,bool,config,"APP_OPTIONS")
		
		MRPT_LOAD_CONFIG_VAR(from_step,int,config,"APP_OPTIONS")
		MRPT_LOAD_CONFIG_VAR(to_step,int,config,"APP_OPTIONS")
		MRPT_LOAD_CONFIG_VAR(max_num_kfs,int,config,"APP_OPTIONS")
		
		MRPT_LOAD_CONFIG_VAR(save_state_to_file,bool,config,"APP_OPTIONS")
		MRPT_LOAD_CONFIG_VAR(save_at_iteration,int,config,"APP_OPTIONS")
		MRPT_LOAD_CONFIG_VAR(state_file,string,config,"APP_OPTIONS")
		if( save_state_to_file ) 
			load_state_from_file = false;
		else
		{
			MRPT_LOAD_CONFIG_VAR(load_state_from_file,bool,config,"APP_OPTIONS")
		}

		int aux = config.read_int("APP_OPTIONS","capture_source",cap_src);
		switch(aux)
		{
		case 0 : cap_src = csRawlog; break;
		case 1 : default: cap_src = csImgDir; break;
		}

		MRPT_LOAD_CONFIG_VAR(rawlog_file,string,config,"IMG_SOURCE")

		MRPT_LOAD_CONFIG_VAR(image_dir_url,string,config,"IMG_SOURCE")
		MRPT_LOAD_CONFIG_VAR(left_format,string,config,"IMG_SOURCE")
		MRPT_LOAD_CONFIG_VAR(right_format,string,config,"IMG_SOURCE")
		MRPT_LOAD_CONFIG_VAR(start_index,int,config,"IMG_SOURCE")
		MRPT_LOAD_CONFIG_VAR(end_index,int,config,"IMG_SOURCE")

	} // end-loadFromConfigFile

	/** Show options */
	void dumpToConsole( )
	{
		cout << "---------------------------------------------------------" << endl;
		cout << " Application options" << endl;
		cout << "---------------------------------------------------------" << endl;
		if( cap_src == csRawlog )
			cout << "	:: Rawlog file: " << endl << "	   " << rawlog_file << endl;
		else if( cap_src == csImgDir )
		{
			cout << "	:: Image directory: " << endl << "	   " << image_dir_url << endl;
			cout << "	:: Left image format: " << left_format << endl;
			cout << "	:: Right image format: " << right_format << endl;
			cout << "	:: Start index: " << start_index << endl;
			cout << "	:: End index: " << end_index << endl;
		}

		cout << "	:: Steps: From " << from_step << " to " << to_step << endl;
		cout << "	:: Max number of keyframes "; max_num_kfs > 0 ? cout << max_num_kfs : cout << " unlimited"; cout << endl;
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Debug?: ", debug)
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Show3D?: ", show3D)
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Enable time logger?: ", enable_logger)
		cout << "	:: Output directory: '" << out_dir << "'" << endl;
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Load state from file?: ", load_state_from_file)
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Save state to file?: ", save_state_to_file)
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Pause at each iteration?: ", pause_at_each_iteration)

		if( load_state_from_file || save_state_to_file) cout << "	:: State file: " << state_file << endl;
		
		if( pause_after_show_op )  system::pause();
	}

} TGeneralOptions;

/*********************************************
STRUCT: SRBA Stereo SLAM options
**********************************************/
typedef struct TSRBAStereoSLAMOptions
{
	// -- declare enum types
	enum TDetectMethod { DM_ORB_ONLY = 0, DM_FAST_ORB };
	enum TNonMaxSuppMethod { NMSM_STANDARD = 0, NMSM_ADAPTIVE };
	enum TDAStage2Method { ST2M_NONE = 0, ST2M_FUNDMATRIX, ST2M_CHANGEPOSE, ST2M_BOTH };

	utils::TStereoCamera	stereo_camera;			//!< The stereo camera parameters

	CPose3DRotVec			camera_pose_on_robot_rvt, 
							camera_pose_on_robot_rvt_inverse;
    
	TDetectMethod			detect_method;			//!< (def:ORB) -- Feature extraction method for SRBA system
	
	// detect
    size_t	n_levels,								//!< (def:1) -- Number of levels in the image pyramid -- fixed by now
			n_feats;								//!< (def:500) -- Desired number of feats to be detected in the images
	
	int		min_ORB_distance,						//!< (def:0) For non-max-suppression
			detect_fast_th,							//!< (def:5) -- Initial FAST Threshold for ORB keypoints (will be adaptad if 'orb_adaptive_fast_th' is true)
			adaptive_th_min_matches;				//!< (def:100) -- Minimum number of stereo matches to force adaptation of FAST and/or ORB thresholds.
	
	bool	orb_adaptive_fast_th;					//!< (def:false) -- Set/Unset adaptive FAST threshold (within ORB method) to get the desired number of feats
	
	TNonMaxSuppMethod non_max_supp_method;			//!< (def:standard) Method to perform nom maximal suppression

	vision::TMatchingOptions matching_options;		//!< Matching options

	// inter-frame match
	double	ransac_fit_prob,						//!< (def:0.95) -- Filtering by fundamental matrix (RANSAC fit threshold)
			max_y_diff_epipolar,					//!< (def:1.5) -- Filtering by fundamental matrix (epipolar threshold)
			max_orb_distance_da;					//!< (def:60) -- Maximum ORB distance for data association 

	// least-squares
	TDAStage2Method da_stage2_method;				//!< (def:2) -- Method for filtering outliers during second stage of DA (after ORB matching): [0] None ; [1] Fundamental Matrix; [2] Change in pose only ; [3] Both
	double			query_score_th;					//!< (def:0.04) -- Minimum allowed query value for the most similar KF (will raise an error if it falls below this)

	bool	use_initial_pose;						//!< (def:true) -- Use an initial estimation of the position of this KF taken from the odometry
	int		vo_id_tracking_th;						//!< (def:40) -- Threshold for the number of tracked features from last KF

	// da-filters
	bool	da_filter_by_direction,					//!< (def:false) -- Filter DA matches by their direction
			da_filter_by_orb_distance,				//!< (def:true) -- Filter DA matches by their ORB distance
			da_filter_by_fund_matrix,				//!< (def:true) -- Filter DA matches by computing left-left fundamental matrix and checking distance to epipolar lines
			da_filter_by_pose_change;				//!< (def:true) -- Filter DA matches by computing pose change and checking reprojection errors
			
	// general
	double	residual_th, 
			max_rotation,							//!< (def:15) -- Rotation limit for checking new KFs (in degrees)
			max_translation,						//!< (def:0.3) -- Translation limit for checking new KFs (in meters)
			srba_kernel_param;						//!< (def:3.0) -- Pseudo-huber kernel param for least-squares optimization.
	
	bool	non_maximal_suppression,				//!< (def:false) -- Perform non-maximal-suppression
			pause_after_show_op,					//!< (def:false) -- Pause after showing parameters
			srba_use_robust_kernel,					//!< (def:true) -- Use robust kernel for optimization
			srba_use_robust_kernel_stage1;			//!< (def:true) -- Use robust kernel for optimization in SRBA stage 1
		
	size_t	updated_matches_th,						//!< (def:50) -- Minimum number of tracked matches to insert a new KF after DA
			up_matches_th_plus,						//!< (def:25) -- This+'updated_matches_th' sets the minimum number of tracked matches to define new KF checking (geometrical) limits
			lc_distance,							//!< (def:2) -- Minimum distance between KFs to consider a loop closure
			srba_submap_size,						//!< (def:15) -- Number of KFs within submaps
			srba_max_tree_depth,					//!< (def:3) -- Maximum depth to keep spanning trees
			srba_max_optimize_depth;				//!< (def:3) -- Maximum depth to optimize the graph

	/** Default constructor */
	TSRBAStereoSLAMOptions() :  
			detect_method( DM_ORB_ONLY ), 
			n_levels(1), 
			n_feats(500), 
			min_ORB_distance(0), 
			detect_fast_th(5), 
			adaptive_th_min_matches(100), 
			orb_adaptive_fast_th(false), 
			non_max_supp_method( NMSM_STANDARD ),
			ransac_fit_prob(0.95), 
			max_y_diff_epipolar(1.5), 
			max_orb_distance_da(60),
			da_stage2_method( ST2M_CHANGEPOSE ), 
			query_score_th(0.04),
			use_initial_pose(true),
			vo_id_tracking_th(40), 
			da_filter_by_direction(false),
			da_filter_by_orb_distance(true),
			da_filter_by_fund_matrix(true),
			da_filter_by_pose_change(true),
			residual_th(50), 
			max_rotation(15.), 
			max_translation(0.30),
			srba_kernel_param(3.0),
			non_maximal_suppression(false), 
			pause_after_show_op (false),
			srba_use_robust_kernel(true),
			srba_use_robust_kernel_stage1(true),
			updated_matches_th(50), 
			up_matches_th_plus(25), 
			lc_distance(2),
			srba_submap_size(15),
			srba_max_tree_depth(3),
			srba_max_optimize_depth(3)
	{}

    /** Copy operator */
	void operator=( const TSRBAStereoSLAMOptions & o )
	{
		camera_pose_on_robot_rvt			= o.camera_pose_on_robot_rvt;
		camera_pose_on_robot_rvt_inverse	= o.camera_pose_on_robot_rvt_inverse;
		detect_method						= o.detect_method;
		n_levels							= o.n_levels;
		n_feats								= o.n_feats;
		min_ORB_distance					= o.min_ORB_distance;
		detect_fast_th						= o.detect_fast_th;
		adaptive_th_min_matches				= o.adaptive_th_min_matches;
		orb_adaptive_fast_th				= o.orb_adaptive_fast_th;
		non_max_supp_method					= o.non_max_supp_method;
		ransac_fit_prob						= o.ransac_fit_prob;
		max_y_diff_epipolar       			= o.max_y_diff_epipolar;
		max_orb_distance_da       			= o.max_orb_distance_da;
		da_stage2_method					= o.da_stage2_method;
		query_score_th						= o.query_score_th;
		use_initial_pose					= o.use_initial_pose;
		vo_id_tracking_th					= o.vo_id_tracking_th;
		da_filter_by_direction				= o.da_filter_by_direction;
		da_filter_by_orb_distance			= o.da_filter_by_orb_distance;
		da_filter_by_fund_matrix			= o.da_filter_by_fund_matrix;
		da_filter_by_pose_change			= o.da_filter_by_pose_change;
		residual_th							= o.residual_th;
		max_rotation						= o.max_rotation;
		max_translation						= o.max_translation;
		srba_kernel_param					= o.srba_kernel_param;
		non_maximal_suppression				= o.non_maximal_suppression;
		pause_after_show_op					= o.pause_after_show_op;
		srba_use_robust_kernel				= o.srba_use_robust_kernel;
		srba_use_robust_kernel_stage1		= o.srba_use_robust_kernel_stage1;
		updated_matches_th					= o.updated_matches_th;
		up_matches_th_plus					= o.up_matches_th_plus;
		lc_distance							= o.lc_distance;
		srba_submap_size					= o.srba_submap_size;
		srba_max_tree_depth					= o.srba_max_tree_depth;
		srba_max_optimize_depth				= o.srba_max_optimize_depth;
	}

	/** Load options from an .ini file */
	void loadFromConfigFile( const mrpt::utils::CConfigFile & config )
	{
		// stereo camera
		stereo_camera.loadFromConfigFile("CAMERA",config);	// will be used for both the SRBA and the Visual Odometry engines

		// general parameters
		MRPT_LOAD_CONFIG_VAR(pause_after_show_op,bool,config,"SRBA_GENERAL")
		MRPT_LOAD_CONFIG_VAR(srba_max_tree_depth,int,config,"SRBA_GENERAL")
		MRPT_LOAD_CONFIG_VAR(srba_max_optimize_depth,int,config,"SRBA_GENERAL")
		MRPT_LOAD_CONFIG_VAR(srba_submap_size,int,config,"SRBA_GENERAL")
		MRPT_LOAD_CONFIG_VAR(srba_use_robust_kernel,bool,config,"SRBA_GENERAL")
		MRPT_LOAD_CONFIG_VAR(srba_use_robust_kernel_stage1,bool,config,"SRBA_GENERAL")
		MRPT_LOAD_CONFIG_VAR(srba_kernel_param,double,config,"SRBA_GENERAL")
		
		// keypoints detection
		MRPT_LOAD_CONFIG_VAR(n_feats,int,config,"SRBA_DETECT")						
		MRPT_LOAD_CONFIG_VAR(orb_adaptive_fast_th,bool,config,"SRBA_DETECT")		
		MRPT_LOAD_CONFIG_VAR(detect_fast_th,int,config,"SRBA_DETECT")				
		MRPT_LOAD_CONFIG_VAR(adaptive_th_min_matches,int,config,"SRBA_DETECT")		

		// data association
		int aux	= config.read_int("SRBA_DATA_ASSOCIATION","da_stage2_method",da_stage2_method,false);
		switch( aux )
		{
			case 0 : default : da_stage2_method = ST2M_NONE; break;
			case 1 : da_stage2_method = ST2M_FUNDMATRIX; break;
			case 2 : da_stage2_method = ST2M_CHANGEPOSE; break;
			case 3 : da_stage2_method = ST2M_BOTH; break;
		}
		MRPT_LOAD_CONFIG_VAR(residual_th,double,config,"SRBA_DATA_ASSOCIATION")			
		MRPT_LOAD_CONFIG_VAR(max_y_diff_epipolar,double,config,"SRBA_DATA_ASSOCIATION")	
		MRPT_LOAD_CONFIG_VAR(ransac_fit_prob,double,config,"SRBA_DATA_ASSOCIATION")		
		MRPT_LOAD_CONFIG_VAR(max_orb_distance_da,double,config,"SRBA_DATA_ASSOCIATION")	
		MRPT_LOAD_CONFIG_VAR(query_score_th,double,config,"SRBA_DATA_ASSOCIATION")

		MRPT_LOAD_CONFIG_VAR(da_filter_by_direction,bool,config,"SRBA_DATA_ASSOCIATION")
		MRPT_LOAD_CONFIG_VAR(da_filter_by_orb_distance,bool,config,"SRBA_DATA_ASSOCIATION")
		MRPT_LOAD_CONFIG_VAR(da_filter_by_fund_matrix,bool,config,"SRBA_DATA_ASSOCIATION")
		MRPT_LOAD_CONFIG_VAR(da_filter_by_pose_change,bool,config,"SRBA_DATA_ASSOCIATION")

		// new kf creation
		MRPT_LOAD_CONFIG_VAR(max_rotation,double,config,"SRBA_KF_CREATION")				
		MRPT_LOAD_CONFIG_VAR(max_translation,double,config,"SRBA_KF_CREATION")			
		MRPT_LOAD_CONFIG_VAR(updated_matches_th,int,config,"SRBA_KF_CREATION")			
		MRPT_LOAD_CONFIG_VAR(up_matches_th_plus,int,config,"SRBA_KF_CREATION")			

		MRPT_LOAD_CONFIG_VAR(lc_distance,int,config,"SRBA_KF_CREATION")					
		MRPT_LOAD_CONFIG_VAR(vo_id_tracking_th,int,config,"SRBA_KF_CREATION")			
		MRPT_LOAD_CONFIG_VAR(use_initial_pose,bool,config,"SRBA_KF_CREATION")
		
		// detect_method = config.read_int   ("SRBA","srba_detect_method",detect_method,false) == 0 ? DM_ORB_ONLY : DM_FAST_ORB;
		// MRPT_LOAD_CONFIG_VAR(n_levels,int,config,"SRBA_DETECT")						// <- by now, will be 1 for only ORB
		// MRPT_LOAD_CONFIG_VAR(non_maximal_suppression,bool,config,"DETECT")		// <- for visual odometry
		
		// matching_options.loadFromConfigFile( config, "MATCH" );			// <- for stereo matching, LIKELY TO BE DELETED
		// MRPT_LOAD_CONFIG_VAR(min_ORB_distance,int,config,"DETECT")		// UNUSED
		// non_max_supp_method		= config.read_int   ("DETECT","non_max_supp_method",non_max_supp_method,false) == 0 ? NMSM_STANDARD : NMSM_ADAPTIVE;

	} // end loadFromConfigFile

	/** Show options on the console */
	void dumpToConsole( )
	{
		cout << "---------------------------------------------------------" << endl;
		cout << " Stereo SLAM system with the following options" << endl;
		cout << "---------------------------------------------------------" << endl;
		
		// General options
		cout << " [General] " << endl;
		cout << "	Max tree depth: " << srba_max_tree_depth << endl;
		cout << "	Max optimization depth: " << srba_max_optimize_depth << endl;
		cout << "	Submap size: " << srba_submap_size << endl;
		DUMP_BOOL_VAR_TO_CONSOLE("	Use robust kernel in optimization (stage 1): ", srba_use_robust_kernel_stage1)
		DUMP_BOOL_VAR_TO_CONSOLE("	Use robust kernel in optimization: ", srba_use_robust_kernel)
		if( srba_use_robust_kernel_stage1 || srba_use_robust_kernel )
		cout << "	Robust kernel parameter: " << srba_kernel_param << endl;

		// Detection options
		cout << " [Detection] " << endl;
		cout << "	Detection method: "; detect_method == 0 ? cout << "ORB" : cout << "FAST+ORB"; cout << endl;
		cout << "	Number of keypoints to detect: " << n_feats << endl;
		DUMP_BOOL_VAR_TO_CONSOLE("	Use adaptive FAST threshold in ORB: ", orb_adaptive_fast_th)
		cout << "	Initial FAST Threshold for ORB keypoints: " << detect_fast_th << endl;
		cout << "	Minimum number of matches to force adaptation of FAST/ORB thresholds: " << adaptive_th_min_matches << endl;

		// matching_options.dumpToConsole();

		// Data association options
		cout << " [Data Association] " << endl;
		DUMP_BOOL_VAR_TO_CONSOLE("	Filter by match direction?: ", da_filter_by_direction)
		DUMP_BOOL_VAR_TO_CONSOLE("	Filter by ORB distance?: ", da_filter_by_orb_distance)
		DUMP_BOOL_VAR_TO_CONSOLE("	Filter by fundamental matrix?: ", da_filter_by_fund_matrix)
		DUMP_BOOL_VAR_TO_CONSOLE("	Filter by pose change?: ", da_filter_by_pose_change)

		cout << "	Stage 2 filtering method: ";
		switch( da_stage2_method )
		{
			case ST2M_NONE : cout << "None"; break;
			case ST2M_FUNDMATRIX : cout << "Fundamental matrix"; break;
			case ST2M_CHANGEPOSE : cout << "Change in pose"; break;
			case ST2M_BOTH : cout << "Fundamental matrix + Change in pose"; break;
		} 
		cout << endl;

		cout << "	Residual threshold: " << residual_th << endl;
		cout << "	Max feat dist to ep-line in inter-frame matching: " << max_y_diff_epipolar << " px." << endl;
		cout << "	Max distance between ORB descriptors for data association: " << max_orb_distance_da << endl;
		cout << "	Probability of RANSAC Fundamental Matrix fit: " << ransac_fit_prob << endl;
		cout << "	DB query result minimum value to keep running: " << query_score_th << endl;

		// KF creation options
		cout << " [Key-frame creation] " << endl;
		cout << "	Initial threshold for testing new KF: " << max_translation << " m. and " << max_rotation << " deg." << endl;
		cout << "	Update map when # of inter-frame (IF) matches is below: " << updated_matches_th << endl;
		cout << "	Adapt movement th. when # of IF matches is below: " << up_matches_th_plus+updated_matches_th << endl;
		cout << "	KF distance to consider a Loop Closure (LC): " << lc_distance << endl;
		cout << "	Threshold for the number of tracked features from last KF: " << vo_id_tracking_th << endl;
		DUMP_BOOL_VAR_TO_CONSOLE("	Use initial pose?: ", use_initial_pose)

		if( pause_after_show_op )  system::pause();
	} // end dumpToConsole

	private:

} TSRBAStereoSLAMOptions;

/*********************************************
STRUCT: System statistics at each KF insertion
**********************************************/
typedef struct TStatsSRBA
{
	double time;
	size_t numberKFs, numberFeatsNew, numberFeatsCommon;

	TStatsSRBA( const double _time, 
				const size_t _numberFeatsNew = 0,
				const size_t _numberFeatsCommon = 0,
				const size_t _numberKFs = 0 ) : 
		time(_time), 
		numberFeatsNew(_numberFeatsNew), 
		numberFeatsCommon(_numberFeatsCommon),
		numberKFs(_numberKFs)
	{}
} TStatsSRBA;
typedef vector<TStatsSRBA> TStatsSRBAVector;

typedef vector< pair<int,float> > t_vector_pair_idx_distance;

// These are static methods (only available when the header file is included)
// -- comparison methods
bool compareKeypointLists( const TKeyPointList & list1, const Mat & desc1, const TKeyPointList & list2, const Mat & desc2 );
bool compareMatchesLists( const TDMatchList & list1, const TDMatchList & list2 );
bool compareOptions( const TSRBAStereoSLAMOptions & opt1, const TSRBAStereoSLAMOptions & opt2 );

// -- threshold management 
double updateTranslationThreshold( const double x, const double th );
double updateRotationThreshold( const double x, const double th );

// -- others
void show_kf_numbers( 
	COpenGLScenePtr 			& scene, 
	const size_t 				& num_kf, 
	const DBoW2::QueryResults 	& ret, 
	const double 				& th = 0.0 );

CPose3DRotVec getRelativePose( 
	 const TKeyFrameID		& fromId, 
	 const TKeyFrameID		& toId, 
	 const CPose3DRotVec	& voIncrPose ); // <-- ?? TODO: Remove this if unnecessary

// -- inline methods
inline void computeDispersion( 
	const TKeyPointList 	& list, 
	const TDMatchList 		& matches, 
	double 					& std_x, 
	double 					& std_y )
{
	double mx = 0, my = 0;
	for( TDMatchList::const_iterator it = matches.begin(); it != matches.end(); ++it )
	{
		mx += list[it->queryIdx].pt.x;
		my += list[it->queryIdx].pt.y;
	}
	mx /= matches.size();
	my /= matches.size();

	for( TDMatchList::const_iterator it = matches.begin(); it != matches.end(); ++it )
	{
		std_x += mrpt::utils::square(list[it->queryIdx].pt.x-mx);
		std_y += mrpt::utils::square(list[it->queryIdx].pt.y-my);
	}
	std_x = sqrt(std_x);
	std_y = sqrt(std_y);
} // end-computeDispersion

inline mrpt::math::TPoint3D projectMatchTo3D(
	 const double						& ul, 
	 const double						& vl, 
	 const double						& ur,
	 const mrpt::utils::TStereoCamera	& stereoCamera )
 {
	 // camera
	const double & cul		= stereoCamera.leftCamera.cx();
	const double & cvl		= stereoCamera.leftCamera.cy();
	const double & fl		= stereoCamera.leftCamera.fx();
	const double & cur		= stereoCamera.rightCamera.cx();
	const double & fr		= stereoCamera.rightCamera.fx();
	const double & baseline = stereoCamera.rightCameraPose[0];
			
	const double b_d = baseline/(fl*(cur-ur)+fr*(ul-cul));
	return mrpt::math::TPoint3D(b_d*fr*(ul-cul),b_d*fr*(vl-cvl),b_d*fl*fr);
 } // end-projectMatchTo3D
