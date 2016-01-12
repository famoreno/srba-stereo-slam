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

enum LCResult { lcr_NO_LC, lcr_FOUND_LC, lcr_BAD_DATA, lcr_NOT_ENOUGH_DATA }; //<! Enum for defining the result of loop closure search

/*********************************************
STRUCT: Data association information
**********************************************/
typedef struct t_kf_da_info {
	int									kf_idx;				// idx of the 'other kf'
	size_t								tracked_matches;	// number of tracked matches with the 'other kf'
	vector< pair< int, float > >		tracking_info;		// pair of 'other_match_idx' -> 'mean_distance'. Vector size == number of current matches

	t_kf_da_info() : kf_idx(INVALID_KF_ID), tracked_matches(0), tracking_info( vector< pair< int, float > >() ) {}

} t_kf_da_info;
typedef vector<t_kf_da_info> TVectorKfsDaInfo;			// Vector of DA information, one for each KEYFRAME compared with the current one

/*********************************************
STRUCT: Loop closure information
**********************************************/
typedef struct TLoopClosureInfo {
	vector<TKeyFrameID> similar_kfs;		// a vector containing the ids of the kf similar to this
	TKeyFrameID			lc_id;				// in case there is loop closure, here is the target kf id
	vector<CPose3D>		similar_kfs_poses;

	TLoopClosureInfo() : similar_kfs( vector<TKeyFrameID>() ), lc_id(INVALID_KF_ID), similar_kfs_poses(vector<CPose3D>()) {}
} TLoopClosureInfo;

/*********************************************
STRUCT: Application options
**********************************************/
typedef struct TGeneralOptions
{
	enum captureSource { csRawlog, csImgDir };	
	captureSource cap_src;						//<! Input configuration: image source -- rawlog file or image directory

	int		from_step, 							//<! Rawlog processing: first step
			to_step, 							//<! Rawlog processing: last step
			save_at_iteration, 					
			max_num_kfs,
			start_index, 						//<! Input configuration (Image dir): starting image
			end_index,							//<! Input configuration (Image dir): ending image
			verbose_level;						//<! Verbose level
			
	bool	debug, 
			show3D, 
			enable_logger,
			load_state_from_file, 
			save_state_to_file,
			pause_after_show_op, 
			pause_at_each_iteration;
			
	string	out_dir, 
			rawlog_file, 
			state_file,
			image_dir_url, 
			left_format, 
			right_format;

	TGeneralOptions() : 
		cap_src(csImgDir), 
		from_step(0), to_step(0), save_at_iteration(0), max_num_kfs(0), 
		start_index(0), end_index(100), verbose_level(0),
		debug(false), show3D(false), enable_logger(false),
		load_state_from_file(false), save_state_to_file(false), pause_after_show_op(false),
		pause_at_each_iteration(false), 
		out_dir(""), rawlog_file(""), state_file(""), 
		image_dir_url(""), left_format(""), right_format("")
	{}

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

	utils::TStereoCamera	stereo_camera;

	CPose3DRotVec			camera_pose_on_robot_rvt, 
							camera_pose_on_robot_rvt_inverse;
    
	TDetectMethod			detect_method;			//<! [def: ORB] Feature extraction method for SRBA system
	
	// detect
    size_t	n_levels,								//<! [def: 8] Number of levels in the image pyramid
			n_feats;								//<! [def: 500] Target number of feats to be detected in the images
	
	int		min_ORB_distance,						//<! [def: 0] For non-max-suppression
			detect_fast_th,							//<! [def: 5] For DM_FAST_ORB
			adaptive_th_min_matches;				//<! [def: 100] Minimum number of stereo matches to force adaptation of FAST and/or ORB thresholds.
	
	bool	orb_adaptive_fast_th;					//<! [def: false] For DM_ORB --> set if the FAST threshold (within ORB method) should be modified in order to get the desired number of feats
	
	TNonMaxSuppMethod non_max_supp_method;			//<! [def: standard] Method to perform nom maximal suppression

	vision::TMatchingOptions matching_options;		//<! [] Matching options

	// inter-frame match
	double	ransac_fit_prob,	
			max_y_diff_epipolar,
			max_orb_distance_da;

	// least-squares
	TDAStage2Method da_stage2_method;
	double			query_score_th;

	bool	use_initial_pose;					//<! Use an initial estimation of the position of this KF taken from the odometry
	int		vo_id_tracking_th;					//<! Threshold for the number of tracked features from last KF

	
	// general
	double	residual_th, 
			max_rotation, 
			max_translation, 
			lc_strictness_th;
	
	bool	non_maximal_suppression, 
			lc_strict, 
			pause_after_show_op;
	
	size_t	updated_matches_th, 
			up_matches_th_plus, 
			lc_distance;

    // ----------------------------------------------------------
	// default constructor
	// ----------------------------------------------------------
	TSRBAStereoSLAMOptions() :  
			detect_method( DM_ORB_ONLY ), 
			n_levels(1), n_feats(500), 
			min_ORB_distance(0), detect_fast_th(5), adaptive_th_min_matches(100), 
			orb_adaptive_fast_th(false), 
			non_max_supp_method( NMSM_STANDARD ),
			ransac_fit_prob(0.95), max_y_diff_epipolar(1.5), max_orb_distance_da(60),
			da_stage2_method( ST2M_CHANGEPOSE ), 
			query_score_th(0.2),
			residual_th(50), max_rotation(15.), max_translation(0.30), lc_strictness_th(0.9), 
			non_maximal_suppression(false), lc_strict(true), pause_after_show_op (false),
			updated_matches_th(50), up_matches_th_plus(25), lc_distance(2), use_initial_pose(true)
	{}

    // ----------------------------------------------------------
	// copy operator
	// ----------------------------------------------------------
	void operator=( const TSRBAStereoSLAMOptions & o )
	{
		camera_pose_on_robot_rvt			= o.camera_pose_on_robot_rvt;
		camera_pose_on_robot_rvt_inverse	= o.camera_pose_on_robot_rvt_inverse;
		residual_th							= o.residual_th;
		max_rotation						= o.max_rotation;
		max_translation						= o.max_translation;
		updated_matches_th					= o.updated_matches_th;
		up_matches_th_plus					= o.up_matches_th_plus;
		matching_options					= o.matching_options;
		max_y_diff_epipolar       			= o.max_y_diff_epipolar;
		max_orb_distance_da       			= o.max_orb_distance_da;
		ransac_fit_prob						= o.ransac_fit_prob;
		min_ORB_distance					= o.min_ORB_distance;
		detect_method						= o.detect_method;
		orb_adaptive_fast_th				= o.orb_adaptive_fast_th;
		adaptive_th_min_matches				= o.adaptive_th_min_matches;
		non_max_supp_method					= o.non_max_supp_method;
		detect_fast_th						= o.detect_fast_th;
		n_feats								= o.n_feats;
		n_levels							= o.n_levels;
		non_maximal_suppression				= o.non_maximal_suppression;
		da_stage2_method					= o.da_stage2_method;
		lc_distance							= o.lc_distance;
		lc_strictness_th					= o.lc_strictness_th;
		lc_strict							= o.lc_strict;
		pause_after_show_op					= o.pause_after_show_op;
	}

    // ----------------------------------------------------------
	// loads the options from an .ini file
	// ----------------------------------------------------------
	void loadFromConfigFile( const mrpt::utils::CConfigFile & config )
	{
		// general parameters
		MRPT_LOAD_CONFIG_VAR(pause_after_show_op,bool,config,"SRBA_GENERAL")

		// stereo camera
		stereo_camera.loadFromConfigFile("CAMERA",config);							// will be used for both the SRBA and the Visual Odometry engines
		
		// keypoints detection
		MRPT_LOAD_CONFIG_VAR(n_feats,int,config,"SRBA_DETECT")						// Number of feats to find -- will overwrite that visual odometer
		MRPT_LOAD_CONFIG_VAR(orb_adaptive_fast_th,bool,config,"SRBA_DETECT")		// Set/unset use an adaptive FAST threshold for ORB 
		MRPT_LOAD_CONFIG_VAR(detect_fast_th,int,config,"SRBA_DETECT")				// Initial FAST threshold for ORB
		MRPT_LOAD_CONFIG_VAR(adaptive_th_min_matches,int,config,"SRBA_DETECT")		// Number of matches to adapt FAST th for ORB

		// data association
		int aux	= config.read_int("SRBA_DATA_ASSOCIATION","da_stage2_method",da_stage2_method,false);
		switch( aux )
		{
			case 0 : default : da_stage2_method = ST2M_NONE; break;
			case 1 : da_stage2_method = ST2M_FUNDMATRIX; break;
			case 2 : da_stage2_method = ST2M_CHANGEPOSE; break;
			case 3 : da_stage2_method = ST2M_BOTH; break;
		}
		MRPT_LOAD_CONFIG_VAR(residual_th,double,config,"SRBA_DATA_ASSOCIATION")			// Filtering by change in pose (residual th)
		MRPT_LOAD_CONFIG_VAR(max_y_diff_epipolar,double,config,"SRBA_DATA_ASSOCIATION")	// Filtering by fundamental matrix (epipolar th)
		MRPT_LOAD_CONFIG_VAR(ransac_fit_prob,double,config,"SRBA_DATA_ASSOCIATION")		// Filtering by fundamental matrix (RANSAC fit th)
		MRPT_LOAD_CONFIG_VAR(max_orb_distance_da,double,config,"SRBA_DATA_ASSOCIATION")	// Maximum ORB distance for data association 

		// new kf creation
		MRPT_LOAD_CONFIG_VAR(max_rotation,double,config,"SRBA_KF_CREATION")				// Rotation limit for checking new KF
		MRPT_LOAD_CONFIG_VAR(max_translation,double,config,"SRBA_KF_CREATION")			// Translation limit for checking new KF
		MRPT_LOAD_CONFIG_VAR(updated_matches_th,int,config,"SRBA_KF_CREATION")			// Threshold for number of tracked matches to insert a new KF after DA
		MRPT_LOAD_CONFIG_VAR(up_matches_th_plus,int,config,"SRBA_KF_CREATION")			// Threshold above the previous threshold for defining new KF checking (geometrical) limits

		MRPT_LOAD_CONFIG_VAR(lc_distance,int,config,"SRBA_KF_CREATION")					// Minimum KFs distance for considering oop closure
		MRPT_LOAD_CONFIG_VAR(vo_id_tracking_th,int,config,"SRBA_KF_CREATION")			// Threshold for number of tracked matches (from VO) to perform a new KF check
		MRPT_LOAD_CONFIG_VAR(use_initial_pose,bool,config,"SRBA_KF_CREATION")			// For using an adaptive ORB FAST threshold		

		// detect_method = config.read_int   ("SRBA","srba_detect_method",detect_method,false) == 0 ? DM_ORB_ONLY : DM_FAST_ORB;
		// MRPT_LOAD_CONFIG_VAR(n_levels,int,config,"SRBA_DETECT")						// <- by now, will be 1 for only ORB
		// MRPT_LOAD_CONFIG_VAR(non_maximal_suppression,bool,config,"DETECT")		// <- for visual odometry
		// MRPT_LOAD_CONFIG_VAR(query_score_th,double,config,"LEAST_SQUARES")		// <- threshold for BoW query results LIKELY TO BE UNUSED
		// MRPT_LOAD_CONFIG_VAR(lc_strictness_th,double,config,"GENERAL")	//		unused
		// MRPT_LOAD_CONFIG_VAR(lc_strict,bool,config,"GENERAL")			//		unused
		// matching_options.loadFromConfigFile( config, "MATCH" );			// <- for stereo matching, LIKELY TO BE DELETED
		// MRPT_LOAD_CONFIG_VAR(min_ORB_distance,int,config,"DETECT")		// UNUSED
		// non_max_supp_method		= config.read_int   ("DETECT","non_max_supp_method",non_max_supp_method,false) == 0 ? NMSM_STANDARD : NMSM_ADAPTIVE;

	} // end loadFromConfigFile

    // ----------------------------------------------------------
	// shows the options on the console
	// ----------------------------------------------------------
	void dumpToConsole( )
	{
		cout << "---------------------------------------------------------" << endl;
		cout << " Stereo SLAM system with the following options" << endl;
		cout << "---------------------------------------------------------" << endl;
		cout << "	:: [General] Residual threshold: " << residual_th << endl;
		cout << "	:: [General] Initial threshold for testing new KF: " << max_translation << " m. and " << max_rotation << " deg." << endl;
		cout << "	:: [General] Residual threshold: " << residual_th << endl;
		cout << "	:: [General] Update map when # of inter-frame (IF) matches is below: " << updated_matches_th << endl;
		cout << "	:: [General] Adapt movement th. when # of IF matches is below: " << up_matches_th_plus+updated_matches_th << endl;
		cout << "	:: [General] KF distance to consider a Loop Closure (LC): " << lc_distance << endl;
		DUMP_BOOL_VAR_TO_CONSOLE("	:: [General] Strict LC?: ", lc_strict )
		if( !lc_strict )
			cout << "	:: [General] DB results threshold LC: " << lc_strictness_th << endl;
		cout << endl;

		cout << "	:: [Detect]  Detection method: ";
		detect_method == 0 ? cout << "ORB" : cout << "FAST+ORB"; cout << endl;
		if( detect_method == 1 ) cout << "	:: [Detect]  FAST detection threshold: " << detect_fast_th << endl; 
		cout << "	:: [Detect]  Number of ORB features: " << n_feats << endl;
		cout << "	:: [Detect]  Number of ORB levels: " << n_levels << " levels." << endl;
		DUMP_BOOL_VAR_TO_CONSOLE("	:: [Detect]  Use adaptive FAST threshold for ORB?: ", orb_adaptive_fast_th)
		DUMP_BOOL_VAR_TO_CONSOLE("	:: [Detect]  Perform Non maximal Suppression (NMS)?: ", non_maximal_suppression)

		if( non_maximal_suppression ) 
		{
			cout << "	:: [Detect]  NMS method: ";
			non_max_supp_method == 0 ? cout << "Standard" : cout << "Adaptive"; cout << endl;
			if( non_max_supp_method == 0 ) cout << "	:: [Detect]  Minimum distance between features: " << min_ORB_distance << endl;
		}
		
		matching_options.dumpToConsole();

		cout << endl;
		cout << "	:: [Match]   Max feat dist to ep-line in inter-frame matching: " << max_y_diff_epipolar << " px." << endl;
		cout << "	:: [Match]   Max distance between ORB desc for data association: " << max_orb_distance_da << endl;
		cout << "	:: [Match]   Probability of RANSAC Fundamental Matrix fit: " << ransac_fit_prob << endl;
		cout << "	:: [Match]	 Adaptive threshold matches limit: " << adaptive_th_min_matches << endl;
		cout << endl;
		cout << "	:: [LS]      Stage 2 method in data association: ";
		
		switch( da_stage2_method )
		{
			case ST2M_NONE : cout << "None"; break;
			case ST2M_FUNDMATRIX : cout << "Fundamental matrix"; break;
			case ST2M_CHANGEPOSE : cout << "Change in pose"; break;
			case ST2M_BOTH : cout << "Fundamental matrix + Change in pose"; break;
		} 
		cout << endl;

		cout << "	:: [LS]      DB query result threshold for performing DA: ";
		query_score_th != 0 ? cout << query_score_th << endl : cout << "Adaptive" << endl;

		DUMP_BOOL_VAR_TO_CONSOLE("	:: Use initial pose?: ", use_initial_pose)

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

inline double updateQueryScoreThreshold( const size_t & numberTrackedFeats )
{
	return UNINITIALIZED_TRACKED_NUMBER ? 0.2 : std::max( 0.15, std::min(0.5, (-0.35/50.0)*(numberTrackedFeats-75)+0.15 ) );
} // end-updateQueryScoreThreshold
