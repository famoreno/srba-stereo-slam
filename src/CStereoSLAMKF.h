/* +---------------------------------------------------------------------------+
   |                 The Mobile Robot Programming Toolkit (MRPT)               |
   |                                                                           |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2013, Individual contributors, see AUTHORS file        |
   | Copyright (c) 2005-2013, MAPIR group, University of Malaga                |
   | Copyright (c) 2012-2013, University of Almeria                            |
   | All rights reserved.                                                      |
   |                                                                           |
   | Redistribution and use in source and binary forms, with or without        |
   | modification, are permitted provided that the following conditions are    |
   | met:                                                                      |
   |    * Redistributions of source code must retain the above copyright       |
   |      notice, this list of conditions and the following disclaimer.        |
   |    * Redistributions in binary form must reproduce the above copyright    |
   |      notice, this list of conditions and the following disclaimer in the  |
   |      documentation and/or other materials provided with the distribution. |
   |    * Neither the name of the copyright holders nor the                    |
   |      names of its contributors may be used to endorse or promote products |
   |      derived from this software without specific prior written permission.|
   |                                                                           |
   | THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       |
   | 'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED |
   | TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR|
   | PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE |
   | FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL|
   | DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR|
   |  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)       |
   | HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,       |
   | STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN  |
   | ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           |
   | POSSIBILITY OF SUCH DAMAGE.                                               |
   +---------------------------------------------------------------------------+ */
#pragma once

// opencv
#include <cv.h>
#include <highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

// mrpt
#include <mrpt/opengl/CGridPlaneXY.h>
#include <mrpt/opengl/CText.h>
#include <mrpt/opengl/CSimpleLine.h>

#include <mrpt/vision/types.h>
#include <mrpt/vision/CFeatureExtraction.h>

#include <mrpt/obs/CRawlog.h>
#include <mrpt/hwdrivers/CCameraSensor.h>
#include <mrpt/poses/CPose3DRotVec.h>
#include <mrpt/utils/CFileStream.h>

// srba
#define SRBA_DETAILED_TIME_PROFILING   1
#include <srba.h>

// graph-slam
#include <mrpt/graphslam.h> // For global map recovery only
#include <mrpt/opengl/graph_tools.h> // To render the global map

// visual odometry
#include <libstereo-odometry.h>

// bag of words
#include "DBoW2.h"		// defines Surf64Vocabulary and Surf64Database
#include "DUtils.h"
#include "DUtilsCV.h"	// defines macros CVXX
#include "DVision.h"

// others
#include <fstream>

// defines
#define INVALID_KF_ID -1
#define INVALID_IDX -1											// to do: take these two lines to the header file
#define OUTLIER_ID -2
#define UNINITIALIZED_TRACKED_NUMBER -1
#define GENERATE_NAME_WITH_KF(STR) mrpt::format("%s\\%s_kf%04d.txt", app_options.out_dir.c_str(), #STR, this->m_kfID)
#define GENERATE_NAME_WITH_2KF(STR,OKF_ID) mrpt::format("%s\\%s_kf%04d_with_kf%04d.txt", app_options.out_dir.c_str(), #STR, this->m_kfID, OKF_ID)
#define GENERATE_NAME_WITH_KF_OUT(STR,KF) mrpt::format("%s\\%s_kf%04d.txt", app_options.out_dir.c_str(), #STR, KF.m_kfID)
#define DUMP_BOOL_VAR_TO_CONSOLE(_MSG,_VAR) cout << _MSG; _VAR ? cout << "Yes " : cout << "No "; cout << endl;

using namespace mrpt;
using namespace mrpt::gui;
using namespace mrpt::opengl;
using namespace mrpt::utils;
using namespace mrpt::vision;
using namespace mrpt::system;
using namespace mrpt::obs;
using namespace std;
using namespace srba;

using namespace DBoW2;
using namespace DUtils;
using namespace DVision;

using namespace cv;

class CStereoSLAMKF;
//typedef vector<CStereoSLAMKF> t_vector_kf;
typedef deque<CStereoSLAMKF> t_vector_kf;

typedef size_t match_idx;

// data association information
typedef pair<int,size_t>	da_info_t;				// idx of the KF (INVALID_KF_ID if none), idx of the match
typedef vector<da_info_t>	vector_da_info_t;

typedef struct t_da_info{
	int			kf;					// the id of the other kf
	size_t		other_idx;			// the idx of the other match
	float		mean_distance;		// the mean ORB distance (computed from left and right inter-frame matches) between this match and the previous one

	t_da_info() : kf(INVALID_KF_ID), other_idx(0), mean_distance(0.0) {}
} t_da_info;
typedef vector<t_da_info> t_vector_da_info;

// data association information
typedef struct t_kf_da_info {
	int									kf_idx;				// idx of the 'other kf'
	size_t								tracked_matches;	// number of tracked matches with the 'other kf'
	vector< pair< int, float > >		tracking_info;		// pair of 'other_match_idx' -> 'mean_distance'. Vector size == number of current matches

	t_kf_da_info() : kf_idx(INVALID_KF_ID), tracked_matches(0), tracking_info( vector< pair< int, float > >() ) {}

} t_kf_da_info;
typedef vector<t_kf_da_info> TVectorKfsDaInfo;			// Vector of DA information, one for each KEYFRAME compared with the current one
// ------------------------------------------------------------------

typedef struct TLoopClosureInfo {
	vector<TKeyFrameID> similar_kfs;		// a vector containing the ids of the kf similar to this
	TKeyFrameID			lc_id;				// in case there is loop closure, here is the target kf id
	vector<CPose3D>		similar_kfs_poses;

	TLoopClosureInfo() : similar_kfs( vector<TKeyFrameID>() ), lc_id(INVALID_KF_ID), similar_kfs_poses(vector<CPose3D>()) {}
} TLoopClosureInfo;

typedef vector< pair<int,float> > t_vector_pair_idx_distance;

typedef struct {
	size_t			kf_id;
	Mat				desc_left, desc_right;
	vector<size_t>	ids;
	vector<size_t>  idx;							
} t_da_one_kf;
typedef vector<t_da_one_kf> t_da_input_map;

/** Application options */
typedef struct TAppOptions
{
	enum captureSource { csRawlog, csImgDir };

	int		from_step, to_step, save_at_iteration, vo_id_tracking_th;
	int		max_num_kfs;
	bool	debug, show3D, enableLogger;
	string	out_dir, rawlog_file, state_file;
	bool	load_state_from_file, save_state_fo_file, pause_after_show_op;
	bool	pause_at_each_iteration;
	bool	useInitialPose;
	
	
	string	cap_dir_url, cap_img_left_format, cap_img_right_format;
	int		cap_img_start_index, cap_img_end_index;
	int		verbose_level;

	captureSource cap_src;

	TAppOptions() : 
		from_step(0), to_step(0), save_at_iteration(0), vo_id_tracking_th(0), max_num_kfs(0), 
		debug(false), show3D(false), enableLogger(false),
		out_dir(""), rawlog_file(""), state_file(""), 
		load_state_from_file(false), save_state_fo_file(false), pause_after_show_op(false),
		pause_at_each_iteration(false), useInitialPose(true),
		cap_dir_url(""), cap_img_left_format(""), cap_img_right_format(""), 
		cap_img_start_index(0), cap_img_end_index(100), verbose_level(0),
		cap_src(csRawlog) 
	{}

	void loadFromConfigFile( const CConfigFile & config )
	{
		this->from_step					= config.read_int("GENERAL","fromStep",this->from_step,false);	
		this->to_step					= config.read_int("GENERAL","toStep",this->to_step,false);		
		this->save_at_iteration			= config.read_int("GENERAL","save_at_iteration",this->save_at_iteration,false);	
		this->vo_id_tracking_th			= config.read_int("GENERAL","vo_id_tracking_th",this->vo_id_tracking_th,false);		
		this->max_num_kfs				= config.read_int("GENERAL","max_num_kfs",this->max_num_kfs,false);		
		this->debug						= config.read_bool("GENERAL","debug",this->debug,false);
		this->show3D					= config.read_bool("GENERAL","show3D",this->show3D,false);
		this->enableLogger				= config.read_bool("GENERAL","enableLogger",this->enableLogger,false);
		this->out_dir					= config.read_string("GENERAL","outDir",this->out_dir,false);
		this->rawlog_file				= config.read_string("GENERAL","rawlogFile",this->rawlog_file,false);
		this->load_state_from_file		= config.read_bool("GENERAL","load_state_from_file",this->load_state_from_file,false);
		this->save_state_fo_file		= config.read_bool("GENERAL","save_state_fo_file",this->save_state_fo_file,false);
		this->state_file				= config.read_string("GENERAL","state_file",this->state_file,false);

		if( this->save_state_fo_file ) 
			this->load_state_from_file = false;

		this->pause_after_show_op		= config.read_bool("GENERAL","pause_after_show_op",this->pause_after_show_op,false);
		this->pause_at_each_iteration	= config.read_bool("GENERAL","pause_at_each_iteration",this->pause_at_each_iteration,false);
		this->useInitialPose			= config.read_bool("GENERAL","use_initial_pose",this->useInitialPose,false);

		this->cap_dir_url				= config.read_string("GENERAL","cap_dir_url",this->cap_dir_url,false);
		this->cap_img_left_format		= config.read_string("GENERAL","cap_img_left_format",this->cap_img_left_format,false);
		this->cap_img_right_format		= config.read_string("GENERAL","cap_img_right_format",this->cap_img_right_format,false);
		this->cap_img_start_index		= config.read_int("GENERAL","cap_img_start_index",this->cap_img_start_index,false);
		this->cap_img_end_index			= config.read_int("GENERAL","cap_img_end_index",this->cap_img_end_index,false);
		this->verbose_level				= config.read_int("GENERAL","verbose_level",this->verbose_level,false);
		int aux							= config.read_int("GENERAL","capture_source",this->cap_src);
		switch(aux)
		{
		case 0 : this->cap_src = csRawlog; break;
		case 1 : default: this->cap_src = csImgDir; break;
		}
		
	}

	void dumpToConsole( )
	{
		cout << "---------------------------------------------------------" << endl;
		cout << " Application options" << endl;
		cout << "---------------------------------------------------------" << endl;
		if( this->cap_src == csRawlog )
			cout << "	:: Rawlog file: " << endl << "	   " << rawlog_file << endl;
		else if( this->cap_src == csImgDir )
		{
			cout << "	:: Image directory: " << endl << "	   " << this->cap_dir_url << endl;
			cout << "	:: Left image format: " << this->cap_img_left_format << endl;
			cout << "	:: Right image format: " << this->cap_img_right_format << endl;
			cout << "	:: Start index: " << this->cap_img_start_index << endl;
			cout << "	:: End index: " << this->cap_img_end_index << endl;
		}

		cout << "	:: Steps: From " << from_step << " to " << to_step << endl;
		cout << "	:: Max number of keyframes "; max_num_kfs > 0 ? cout << max_num_kfs : cout << " unlimited"; cout << endl;
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Debug?: ", debug)
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Show3D?: ", show3D)
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Enable time logger?: ", enableLogger)
		cout << "	:: Output directory: '" << out_dir << "'" << endl;
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Use initial pose?: ", useInitialPose)
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Load state from file?: ", load_state_from_file)
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Save state to file?: ", save_state_fo_file)
		DUMP_BOOL_VAR_TO_CONSOLE("	:: Pause at each iteration?: ", pause_at_each_iteration)
		if( load_state_from_file || save_state_fo_file) cout << "	:: State file: " << state_file << endl;
		
		if( pause_after_show_op )  system::pause();
	}

} TAppOptions;

/** Application options */
typedef struct TStereoSLAMOptions
{
	TStereoCamera stCamera;
	CPose3DRotVec camera_pose_on_robot_rvt, camera_pose_on_robot_rvt_inverse;

	enum TDetectMethod { DM_ORB_ONLY = 0, DM_FAST_ORB };
	enum TNonMaxSuppMethod { NMSM_STANDARD = 0, NMSM_ADAPTIVE };
	enum TDAStage2Method { ST2M_NONE = 0, ST2M_FUNDMATRIX, ST2M_CHANGEPOSE, ST2M_BOTH };

    // detect
    size_t n_levels;						// [def: 8]			number of levels in the image pyramid
    size_t n_feats;							// [def: 500]		target number of feats to be detected in the images
	int min_ORB_distance;					// [def: 0]			for non-max-suppression
	int detect_fast_th;						// [def: 5]			for DM_FAST_ORB
	bool orb_adaptive_fast_th;				// [def: false]		for DM_ORB --> set if the FAST threshold (within ORB method) should be modified in order to get the desired number of feats
	int adaptive_th_min_matches;			// [int] [def:100]	Minimum number of stereo matches to force adaptation of FAST and/or ORB thresholds.
	TDetectMethod detect_method;			// [def: ORB]		feature extraction method
	TNonMaxSuppMethod non_max_supp_method;	// [def: standard]	method to perform nom maximal suppression

	TMatchingOptions matching_options;

    // match
	double max_distance_keyframes;		// probably this can be deleted

	// inter-frame match
	double ransac_fit_prob;	
	double max_y_diff_epipolar;
	double max_orb_distance_da;

	// least-squares
	TDAStage2Method da_stage2_method;
	double query_score_th;
	
	// general
	double residual_th, max_rotation, max_translation;
	bool non_maximal_suppression, lc_strict;
	size_t updated_matches_th, up_matches_th_plus, lc_distance;
	double lc_strictness_th;
	bool	pause_after_show_op;

    // ----------------------------------------------------------
	// default constructor
	// ----------------------------------------------------------
	TStereoSLAMOptions() :  stCamera(), camera_pose_on_robot_rvt(CPose3DRotVec()), camera_pose_on_robot_rvt_inverse(CPose3DRotVec()),
							n_levels(8), n_feats(500), min_ORB_distance(0), detect_fast_th(5), orb_adaptive_fast_th(false), adaptive_th_min_matches(100), detect_method( DM_ORB_ONLY ), non_max_supp_method( NMSM_STANDARD ),
                            matching_options(), max_y_diff_epipolar(1.5), ransac_fit_prob(0.95), max_orb_distance_da(60),
							da_stage2_method( ST2M_CHANGEPOSE ), query_score_th(0.2),
							residual_th(50), max_rotation(15.), max_translation(0.30), non_maximal_suppression(false), lc_strict(true), updated_matches_th(50), up_matches_th_plus(25), 
							lc_distance(2), lc_strictness_th(0.9), pause_after_show_op (false)
                            {}

    // ----------------------------------------------------------
	// copy operator
	// ----------------------------------------------------------
	void operator=( const TStereoSLAMOptions &o )
	{
		this->camera_pose_on_robot_rvt	= o.camera_pose_on_robot_rvt;
		this->camera_pose_on_robot_rvt_inverse	= o.camera_pose_on_robot_rvt_inverse;
		this->residual_th				= o.residual_th;
		this->max_rotation				= o.max_rotation;
		this->max_translation			= o.max_translation;
		this->max_distance_keyframes	= o.max_distance_keyframes;
		this->updated_matches_th		= o.updated_matches_th;
		this->up_matches_th_plus		= o.up_matches_th_plus;
		this->matching_options			= o.matching_options;
		this->max_y_diff_epipolar       = o.max_y_diff_epipolar;
		this->max_orb_distance_da       = o.max_orb_distance_da;
		this->ransac_fit_prob			= o.ransac_fit_prob;
		this->min_ORB_distance			= o.min_ORB_distance;
		this->detect_method				= o.detect_method;
		this->orb_adaptive_fast_th		= o.orb_adaptive_fast_th;
		this->adaptive_th_min_matches	= o.adaptive_th_min_matches;
		this->non_max_supp_method		= o.non_max_supp_method;
		this->detect_fast_th			= o.detect_fast_th;
		this->n_feats					= o.n_feats;
		this->n_levels					= o.n_levels;
		this->non_maximal_suppression	= o.non_maximal_suppression;
		this->da_stage2_method			= o.da_stage2_method;
		this->lc_distance				= o.lc_distance;
		this->lc_strictness_th			= o.lc_strictness_th;
		this->lc_strict					= o.lc_strict;
		this->pause_after_show_op		= o.pause_after_show_op;
	}

    // ----------------------------------------------------------
	// loads the options from an .ini file
	// ----------------------------------------------------------
	void loadFromConfigFile( const CConfigFile & config )
	{
		stCamera.loadFromConfigFile("CAMERA",config);

		this->residual_th				= config.read_double("GENERAL","residual_th",this->residual_th,false);
		this->max_rotation				= config.read_double("GENERAL","max_rotation",this->max_rotation,false);
		this->max_translation			= config.read_double("GENERAL","max_translation",this->max_translation,false);
		this->max_distance_keyframes	= config.read_double("GENERAL","max_distance_KF",this->max_distance_keyframes,false);
		this->updated_matches_th		= config.read_int   ("GENERAL","updated_matches_th",this->updated_matches_th,false);
		this->up_matches_th_plus		= config.read_int   ("GENERAL","up_matches_th_plus",this->up_matches_th_plus,false);
		this->lc_distance				= config.read_int   ("GENERAL","lc_distance",this->lc_distance,false);
		this->lc_strictness_th			= config.read_double("GENERAL","lc_strictness_th",this->lc_strictness_th,false);
		this->lc_strict					= config.read_bool  ("GENERAL","lc_strict",this->lc_strict,false);
		this->pause_after_show_op		= config.read_bool  ("GENERAL","pause_after_show_op",this->pause_after_show_op,false);

		this->matching_options.loadFromConfigFile( config, "MATCH" );

		this->max_y_diff_epipolar       = config.read_double("MATCH","max_y_diff_epipolar",this->max_y_diff_epipolar,false);
		this->max_orb_distance_da       = config.read_double("MATCH","max_orb_distance_da",this->max_orb_distance_da,false);
		this->ransac_fit_prob			= config.read_double("MATCH","ransac_fit_prob",this->ransac_fit_prob,false);

		this->min_ORB_distance			= config.read_int ("DETECT","min_distance",this->min_ORB_distance,false);
		this->detect_method				= config.read_int ("DETECT","detect_method",this->detect_method,false) == 0 ? DM_ORB_ONLY : DM_FAST_ORB;
		this->non_max_supp_method		= config.read_int ("DETECT","non_max_supp_method",this->non_max_supp_method,false) == 0 ? NMSM_STANDARD : NMSM_ADAPTIVE;
		this->detect_fast_th			= config.read_int ("DETECT","detect_fast_th",this->detect_fast_th,false);
		this->n_feats					= config.read_int ("DETECT","orb_nfeats",this->n_feats,false);
		this->n_levels					= config.read_int ("DETECT","orb_nlevels",this->n_levels,false);
		this->non_maximal_suppression	= config.read_bool("DETECT","non_maximal_suppression",this->non_maximal_suppression,false);
		this->orb_adaptive_fast_th		= config.read_bool("DETECT","orb_adaptive_fast_th",this->orb_adaptive_fast_th,false);
		this->adaptive_th_min_matches	= config.read_int ("MATCH","adaptive_th_min_matches",this->adaptive_th_min_matches,false);
		this->query_score_th			= config.read_double("LEAST_SQUARES","query_score_th",this->query_score_th,false);
		int aux							= config.read_int   ("LEAST_SQUARES","stage2_method",this->da_stage2_method,false);
		switch( aux )
		{
			case 0 : default : this->da_stage2_method = ST2M_NONE; break;
			case 1 : this->da_stage2_method = ST2M_FUNDMATRIX; break;
			case 2 : this->da_stage2_method = ST2M_CHANGEPOSE; break;
			case 3 : this->da_stage2_method = ST2M_BOTH; break;
		}
	} // end loadFromConfigFile

    // ----------------------------------------------------------
	// shows the options on the console
	// ----------------------------------------------------------
	void dumpToConsole( )
	{
		cout << "---------------------------------------------------------" << endl;
		cout << " Stereo SLAM system with the following options" << endl;
		cout << "---------------------------------------------------------" << endl;
		cout << "	:: [General] Maximum distance between KF: " << max_distance_keyframes << " m." << endl;
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

		if( pause_after_show_op )  system::pause();
	} // end dumpToConsole

	private:

} TStereoSLAMOptions;

/** CLASS REPRESENTING A KEYFRAME FOR SPARSE BUNDLE ADJUSTMENT */
class CStereoSLAMKF
{
    public:
		/** Default empty constructor */
		CStereoSLAMKF() :
			m_keyPointsLeft(), m_keyPointsRight(),
			m_keyDescLeft(), m_keyDescRight(),
			m_matches(),
			m_matches_ID(),
			m_camPose(),	
			m_kfID(0)
			{}				

		/** Initialization constructor */
		CStereoSLAMKF(	const CObservationStereoImagesPtr	& stImgs,
						const TStereoSLAMOptions			& options ) {
			this->create( stImgs, options );
		} // end custom constructor

		/** Projects match with input id into 3D according to the input stereo options */
		inline TPoint3D projectMatchTo3D( const size_t idx, const TStereoSLAMOptions & options )
		{
			// camera
			const double & cul		= options.stCamera.leftCamera.cx();
			const double & cvl		= options.stCamera.leftCamera.cy();
			const double & fl		= options.stCamera.leftCamera.fx();
			const double & cur		= options.stCamera.rightCamera.cx();
			const double & fr		= options.stCamera.rightCamera.fx();
			const double & baseline = options.stCamera.rightCameraPose[0];
			
			// keypoints
			const double ul  = this->m_keyPointsLeft[this->m_matches[idx].queryIdx].pt.x;
			const double vl  = this->m_keyPointsLeft[this->m_matches[idx].queryIdx].pt.y;
			const double ur  = this->m_keyPointsRight[this->m_matches[idx].trainIdx].pt.x;
			
			// aux
			const double b_d = baseline/(fl*(cur-ur)+fr*(ul-cul));
		
			return TPoint3D(b_d*fr*(ul-cul),b_d*fr*(vl-cvl),b_d*fl*fr);
		} // end-projectMatchTo3D

		/** Sets the ID of this keyframe */
		inline void setKFID( const size_t ID ) { 
			this->m_kfID = ID; 
		}

		/** Gets the last keypoints & descriptors lists and the left-right matches from the visual odometry estimator */
		inline void getDataFromVOEngine( rso::CStereoOdometryEstimator & voEngine ) { 
			voEngine.getValues( this->m_keyPointsLeft, this->m_keyPointsRight, this->m_keyDescLeft, this->m_keyDescRight, this->m_matches ); 
		}

		/** Generates IDs for the inner matches,
		 *		-- useful when we insert the matches into this KF from another source (eg. odometry) and IDs were not generated
		 */
		inline void generateMatchesIDs()
		{
			for( size_t m = 0; m < this->m_matches.size(); ++m )
				this->m_matches_ID.push_back( CStereoSLAMKF::m_last_match_ID++ );
		} // end -- generateMatchesIDs

		/** Inserts ALL the descriptors from this frame (left image only) into a binary database */
		inline void insertIntoDB( BriefDatabase & db )
		{
			vector<BRIEF::bitset> out;
			m_change_structure_binary( this->m_keyDescLeft, out );
			db.add( out );
		} // end -- insertIntoDB

		/** Queries the actual left image features into the binary database */
		inline void queryDB( BriefDatabase & db, QueryResults & ret, unsigned int num_results = 1 )
		{ 
			vector<BRIEF::bitset> out;
			m_change_structure_binary( this->m_keyDescLeft, out );
			db.query( out, ret, num_results ); 
		} // end -- queryDB

		/** Creates the keyframe and fill it with information
		 *      -- detect ORB(multi-scale) features in left and right image
		 *      -- match the features and create a new KF with the pairings
		 */
		void create( const CObservationStereoImagesPtr	& stImgs, const TStereoSLAMOptions	& options = TStereoSLAMOptions() /*default*/ );

		/** Computes matchings between a set of KF and this KF according to the query results */
		void performDataAssociation(
							const t_vector_kf					& keyframes,								// INPUT
							const TLoopClosureInfo						& lc_info,									// INPUT
							rso::CStereoOdometryEstimator		& voEngine,									// INPUT
							TVectorKfsDaInfo				& out_da,									// OUTPUT
							const TStereoCamera					& stereo_camera = TStereoCamera(),			// oINPUT
							const TStereoSLAMOptions			& stSLAMOpts = TStereoSLAMOptions(),		// oINPUT
							const CPose3DRotVec					& odomPoseFromLastKF = CPose3DRotVec() );	// oINPUT
		
		/** Computes matching between a certain KF and this KF */
		void internal_performDataAssociation( 
							const CStereoSLAMKF					& other_kf,										// INPUT  -- The other KF to perform DA with
							const Mat							& curLDesc,										// INPUT
							const Mat							& curRDesc,										// INPUT
							t_kf_da_info						& out_da,										// OUTPUT -- DA information from this KF wrt the other one
							rso::CStereoOdometryEstimator		& voEngine,										// INPUT  -- The visual odometry engine to compute the change in pose
							rso::CStereoOdometryEstimator::TStereoOdometryResult & stOdomResult, /*output*/
							const TStereoCamera					& stereo_camera = TStereoCamera(),				// oINPUT -- Stereo camera parameters
							const TStereoSLAMOptions			& stSLAMOpts = TStereoSLAMOptions(),			// oINPUT -- Stereo SLAM options
							const CPose3DRotVec					& kf_ini_rel_pose = CPose3DRotVec() );		// oINPUT -- Initial estimation of the relative pose between this KF and the other one

		/** Shows the content of the keyframe on the console */
		void dumpToConsole();

		/** Saves the information of the keyframe into a set of files
		 *	-- matched features in the image
		 */
		void saveInfoToFiles( const string & str_modif = string() );

		/** Gets the input data from the list of keyframes and creates a structure with all the needed data for data association */
		static void createInputData( 
							const t_vector_kf					& kf_list, 
							const vector< pair<size_t,size_t> > & kf_to_test, 
							t_da_input_map						& input_map );

		// ------------------------------------------------------------------------------
		// DATA MEMBERS
		// ------------------------------------------------------------------------------
		vector<KeyPoint> m_keyPointsLeft, m_keyPointsRight;		//!< vectors of keypoints (left and right)
		Mat m_keyDescLeft, m_keyDescRight;						//!< vectors of ORB descriptors
		vector<DMatch> m_matches;								//!< vector of l-r matches
		vector<size_t> m_matches_ID;							//!< vector of ids of the matches
		CPose3DRotVec m_camPose;								//!< estimated camera pose
		size_t m_kfID;											//!< the id of this keyframe
		static size_t m_last_match_ID;							//!< id of the last match for all the keyframes (static)

	// ----------------------------------------------------------
	// PRIVATE METHODS
	// ----------------------------------------------------------
	private:

	/** Transforms Mat containing descriptors to a readable vector for BoW */
	void m_change_structure_binary( const Mat & plain, vector<BRIEF::bitset> & out );

    /** Detect features */
    void m_detect_features( const CObservationStereoImagesPtr & stImgs, const TStereoSLAMOptions & stSLAMOpts );

    /** Find stereo matches */
    void m_match_features( const TStereoSLAMOptions	& stSLAMOpts );

    /** Performs adaptive non-maximal suppression with the ORB features as explained in [CITE] */
	void m_adaptive_non_max_suppression(	
					const size_t				& num_out_points, 
					const vector<KeyPoint>		& keypoints, 
					const Mat					& descriptors,
					vector<KeyPoint>			& out_kp_rad,
					Mat							& out_kp_desc,
					const double				& min_radius_th = 0.0 );

	void m_detect_outliers_with_change_in_pose ( 
					t_vector_pair_idx_distance			& other_matched, 
					const CStereoSLAMKF					& this_kf, 
					const CStereoSLAMKF					& other_kf, 
					rso::CStereoOdometryEstimator		& voEngine,
					vector<size_t>						& outliers /*output*/,
					rso::CStereoOdometryEstimator::TStereoOdometryResult & result, /*output*/
					const TStereoCamera					& stereo_camera,
					const TStereoSLAMOptions			& stSLAMOpts,
					const CPose3DRotVec					& kf_ini_rel_pose );

    /** Compute the fundamental matrix between the left images and also between the right ones and find outliers */
	void m_detect_outliers_with_F ( 
					const t_vector_pair_idx_distance	& this_matched, 
					const CStereoSLAMKF					& this_kf, 
					const CStereoSLAMKF					& other_kf, 
					vector<size_t>						& outliers,
					const TStereoSLAMOptions			& stSLAMOpts = TStereoSLAMOptions() /*default*/ );

}; // end -- CStereoSLAMKF

/** KpResponseSorter: helpful struct to order a vector of KeyPoints according to their response */
struct KpResponseSorter : public std::binary_function<size_t,size_t,bool>
{
	const vector<cv::KeyPoint> & m_data;
	KpResponseSorter( const vector<cv::KeyPoint> & data ) : m_data( data ) { }
	bool operator() (size_t k1, size_t k2 ) const {
		return (m_data[k1].response > m_data[k2].response);
	}
}; // end -- KpResponseSorter

/** DATrackedSorter: helpful struct to order a vector of t_kf_da_info according to the number of tracked features */
struct DATrackedSorter : public std::binary_function<size_t,size_t,bool>
{
	const TVectorKfsDaInfo & m_data;
	DATrackedSorter( const TVectorKfsDaInfo & data ) : m_data( data ) { }
	bool operator() (size_t k1, size_t k2 ) const {
		return ( m_data[k1].tracked_matches > m_data[k2].tracked_matches );
	}
}; // end -- DATrackedSorter