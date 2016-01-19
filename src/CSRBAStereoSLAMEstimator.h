#pragma once

#include "srba-stereo-slam_common.h"	// Common defines and headers
#include "srba-stereo-slam_utils.h"		
#include "srba-stereo-slam.h"			// mySRBA class definition
#include "CStereoSLAMKF.h"		
#include "CBoWManager.h"				// Bag of words manager		

extern TGeneralOptions general_options;

class CSRBAStereoSLAMEstimator
{
public:
	~CSRBAStereoSLAMEstimator() {} 	//!< Default destructor
	CSRBAStereoSLAMEstimator() :	//!< Default constructor
		m_last_kf_ID(0), 
		m_last_match_ID(0),
		m_last_num_tracked_feats(UNINITIALIZED_TRACKED_NUMBER)
	{
		opengl_params.span_tree_max_depth			= 1000;
		opengl_params.draw_unknown_feats_ellipses	= false;
		opengl_params.show_unknown_feats_ids		= false;
		opengl_params.draw_kf_hierarchical			= true;
	}

	TSRBAStereoSLAMOptions 					srba_options;
	mySRBA::TOpenGLRepresentationOptions	opengl_params;
	
	mySRBA 				rba;	// main member
	
	void initialize( const CConfigFile & config );
	void performStereoSLAM();
	
private:
	// -- variables
	t_vector_kf 		m_keyframes;
	size_t 				m_last_kf_ID,
						m_last_match_ID,
						m_last_num_tracked_feats;

	double 				m_max_rotation, 
						m_max_translation,
						m_max_rotation_limit,
						m_max_translation_limit;
	
	CTimeLogger 		m_time_logger,
						m_time_logger_define_kf;	// Specific for inserting a new KF (and graph optimization)
	TStatsSRBAVector 	m_stats;
	
	CBoWManager			m_bow_manager;
	
	rso::CStereoOdometryEstimator m_voEngine;
    rso::CStereoOdometryEstimator::TStereoOdometryRequest m_odom_request;
	rso::CStereoOdometryEstimator::TStereoOdometryResult  m_odom_result;
	
	// poses
	CPose3DRotVec 		m_current_pose;                 // The estimated GLOBAL pose of the camera = last_kf_pose + vo_result + camera_pose_on_robot
	CPose3DRotVec 		m_last_kf_pose;					// The GLOBAL pose of the last kf (set from RBA engine at each new KF insertion)
	CPose3DRotVec 		m_incr_pose_from_last_kf;		// Accumulated incremental pose from the last KF (computed from vo_results at each time step, reset at new KF insertion)
	CPose3DRotVec 		m_incr_pose_from_last_check;	// Accumulated incremental pose from the last check of new insertion KF (computed at each time step, reset at new KF check)
	
	// camera
	hwdrivers::CCameraSensor 	m_myCam;
	
	// gui
	gui::CDisplayWindow3DPtr 	m_win;

	// data association
	//struct TDAMatchInfo
	//{
	//	enum TDAMatchStatus{sTRACKED = 0, sNON_TRACKED, sREJ_SLOPE, sREJ_ORB, sREJ_FUND_MATRIX, sREJ_CHANGE_POSE, sREJ_CONSISTENCY};
	//	TDAMatchStatus status;
	//	size_t other_idx;
	//	double distance;
	//	TDAMatchInfo () : status(sTRACKED) , other_idx(INVALID_IDX), distance(0.0) {}
	//};

	struct TVectorDAMatchInfo
	{
		enum TDAMatchStatus{sTRACKED = 0, sNON_TRACKED, sREJ_SLOPE, sREJ_ORB, sREJ_FUND_MATRIX, sREJ_CHANGE_POSE, sREJ_CONSISTENCY};
		vector<TDAMatchStatus> m_status;
		vector<cv::DMatch> m_matches;
	
		/** initialization constructor*/
		TVectorDAMatchInfo( size_t s ) {
			m_status.resize(s,sTRACKED);
			m_matches.resize(s);
		}

		/** get size */
		size_t size() { return m_status.size(); }
	};
	
	// -- methods
	// state management
	// -- load from stream
	bool m_load_options_from_stream( std::ifstream & stream, TSRBAStereoSLAMOptions & options );
	bool m_load_keypoints_from_stream( std::ifstream & stream, TKeyPointList & keypoints, Mat & descriptors );
	bool m_load_matches_from_stream( std::ifstream & stream, TDMatchList & matches, vector<size_t> & matches_ids );
	bool m_load_state();

	// -- save to stream
	bool m_dump_keypoints_to_stream( std::ofstream & stream, const TKeyPointList & keypoints, const Mat & descriptors );
	bool m_dump_options_to_stream( std::ofstream & stream, const TSRBAStereoSLAMOptions & options );
	bool m_dump_matches_to_stream( std::ofstream & stream, const TDMatchList & matches, const vector<size_t> & matches_ids );
	bool m_save_state();
	
	// other
	LCResult m_check_loop_closure( 
		const TKeyFrameID			& new_kf_id,
		const QueryResults			& ret, 
		TLoopClosureInfo			& lc_info );
	
	bool m_get_similar_kfs( 
		const TKeyFrameID			& newKfId,
		const QueryResults			& dbQueryResults,
		TLoopClosureInfo			& out );

	/** Computes matchings between a set of KF and this KF according to the query results */
	void m_data_association(
		const CStereoSLAMKF					& kf,											// INPUT
		const TLoopClosureInfo				& lc_info,										// INPUT
		TVectorKfsDaInfo					& out_da );										// OUTPUT

	/** Computes matching between a certain KF and this KF */
	void m_internal_data_association( 
		const CStereoSLAMKF					& this_kf,										// INPUT  -- One KF to perform DA
		const CStereoSLAMKF					& other_kf,										// INPUT  -- The other KF to perform DA with
		const Mat							& curLDesc,										// INPUT  -- Left image descriptors from current KF
		const Mat							& curRDesc,										// INPUT  -- Right image descriptors from current KF
		t_kf_da_info						& out_da,										// OUTPUT -- DA information from this KF wrt the other one
		const CPose3DRotVec					& kf_ini_rel_pose = CPose3DRotVec() );			// oINPUT -- Initial estimation of the relative pose between the KFs

	/** Compute the change in pose between frames and check keypoint residuals to find outliers (to be deleted) */
	void m_detect_outliers_with_change_in_pose ( 
		t_vector_pair_idx_distance			& other_matched, 
		const CStereoSLAMKF					& this_kf, 
		const CStereoSLAMKF					& other_kf, 
		vector<size_t>						& outliers,						// OUTPUT
		const CPose3DRotVec					& kf_ini_rel_pose );

	/** Compute the change in pose between frames and check keypoint residuals to find outliers */
	void m_detect_outliers_with_change_in_pose ( 
		TVectorDAMatchInfo					& this_matches, 
		//deque<TDAMatchInfo>					& this_matches, 
		const CStereoSLAMKF					& this_kf, 
		const CStereoSLAMKF					& other_kf, 
		const CPose3DRotVec					& kf_ini_rel_pose );

    /** Compute the fundamental matrix between the left images to find outliers (to be deleted) */
	void m_detect_outliers_with_F ( 
		const t_vector_pair_idx_distance	& other_matched, 
		const CStereoSLAMKF					& this_kf, 
		const CStereoSLAMKF					& other_kf, 
		vector<size_t>						& outliers );					// OUTPUT

	/** Compute the fundamental matrix between the left images to find outliers  */
	void m_detect_outliers_with_F ( 
		TVectorDAMatchInfo					& this_matches,
		//deque<TDAMatchInfo>					& this_matches,
		const size_t						& num_tracked,
		const CStereoSLAMKF					& this_kf, 
		const CStereoSLAMKF					& other_kf );

	/** Check matches directions to find outliers */
	void m_detect_outliers_with_direction ( 
		TVectorDAMatchInfo		& this_matches, 
		// deque<TDAMatchInfo>		& this_matches, 
		const size_t			& offset,
		const CStereoSLAMKF		& this_kf, 
		const CStereoSLAMKF		& other_kf );

	/** Check ORB distances to find outliers */
	void m_detect_outliers_with_orb_distance ( 
		TVectorDAMatchInfo		& this_matches, 
		//deque<TDAMatchInfo>		& this_matches, 
		// const vector<DMatch>	& matL,
		const CStereoSLAMKF		& this_kf, 
		const CStereoSLAMKF		& other_kf );


}; // end--CSRBAStereoSLAMEstimator