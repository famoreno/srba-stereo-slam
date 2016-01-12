#pragma once
#include "srba-stereo-slam_common.h"
#include "srba-stereo-slam_utils.h"			// stereo and general options

// srba
//#define SRBA_DETAILED_TIME_PROFILING   0
//#include <srba.h>

// graph-slam
//#include <mrpt/graphslam.h> // For global map recovery only
//#include <mrpt/opengl/graph_tools.h> // To render the global map

// others
#include <fstream>

// defines
using namespace mrpt;
using namespace mrpt::gui;
using namespace mrpt::opengl;
using namespace mrpt::utils;
using namespace mrpt::vision;
using namespace mrpt::system;
using namespace mrpt::obs;
using namespace std;
using namespace srba;
using namespace cv;

class CStereoSLAMKF;
typedef vector<CStereoSLAMKF> t_vector_kf;

// data association information
/** /
typedef pair<int,size_t>	da_info_t;				// idx of the KF (INVALID_KF_ID if none), idx of the match
typedef vector<da_info_t>	vector_da_info_t;

typedef struct t_da_info{
	int			kf;					// the id of the other kf
	size_t		other_idx;			// the idx of the other match
	float		mean_distance;		// the mean ORB distance (computed from left and right inter-frame matches) between this match and the previous one

	t_da_info() : kf(INVALID_KF_ID), other_idx(0), mean_distance(0.0) {}
} t_da_info;
typedef vector<t_da_info> t_vector_da_info;
/**/
/** /
typedef struct {
	size_t			kf_id;
	Mat				desc_left, desc_right;
	vector<size_t>	ids;
	vector<size_t>  idx;							
} t_da_one_kf;
typedef vector<t_da_one_kf> t_da_input_map;
/**/
/*********************************************
CLASS: Keyframe for SRBA
**********************************************/
class CStereoSLAMKF
{
    public:
		/** Default empty constructor */
		CStereoSLAMKF() : m_kf_ID(0) {}				
#if 0
		/** Initialization constructor */
		CStereoSLAMKF(	const CObservationStereoImagesPtr	& stImgs,
						const TSRBAStereoSLAMOptions			& options ) {
			create( stImgs, options );
		} // end custom constructor

		/** Projects match with input id into 3D according to the input stereo options */
		inline TPoint3D projectMatchTo3D( const size_t idx, const TSRBAStereoSLAMOptions & options )
		{
			// camera
			const double & cul		= options.stereo_camera.leftCamera.cx();
			const double & cvl		= options.stereo_camera.leftCamera.cy();
			const double & fl		= options.stereo_camera.leftCamera.fx();
			const double & cur		= options.stereo_camera.rightCamera.cx();
			const double & fr		= options.stereo_camera.rightCamera.fx();
			const double & baseline = options.stereo_camera.rightCameraPose[0];
			
			// keypoints
			const double ul  = this->m_keypoints_left[this->m_matches[idx].queryIdx].pt.x;
			const double vl  = this->m_keypoints_left[this->m_matches[idx].queryIdx].pt.y;
			const double ur  = this->m_keypoints_right[this->m_matches[idx].trainIdx].pt.x;
			
			// aux
			const double b_d = baseline/(fl*(cur-ur)+fr*(ul-cul));
		
			return TPoint3D(b_d*fr*(ul-cul),b_d*fr*(vl-cvl),b_d*fl*fr);
		} // end-projectMatchTo3D
#endif

		/** Sets the ID of this keyframe */
		inline void setKFID( const size_t ID ) { 
			m_kf_ID = ID; 
		}

		/** Gets the last keypoints & descriptors lists and the left-right matches from the visual odometry estimator */
		inline void getDataFromVOEngine( rso::CStereoOdometryEstimator & voEngine ) { 
			voEngine.getValues( m_keypoints_left, m_keypoints_right, m_descriptors_left, m_descriptors_right, m_matches, m_matches_ID ); 
		}

		/** Generates IDs for the inner matches,
		 *		-- useful when we insert the matches into this KF from another source (eg. odometry) and IDs were not generated
		 */
		inline void generateMatchesIDs( const size_t _starting_idx )
		{
			size_t starting_idx = _starting_idx;
			for( size_t m = 0; m < m_matches.size(); ++m )
				m_matches_ID.push_back( starting_idx++ );
		} // end -- generateMatchesIDs

		/** Creates the keyframe and fill it with information
		 *      -- detect ORB(multi-scale) features in left and right image
		 *      -- match the features and create a new KF with the pairings
		 */
		void create( const CObservationStereoImagesPtr	& stImgs, const TSRBAStereoSLAMOptions	& options = TSRBAStereoSLAMOptions() /*default*/ );

#if 0
		/** Computes matchings between a set of KF and this KF according to the query results */
		void performDataAssociation(
							const t_vector_kf					& keyframes,								// INPUT
							const TLoopClosureInfo				& lc_info,									// INPUT
							rso::CStereoOdometryEstimator		& voEngine,									// INPUT
							TVectorKfsDaInfo					& out_da,									// OUTPUT
							const TStereoCamera					& stereo_camera = TStereoCamera(),			// oINPUT
							const TSRBAStereoSLAMOptions			& stSLAMOpts = TSRBAStereoSLAMOptions(),		// oINPUT
							const CPose3DRotVec					& odomPoseFromLastKF = CPose3DRotVec() );	// oINPUT
		
		/** Computes matching between a certain KF and this KF */
		void internal_performDataAssociation( 
							const CStereoSLAMKF					& other_kf,										// INPUT  -- The other KF to perform DA with
							const Mat							& curLDesc,										// INPUT
							const Mat							& curRDesc,										// INPUT
							t_kf_da_info						& out_da,										// OUTPUT -- DA information from this KF wrt the other one
							rso::CStereoOdometryEstimator		& voEngine,										// INPUT  -- The visual odometry engine to compute the change in pose
							rso::CStereoOdometryEstimator::TStereoOdometryResult & stOdomResult, 				/*output*/
							const utils::TStereoCamera			& stereo_camera = utils::TStereoCamera(),				// oINPUT -- Stereo camera parameters
							const TSRBAStereoSLAMOptions			& stSLAMOpts = TSRBAStereoSLAMOptions(),			// oINPUT -- Stereo SLAM options
							const CPose3DRotVec			& kf_ini_rel_pose = CPose3DRotVec() );		// oINPUT -- Initial estimation of the relative pose between this KF and the other one
#endif

		/** Shows the content of the keyframe on the console */
		void dumpToConsole();

		/** Saves the information of the keyframe into a set of files
		 *	-- matched features in the image
		 */
		void saveInfoToFiles( const string & str_modif = string() );

		/** Gets the input data from the list of keyframes and creates a structure with all the needed data for data association */
		/** /
		static void createInputData( 
							const t_vector_kf					& kf_list, 
							const vector< pair<size_t,size_t> > & kf_to_test, 
							t_da_input_map						& input_map );
		/**/
		// ------------------------------------------------------------------------------
		// DATA MEMBERS
		// ------------------------------------------------------------------------------
		TKeyPointList 		m_keypoints_left, m_keypoints_right;		//!< vectors of keypoints (left and right)
		Mat 				m_descriptors_left, m_descriptors_right;	//!< vectors of ORB descriptors
		TDMatchList 		m_matches;									//!< vector of l-r matches
		vector<size_t> 		m_matches_ID;								//!< vector of ids of the matches
		CPose3DRotVec 		m_camera_pose;								//!< estimated camera pose
		size_t 				m_kf_ID;									//!< the id of this keyframe
		// static size_t 		m_last_match_ID;							//!< id of the last match for all the keyframes (static)

	// ----------------------------------------------------------
	// PRIVATE METHODS
	// ----------------------------------------------------------
	private:
#if 0	
    /** Detect features */
    // <-- ?? Probably unnecessary
	void m_detect_features( 
		const CObservationStereoImagesPtr 	& stImgs, 
		const TSRBAStereoSLAMOptions 			& stSLAMOpts );

    /** Find stereo matches */
	// <-- ?? Probably unnecessary
    void m_match_features( 
		const TSRBAStereoSLAMOptions	& stSLAMOpts );

    /** Performs adaptive non-maximal suppression with the ORB features as explained in [CITE] */
	void m_adaptive_non_max_suppression(	
					const size_t				& num_out_points, 
					const vector<KeyPoint>		& keypoints, 
					const Mat					& descriptors,
					vector<KeyPoint>			& out_kp_rad,
					Mat							& out_kp_desc,
					const double				& min_radius_th = 0.0 );
#endif
#if 0
	void m_detect_outliers_with_change_in_pose ( 
					t_vector_pair_idx_distance			& other_matched, 
					const CStereoSLAMKF					& this_kf, 
					const CStereoSLAMKF					& other_kf, 
					rso::CStereoOdometryEstimator		& voEngine,
					vector<size_t>						& outliers /*output*/,
					rso::CStereoOdometryEstimator::TStereoOdometryResult & result, /*output*/
					const TStereoCamera					& stereo_camera,
					const TSRBAStereoSLAMOptions			& stSLAMOpts,
					const CPose3DRotVec					& kf_ini_rel_pose );

    /** Compute the fundamental matrix between the left images and also between the right ones and find outliers */
	void m_detect_outliers_with_F ( 
					const t_vector_pair_idx_distance	& this_matched, 
					const CStereoSLAMKF					& this_kf, 
					const CStereoSLAMKF					& other_kf, 
					vector<size_t>						& outliers,
					const TSRBAStereoSLAMOptions			& stSLAMOpts = TSRBAStereoSLAMOptions() /*default*/ );
#endif
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

