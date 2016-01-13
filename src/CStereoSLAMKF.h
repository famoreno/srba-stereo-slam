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

/*********************************************
CLASS: Keyframe for SRBA
**********************************************/
class CStereoSLAMKF
{
    public:
		/** Default empty constructor */
		CStereoSLAMKF() : m_kf_ID(0) {}				

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

		/** Shows the content of the keyframe on the console */
		void dumpToConsole();

		/** Saves the information of the keyframe into a set of files
		 *	-- matched features in the image
		 */
		void saveInfoToFiles( const string & str_modif = string() );

		// ------------------------------------------------------------------------------
		// DATA MEMBERS
		// ------------------------------------------------------------------------------
		TKeyPointList 		m_keypoints_left, m_keypoints_right;		//!< vectors of keypoints (left and right)
		Mat 				m_descriptors_left, m_descriptors_right;	//!< vectors of ORB descriptors
		TDMatchList 		m_matches;									//!< vector of l-r matches
		vector<size_t> 		m_matches_ID;								//!< vector of ids of the matches
		CPose3DRotVec 		m_camera_pose;								//!< estimated camera pose
		size_t 				m_kf_ID;									//!< the id of this keyframe

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

