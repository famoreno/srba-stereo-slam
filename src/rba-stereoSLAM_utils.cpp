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
   | Redistribution and use in source and binary forms, with or without
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
#include "rba-stereoSLAM_utils.h"
#include "rba-stereoSLAM-common.h"

extern TAppOptions app_options;

// these are static methods (only available when the header file is included)

void computeDispersion( const vector<KeyPoint> & list, const vector<DMatch> & matches, double std_x, double std_y )
{
	double mx = 0, my = 0;
	for( vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it )
	{
		mx += list[it->queryIdx].pt.x;
		my += list[it->queryIdx].pt.y;
	}
	mx /= matches.size();
	my /= matches.size();

	for( vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it )
	{
		std_x += square(list[it->queryIdx].pt.x-mx);
		std_y += square(list[it->queryIdx].pt.y-my);
	}
	std_x = sqrt(std_x);
	std_y = sqrt(std_y);
}

// ---------------------------------------------------
// comparison (auxiliary methods)
// ---------------------------------------------------
 bool compareKeypointLists( const vector<KeyPoint> & list1, const Mat & desc1, const vector<KeyPoint> & list2, const Mat & desc2 )
{
	if( list1.size() != list2.size() )
		return false;

	if( desc1.size() != desc2.size() )
		return false;

	// keyp
	vector<KeyPoint>::const_iterator it1, it2;
	for( it1 = list1.begin(), it2 = list2.begin(); it1 != list1.end(); ++it1, ++it2 )
	{
		if( it1->pt.x != it2->pt.x || it1->pt.y != it2->pt.y || it1->response != it2->response || it1->angle != it2->angle ||
			it1->class_id != it2->class_id || it1->octave != it2->octave || it1->size != it2->size )
				return false;
	}

	// desc
	MatConstIterator_<uchar> itd1, itd2;
	for( itd1 = desc1.begin<uchar>(), itd2 = desc2.begin<uchar>(); itd1 != desc1.end<uchar>(); ++itd1, ++itd2 ) // stream << *it;
	{
		if( *itd1 != *itd2 )
			return false;
	}

	return true;
}

 bool compareMatchesLists( const vector<DMatch> & list1, const vector<DMatch> & list2 )
{
	if( list1.size() != list2.size() )
		return false;

	vector<DMatch>::const_iterator it1, it2;
	for( it1 = list1.begin(), it2 = list2.begin(); it1 != list1.end(); ++it1, ++it2 )
	{
		if( it1->queryIdx != it2->queryIdx || it1->trainIdx != it2->trainIdx || it1->distance != it2->distance || it1->imgIdx != it2->imgIdx )
			return false;
	}

	return true;
}

 bool compareOptions( const TStereoSLAMOptions & opt1, const TStereoSLAMOptions & opt2 )
{
	return 	opt1.n_levels == opt2.n_levels && opt1.n_feats == opt2.n_feats && opt1.min_ORB_distance == opt2.min_ORB_distance &&
			opt1.matching_options == opt2.matching_options &&
			opt1.max_y_diff_epipolar == opt2.max_y_diff_epipolar &&
			opt1.max_orb_distance_da == opt2.max_orb_distance_da &&
			opt1.max_distance_keyframes == opt2.max_distance_keyframes && opt1.ransac_fit_prob == opt2.ransac_fit_prob &&
			opt1.max_translation == opt2.max_translation && opt1.max_rotation == opt2.max_rotation &&
			opt1.residual_th == opt2.residual_th && opt1.non_maximal_suppression == opt2.non_maximal_suppression &&
			opt1.updated_matches_th == opt2.updated_matches_th && opt1.up_matches_th_plus == opt2.up_matches_th_plus,
			opt1.detect_method == opt2.detect_method && opt1.detect_fast_th == opt2.detect_fast_th,
			opt1.non_max_supp_method == opt2.non_max_supp_method;
}

// ---------------------------------------------------
// dumping methods
// ---------------------------------------------------
 bool dumpKeyPointsToStream( /*CFileStream*/ std::ofstream & stream, const vector<KeyPoint> & keypoints, const Mat & descriptors )
{
	/* FORMAT
	- # of features in image
		- feat x coord
		- feat y coord
		- feat response
		- feat size
		- feat angle
		- feat octave
		- feat class_id
	- # of dimensions of descriptors (D): rows, cols and type
		feat descriptor d_0 ... d_{D-1}
	*/

	if( !stream.is_open() )
		return false;

	size_t num_kp = keypoints.size();
	stream.write( reinterpret_cast<char*>(&num_kp), sizeof(size_t));

	for( size_t f = 0; f < keypoints.size(); ++f )
	{
		stream.write( (char*)(&(keypoints[f].pt.x)), sizeof(float) );
		stream.write( (char*)(&(keypoints[f].pt.y)), sizeof(float) );
		stream.write( (char*)(&(keypoints[f].response)), sizeof(float) );
		stream.write( (char*)(&(keypoints[f].size)), sizeof(float) );
		stream.write( (char*)(&(keypoints[f].angle)), sizeof(float) );
		stream.write( (char*)(&(keypoints[f].octave)), sizeof(int) );
		stream.write( (char*)(&(keypoints[f].class_id)), sizeof(int) );
	} // end-for-keypoints
	int drows = descriptors.rows, dcols = descriptors.cols, dtype = descriptors.type();
	stream.write( (char*)&drows, sizeof(int) );
	stream.write( (char*)&dcols, sizeof(int) );
	stream.write( (char*)&dtype, sizeof(int) );

	for( MatConstIterator_<uchar> it = descriptors.begin<uchar>(); it != descriptors.end<uchar>(); ++it ) // stream << *it;
	{
		uchar value = *it;
		stream.write( (char*)&value, sizeof(uchar) );
	}

	return true;
} // end-dumpKeyPointsToFile

 bool dumpOptionsToStream( std::ofstream & stream, const TStereoSLAMOptions & options )
{
	if( !stream.is_open() )
		return false;

	stream.write( (char*)&(options.n_levels), sizeof(options.n_levels) );
	stream.write( (char*)&(options.n_feats), sizeof(options.n_feats) );
	stream.write( (char*)&(options.min_ORB_distance), sizeof(options.min_ORB_distance) );
	stream.write( (char*)&(options.detect_method), sizeof(options.detect_method) );
	stream.write( (char*)&(options.detect_fast_th), sizeof(options.detect_fast_th) );
	stream.write( (char*)&(options.non_max_supp_method), sizeof(options.non_max_supp_method) );
	stream.write( (char*)&(options.matching_options.epipolar_TH), sizeof(options.matching_options.epipolar_TH) );
	stream.write( (char*)&(options.max_y_diff_epipolar), sizeof(options.max_y_diff_epipolar) );
	stream.write( (char*)&(options.matching_options.maxORB_dist), sizeof(options.matching_options.maxORB_dist) );
	stream.write( (char*)&(options.max_orb_distance_da), sizeof(options.max_orb_distance_da) );
	stream.write( (char*)&(options.max_distance_keyframes), sizeof(options.max_distance_keyframes) );
	stream.write( (char*)&(options.ransac_fit_prob), sizeof(options.ransac_fit_prob) );
	stream.write( (char*)&(options.matching_options.max_disp), sizeof(options.matching_options.max_disp) );
	stream.write( (char*)&(options.matching_options.min_disp), sizeof(options.matching_options.min_disp) );
	stream.write( (char*)&(options.matching_options.enable_robust_1to1_match), sizeof(options.matching_options.enable_robust_1to1_match) );
	stream.write( (char*)&(options.residual_th), sizeof(options.residual_th) );
	stream.write( (char*)&(options.max_translation), sizeof(options.max_translation) );
	stream.write( (char*)&(options.max_rotation), sizeof(options.max_rotation) );
	stream.write( (char*)&(options.non_maximal_suppression), sizeof(options.non_maximal_suppression) );
	stream.write( (char*)&(options.updated_matches_th), sizeof(options.updated_matches_th) );
	stream.write( (char*)&(options.up_matches_th_plus), sizeof(options.up_matches_th_plus) );

	return true;
} // end-dumpOptionsToStream

 bool dumpMatchesToStream( std::ofstream & stream, const vector<DMatch> & matches, const vector<size_t> & matches_ids )
{
	/* FORMAT
	- # of matches
		- match id
		- queryIdx
		- trainIdx
		- distance
	*/

	if( !stream.is_open() )
		return false;

	size_t num_m = matches.size();
	stream.write( (char*)&num_m, sizeof(size_t) );
	for( size_t m = 0; m < matches.size(); ++m )
	{
		stream.write( (char*)&(matches_ids[m]), sizeof(size_t) );
		stream.write( (char*)&(matches[m].queryIdx), sizeof(matches[m].queryIdx) );
		stream.write( (char*)&(matches[m].trainIdx), sizeof(matches[m].trainIdx) );
		stream.write( (char*)&(matches[m].distance), sizeof(matches[m].distance) );
		stream.write( (char*)&(matches[m].imgIdx), sizeof(matches[m].imgIdx) );
	} // end-for-matches

	return true;
} // end-dumpMatchesToStream

// ---------------------------------------------------
// save application state
// ---------------------------------------------------
 bool saveApplicationState(
	const string 					& filename,						// output file
	const size_t					& count,
	const int						& last_num_tracked_feats,
	const CPose3DRotVec				& current_pose,
	const CPose3DRotVec				& last_kf_pose,
	const CPose3DRotVec				& incr_pose_from_last_kf,
	const CPose3DRotVec				& incr_pose_from_last_check,
	const t_vector_kf				& keyframes,
	const TStereoSLAMOptions		& stSLAMOpts,
	rso::CStereoOdometryEstimator	& voEngine,
	TAppOptions						& app_options,
	BriefDatabase					& db )
{
	/*	FORMAT:
		- starting iteration
		- last match id
		- map of reference ids
		- important poses: current_pose, incr_pose??, last_kf_pose, incr_pose_from_last_kf
		- # of kfs
		- options*
		- kf ID
		- kf camera pose (zeros by now)
		- left features*
		- right features*
		- matches*
	*/
	string n_filename = mrpt::format("%s\\%s", app_options.out_dir.c_str(), filename.c_str() );
	std::ofstream state_file_stream( n_filename.c_str(), ios::out | ios::binary );			// write
	if( !state_file_stream.is_open() )
		return false;

	// global parameters
	//	-- iteration number
	state_file_stream.write( (char*)&count, sizeof(size_t) );

	//	-- last match ID
	state_file_stream.write( reinterpret_cast<char*>(&CStereoSLAMKF::m_last_match_ID), sizeof(size_t) );

	//	-- important poses
	DUMP_ROTVEC_TO_STREAM( state_file_stream, current_pose )
	DUMP_ROTVEC_TO_STREAM( state_file_stream, last_kf_pose )
	DUMP_ROTVEC_TO_STREAM( state_file_stream, incr_pose_from_last_kf )
	DUMP_ROTVEC_TO_STREAM( state_file_stream, incr_pose_from_last_check )

	//	-- options
	if( !dumpOptionsToStream( state_file_stream, stSLAMOpts ) )	// save options (only for the first kf, they are all the same)
	{
		cout << "ERROR while saving the state -- options could not be saved. Closed stream?" << endl;
		return false;
	}

	//	-- last number of tracked features
	state_file_stream.write( (char*)(&last_num_tracked_feats), sizeof(int) );

	//	-- kfs
	size_t num_kf = keyframes.size();
	state_file_stream.write( reinterpret_cast<char*>(&num_kf), sizeof(size_t) );											// number of kfs

	for( size_t k = 0; k < keyframes.size(); ++k )
	{
		const CStereoSLAMKF & kf = keyframes[k];									// shortcut

		state_file_stream.write( (char*)&(kf.m_kfID), sizeof(size_t) );

		DUMP_ROTVEC_TO_STREAM( state_file_stream, kf.m_camPose )

		// kf features
		if( !dumpKeyPointsToStream( state_file_stream, kf.m_keyPointsLeft, kf.m_keyDescLeft ) )
		{
			cout << "ERROR while saving the state -- left keypoints could not be saved. Closed stream?" << endl;
			return false;
		}
		if( !dumpKeyPointsToStream( state_file_stream, kf.m_keyPointsRight, kf.m_keyDescRight ) )
		{
			cout << "ERROR while saving the state -- right keypoints could not be saved. Closed stream?" << endl;
			return false;
		}

		// kf matches
		if( !dumpMatchesToStream( state_file_stream, kf.m_matches, kf.m_matches_ID ) )
		{
			cout << "ERROR while saving the state -- matches could not be saved. Closed stream?" << endl;
			return false;
		}
	} // end-for-keyframes

	// save vodometry information (use a different stream)
	int lastindex = n_filename.find_last_of(".");
	string vo_filename = n_filename.substr(0, lastindex)+"_vo."+n_filename.substr(lastindex+1);

	if( !voEngine.saveStateToFile( vo_filename ) )
	{
		cout << "ERROR while saving the state -- vodometry could not be saved. Closed stream?" << endl;
		return false;
	}

	// global parameters
	state_file_stream.close();

	// save bag of words database
	db.save( mrpt::format("%s\\db_saved.gz", app_options.out_dir.c_str() ) );

	return true;
} // end-method

// ---------------------------------------------------
// load methods
// ---------------------------------------------------
 bool loadOptionsFromStream( std::ifstream & stream, TStereoSLAMOptions & options )
{
	if( !stream.is_open() )
		return false;

	stream.read( (char*)&(options.n_levels), sizeof(options.n_levels) );
	stream.read( (char*)&(options.n_feats), sizeof(options.n_feats) );
	stream.read( (char*)&(options.min_ORB_distance), sizeof(options.min_ORB_distance) );
	stream.read( (char*)&(options.detect_method), sizeof(options.detect_method) );
	stream.read( (char*)&(options.detect_fast_th), sizeof(options.detect_fast_th) );
	stream.read( (char*)&(options.non_max_supp_method), sizeof(options.non_max_supp_method) );
	stream.read( (char*)&(options.matching_options.epipolar_TH), sizeof(options.matching_options.epipolar_TH) );
	stream.read( (char*)&(options.max_y_diff_epipolar), sizeof(options.max_y_diff_epipolar) );
	stream.read( (char*)&(options.matching_options.maxORB_dist), sizeof(options.matching_options.maxORB_dist) );
	stream.read( (char*)&(options.max_orb_distance_da), sizeof(options.max_orb_distance_da) );
	stream.read( (char*)&(options.max_distance_keyframes), sizeof(options.max_distance_keyframes) );
	stream.read( (char*)&(options.ransac_fit_prob), sizeof(options.ransac_fit_prob) );
	stream.read( (char*)&(options.matching_options.max_disp), sizeof(options.matching_options.max_disp) );
	stream.read( (char*)&(options.matching_options.min_disp), sizeof(options.matching_options.min_disp) );
	stream.read( (char*)&(options.matching_options.enable_robust_1to1_match), sizeof(options.matching_options.enable_robust_1to1_match) );
	stream.read( (char*)&(options.residual_th), sizeof(options.residual_th) );
	stream.read( (char*)&(options.max_translation), sizeof(options.max_translation) );
	stream.read( (char*)&(options.max_rotation), sizeof(options.max_rotation) );
	stream.read( (char*)&(options.non_maximal_suppression), sizeof(options.non_maximal_suppression) );
	stream.read( (char*)&(options.updated_matches_th), sizeof(options.updated_matches_th) );
	stream.read( (char*)&(options.up_matches_th_plus), sizeof(options.up_matches_th_plus) );

	return true;
} // end-loadOptionsFromStream

 bool loadKeyPointsFromStream( std::ifstream & stream, vector<KeyPoint> & keypoints, Mat & descriptors )
{
	/* FORMAT
	- # of features in image
	- # of dimensions of descriptors (D)
		- feat x coord
		- feat y coord
		- feat response
		- feat scale
		- feat orientation
		- feat descriptor d_0 ... d_{D-1}
	*/
	if( !stream.is_open() )
		return false;

	size_t num_kp;
	stream.read( (char*)&num_kp, sizeof(size_t) );
	keypoints.resize( num_kp );

	for( size_t f = 0; f < keypoints.size(); ++f )
	{
		stream.read( (char*)(&(keypoints[f].pt.x)), sizeof(float) );
		stream.read( (char*)(&(keypoints[f].pt.y)), sizeof(float) );
		stream.read( (char*)(&(keypoints[f].response)), sizeof(float) );
		stream.read( (char*)(&(keypoints[f].size)), sizeof(float) );
		stream.read( (char*)(&(keypoints[f].angle)), sizeof(float) );
		stream.read( (char*)(&(keypoints[f].octave)), sizeof(int) );
		stream.read( (char*)(&(keypoints[f].class_id)), sizeof(int) );

	} // end-for-keypoints
	int drows,dcols,dtype;
	stream.read( (char*)&drows, sizeof(int) );
	stream.read( (char*)&dcols, sizeof(int) );
	stream.read( (char*)&dtype, sizeof(int) );
	descriptors.create(drows,dcols,dtype);

	for( MatIterator_<uchar> it = descriptors.begin<uchar>(); it != descriptors.end<uchar>(); ++it ) // stream << *it;
	{
		uchar value;
		stream.read( (char*)&value, sizeof(uchar) );
		*it = value;
	}

	return true;
} // end-loadKeyPointsFromStream

 bool loadMatchesFromStream( std::ifstream & stream, vector<DMatch> & matches, vector<size_t> & matches_ids )
{
	/* FORMAT
	- # of matches
		- match id
		- queryIdx
		- trainIdx
		- distance
	*/
	if( !stream.is_open() )
		return false;

	size_t num_matches;
	stream.read( (char*)&num_matches, sizeof(size_t) );
	matches.resize( num_matches );
	matches_ids.resize( num_matches );
	for( size_t m = 0; m < matches.size(); ++m )
	{
		stream.read( (char*)&(matches_ids[m]), sizeof(size_t) );
		stream.read( (char*)&(matches[m].queryIdx), sizeof(matches[m].queryIdx) );
		stream.read( (char*)&(matches[m].trainIdx), sizeof(matches[m].trainIdx) );
		stream.read( (char*)&(matches[m].distance), sizeof(matches[m].distance) );
		stream.read( (char*)&(matches[m].imgIdx), sizeof(matches[m].imgIdx) );
	} // end-for-matches

	return true;
} // end-loadMatchesFromStream

// ---------------------------------------------------
// load application state
// ---------------------------------------------------
 bool loadApplicationState(
	const string 					& filename,					// input file
	size_t							& count,
	int								& last_num_tracked_feats,
	CPose3DRotVec					& current_pose,
	CPose3DRotVec					& last_kf_pose,
	CPose3DRotVec					& incr_pose_from_last_kf,
	CPose3DRotVec					& incr_pose_from_last_check,
	t_vector_kf						& keyframes,
	TStereoSLAMOptions				& stSLAMOpts,
	rso::CStereoOdometryEstimator	& voEngine,
	mySRBA /*myRBAEngine*/			& rba,
	TAppOptions						& app_options,
	TStereoSLAMOptions				& stereo_slam_options,
	BriefDatabase					& db )
{
	string n_filename = mrpt::format("%s\\%s", app_options.out_dir.c_str(), filename.c_str() );
	std::ifstream state_file_stream( n_filename.c_str(), ios::in | ios::binary );	// read
	if( !state_file_stream.is_open() )
		return false;

	// global parameters
	//	-- iteration number
	state_file_stream.read( (char*)&count, sizeof(size_t) );

	//	-- last match ID
	state_file_stream.read(reinterpret_cast<char*>(&CStereoSLAMKF::m_last_match_ID),sizeof(size_t)); // >> aux;							// matches last id

	//	-- important poses
	LOAD_ROTVEC_FROM_STREAM( state_file_stream, current_pose )
	LOAD_ROTVEC_FROM_STREAM( state_file_stream, last_kf_pose )
	LOAD_ROTVEC_FROM_STREAM( state_file_stream, incr_pose_from_last_kf )
	LOAD_ROTVEC_FROM_STREAM( state_file_stream, incr_pose_from_last_check )

	if(!loadOptionsFromStream( state_file_stream, stSLAMOpts ))
	{
		cout << "ERROR while loading the state -- options could not be loaded. Closed stream?" << endl;
		return false;
	}

	// last number tracked feats
	state_file_stream.read( (char*)&last_num_tracked_feats, sizeof(int) );

	// kfs
	size_t num_kfs;
	state_file_stream.read(reinterpret_cast<char*>(&num_kfs),sizeof(size_t));
	keyframes.resize( num_kfs );

	VERBOSE_LEVEL(1) << "	" << num_kfs << " keyframes to load: " << endl;

	TStereoSLAMOptions opt;
	for( size_t k = 0; k < num_kfs; ++k )
	{
		CStereoSLAMKF & kf = keyframes[k];

		// kf ID
		size_t kf_id;
		state_file_stream.read( (char*)&kf_id, sizeof(size_t) );
		kf.setKFID( kf_id );

		LOAD_ROTVEC_FROM_STREAM( state_file_stream, kf.m_camPose )

		// kf features
		if( !loadKeyPointsFromStream( state_file_stream, kf.m_keyPointsLeft, kf.m_keyDescLeft ) )
		{
			cout << "ERROR while loading the state -- left keypoints could not be loaded. Closed stream?" << endl;
			return false;
		}
		if( !loadKeyPointsFromStream( state_file_stream, kf.m_keyPointsRight, kf.m_keyDescRight ) )
		{
			cout << "ERROR while loading the state -- right keypoints could not be loaded. Closed stream?" << endl;
			return false;
		}

		// kf matches
		if( !loadMatchesFromStream( state_file_stream, kf.m_matches, kf.m_matches_ID ) )
		{
			cout << "ERROR while loading the state -- matches could not be loaded. Closed stream?" << endl;
			return false;
		}

		// create rba data
		//  :: insert KF data into SRBA engine
		mySRBA::TNewKeyFrameInfo       newKFInfo;
        mySRBA::new_kf_observations_t  listObs;
        mySRBA::new_kf_observation_t   obsField;

		obsField.is_fixed                   = false;    // landmarks have unknown relative positions (i.e. treat them as unknowns to be estimated)
        obsField.is_unknown_with_init_val   = true;     // we don't have any guess on the initial LM position (will invoke the inverse sensor model)

        const size_t num_matches = kf.m_matches.size();
		listObs.resize( num_matches );

		//  :: fill observation fields
		for( size_t m = 0; m < num_matches; ++m )
		{
			const size_t id1 				= kf.m_matches[m].queryIdx;
			const size_t id2 				= kf.m_matches[m].trainIdx;

			const KeyPoint & kpLeft  		= kf.m_keyPointsLeft[id1];
			const KeyPoint & kpRight 		= kf.m_keyPointsRight[id2];

			obsField.obs.feat_id            = kf.m_matches_ID[m];
			obsField.obs.obs_data.left_px   = TPixelCoordf( kpLeft.pt.x,  kpLeft.pt.y );
			obsField.obs.obs_data.right_px  = TPixelCoordf( kpRight.pt.x, kpRight.pt.y );
			obsField.setRelPos( kf.projectMatchTo3D( m, stereo_slam_options ) );

			listObs[m] = obsField;
		} // end for

		const bool optimize_this = k != 0;

		VERBOSE_LEVEL(1) << "	CREATE NEW KEYFRAME #" << k << endl;
		VERBOSE_LEVEL(1) << "--------------------------------" << endl;
		//  :: insert into the rba-slam framework
		rba.define_new_keyframe( listObs,				// list of observations
								 newKFInfo,				// keyframe info
								 optimize_this );		// not optimize the first time

		VERBOSE_LEVEL(1) << endl;

	} // end-for-kfs

	// frame for vo
	int lastindex = n_filename.find_last_of(".");
	string vo_filename = n_filename.substr(0, lastindex)+"_vo."+n_filename.substr(lastindex+1);

	if( !voEngine.loadStateFromFile( vo_filename ) )
	{
		cout << "ERROR while saving the state -- vodometry could not be saved. Closed stream?" << endl;
		return false;
	}
	state_file_stream.close();

	// load bag of words database
	db.load( mrpt::format("%s\\db_saved.gz", app_options.out_dir.c_str() ) );

	return true;
}

// ---------------------------------------------------
// show kf information
// ---------------------------------------------------
 void show_kf_numbers( COpenGLScenePtr & scene, const size_t & num_kf, const QueryResults & ret, const double & th )
{
	CRenderizablePtr obj;
	COpenGLViewportPtr vp = scene->getViewport("keyframes");
	for( size_t k = 0; k < ret.size(); ++k )
	{
		obj = vp->getByName( mrpt::format("ret%d_score",k) );
		if( obj )
		{
			CTextPtr score_txt = static_cast<CTextPtr>(obj);
			score_txt->setString( mrpt::format("%.3f",ret[k].Score) );
			score_txt->setVisibility();
		}

		obj = vp->getByName( mrpt::format("ret%d_id",k) );
		if( obj )
		{
			CTextPtr id_txt = static_cast<CTextPtr>(obj);
			id_txt->setString( mrpt::format("%d",ret[k].Id) );
			id_txt->setVisibility();
		}

		obj = vp->getByName( mrpt::format("ret%d_box",k) );
		if( obj )
		{
			CBoxPtr box = static_cast<CBoxPtr>(obj);
			box->setVisibility();
			box->setBoxCorners(TPoint3D(0.5*k,0,0.0),TPoint3D(0.5*k+0.25,ret[k].Score,0.0));
			box->setColor(TColorf(1-3*ret[k].Score,3*ret[k].Score,0));
		}
	}

	obj = vp->getByName( "th_line" );
	if( obj )
	{
		CSimpleLinePtr line = static_cast<CSimpleLinePtr>(obj);
		line->setLineCoords(-0.1,th,0,-0.15+0.5*ret.size(),th,0);
	}

	obj = vp->getByName( "th_value" );
	if( obj )
	{
		CTextPtr txt = static_cast<CTextPtr>(obj);
		txt->setString( mrpt::format("%.2f",th) );
		txt->setPose( CPoint3D( -0.25+0.5*ret.size(), th+0.1, 0.0 ) );
	}
} // end-show_kf_numbers


double updateTranslationThreshold( const double x, const double th )
{
	double newTh = 0.02 + (0.25/th)*x;
	newTh = newTh < 0.02 ? 0.02 : newTh;
	newTh = newTh > 0.3 ? 0.3 : newTh;
	return newTh;
} // end -- updateTranslationThreshold

double updateRotationThreshold( const double x, const double th )
{
	double newTh = 15 + 13/th*(x-th);
	newTh = newTh < 2 ? 2 : newTh;
	newTh = newTh > 15 ? 15 : newTh;
	return newTh;
} // end -- updateTranslationThreshold

/*------------------------------------------------------------
  Checks the results of a DB query and search for potential
  loop closures, returning true if one is found. It also
  returns the IDs of the most similar keyframes.
 -------------------------------------------------------------*/
bool getSimilarKfs(
	const TKeyFrameID			& newKfId,
	const QueryResults			& dbQueryResults,
	mySRBA						& rba,
	const TStereoSLAMOptions	& stereoSlamOptions,
	TLoopClosureInfo			& out )
{
	if( app_options.verbose_level >= 2 )
		cout << "dbQueryResults: " << dbQueryResults << endl;

	const size_t qSize = dbQueryResults.size();
	if( qSize == 0 )
		THROW_EXCEPTION( "Parameter 'dbQueryResults' contains no results. This method should not be called here." );

	if( qSize == 1 )
	{
		out.similar_kfs.push_back( newKfId-1 );
		return false;
	}

	if( dbQueryResults[0].Score < 0.04 /* TODO: absoluteDbQueryThreshold */ )
	{
		SHOW_WARNING( "Best result in 'dbQueryResults' is below a threshold. Lost camera?" );
	}

	// prepare output
	out.similar_kfs.clear();
	out.similar_kfs.reserve( qSize+1 );
	out.lc_id = INVALID_KF_ID;
	bool foundLoopClosure = false;

	// always insert last kf as a similar one
	out.similar_kfs.push_back( newKfId-1 );

	// we've got enough good data, let's find the loop closure
	mySRBA::rba_problem_state_t & myRbaState = rba.get_rba_state();

	// we've got a LC if in the list there is any far KF with a score large enough
	// if last inserted kf is a base, then use it, if not, use the previous one
	const TKeyFrameID fromIdBase =
		rba.isKFLocalmapCenter( newKfId-1 ) ?
		newKfId-1 :
		rba.getLocalmapCenterID( newKfId-1 );						// get id of the last localmap center

	mySRBA::rba_problem_state_t::TSpanningTree::next_edge_maps_t::const_iterator itFrom =
		myRbaState.spanning_tree.sym.next_edge.find( fromIdBase );	// get spanning tree for the current localmap center

	// check the results
	const double loopClosureTh = 0.8*dbQueryResults[0].Score;
	for( size_t i = 0; i < dbQueryResults.size(); ++i )
	{
		const TKeyFrameID toId = dbQueryResults[i].Id;

		if( toId == newKfId-1 )	// already inserted
			continue;

		// compute topologic distance
		topo_dist_t topoDistance = numeric_limits<topo_dist_t>::max();

		if( fromIdBase == toId )
			topoDistance = 0;
		else
		{
			if( itFrom != myRbaState.spanning_tree.sym.next_edge.end() )
			{
				map<TKeyFrameID,TSpanTreeEntry>::const_iterator itToDist = itFrom->second.find( toId );

				if( itToDist != itFrom->second.end() )
					topoDistance = itToDist->second.distance;
			}
			else
			{
				// *** This shouldn't never happen ***
				THROW_EXCEPTION("[ERROR :: Check Loop Closure] 'it_from' is not into the spanning_tree!");
			}
		}
		bool insertKf = false;
		if( topoDistance > stereoSlamOptions.lc_distance )
		{
			// only set the lc with the first KF found
			if( dbQueryResults[i].Score > 0.05 && out.lc_id == INVALID_KF_ID)
			{
				out.lc_id = toId;
				foundLoopClosure = true;
				insertKf = true;
				VERBOSE_LEVEL(1) << "		FOUND POTENTIAL LOOP CLOSURE " << endl;
			}
		}
		else
		{
			if( dbQueryResults[i].Score > loopClosureTh )
				insertKf = true;
		}
		if( insertKf )
		{
			//	:: set this KF as similar
			out.similar_kfs.push_back( toId );
		}

		VERBOSE_LEVEL(2) << "		Distance from " << fromIdBase
						 << " to " << toId << ":" << topoDistance << endl;
	} // end-for

	// ***** for all similar KFs, get a rough estimation of THIS pose wrt to them

	// prepare similar poses output
	out.similar_kfs_poses.resize( out.similar_kfs.size() );

	// search along the spantree for the poses:
	mySRBA::frameid2pose_map_t  spantree;
	rba.create_complete_spanning_tree(newKfId-1, spantree, rba.parameters.srba.max_tree_depth );
	for( size_t k = 0; k < out.similar_kfs.size(); ++k )
	{
		mySRBA::frameid2pose_map_t::const_iterator itP = spantree.find( out.similar_kfs[k] );
		if( itP == spantree.end() )
			out.similar_kfs_poses[k] = CPose3D();
		else
		{
			out.similar_kfs_poses[k] = itP->second.pose;
			out.similar_kfs_poses[k].inverse();
		}
	}

	// DEBUG ------------------------------------------
	if( app_options.verbose_level >= 2 )
	{
		DUMP_VECTORLIKE( out.similar_kfs )
	}
	// ------------------------------------------------

	return foundLoopClosure;
} // end -- getSimilarKfs

// ----------------------------------------------------------
// checks if there is a loop closure (according to the query database) and the RBA state
// returns true if in 'ret' there is a topologically FAR keyframe strong enough (more than 80% of the best result)
// ----------------------------------------------------------
LCResult checkLoopClosure(
		const TKeyFrameID			& new_kf_id,
		const QueryResults			& ret,
		mySRBA						& rba,
		const TStereoSLAMOptions	& stereo_slam_options,
		TLoopClosureInfo					& lc_info )
{
	// preliminary checks
	if( ret.size() < 4 )
	{
		lc_info.similar_kfs.resize(1);
		lc_info.similar_kfs[0] = new_kf_id-1;
		return lcr_NOT_ENOUGH_DATA;				// at least 4 results, return just the last one
	}

	if( ret[0].Score < 0.04 )
	{
		lc_info.similar_kfs.resize(1);
		lc_info.similar_kfs[0] = new_kf_id-1;
		return lcr_BAD_DATA;					// none of them is over the minimal threshold -- lost camera?
	}

	/**/
	if( ret[0].Score < 0.10 )
	{
		lc_info.similar_kfs.resize(1);
		lc_info.similar_kfs[0] = new_kf_id-1;
		return lcr_NO_LC;						// none of them is over the minimal threshold -- lost camera?
	}
	/**/

	// prepare output
	lc_info.similar_kfs.clear();
	lc_info.similar_kfs.reserve( ret.size()+1 );

	// we've got enough good data, let's find the loop closure
	mySRBA::rba_problem_state_t & my_rba_state = rba.get_rba_state();

	// we've got a LC if in the list there is any far KF with a score large enough
	// if last inserted kf is a base, then use it, if not, use the previous one
	const TKeyFrameID from_id_base =
		rba.isKFLocalmapCenter( new_kf_id-1 ) ?
		new_kf_id-1 :
		rba.getLocalmapCenterID( new_kf_id-1 );							// get id of the last localmap center

	mySRBA::rba_problem_state_t::TSpanningTree::next_edge_maps_t::const_iterator it_from = my_rba_state.spanning_tree.sym.next_edge.find(from_id_base); // get spanning tree for the current localmap center

	// -------------------------------------------------------
	bool found_lc = false;
	const double threshold = 0.8*ret[0].Score;
	for( size_t k = 0; k < ret.size(); ++k )
	{
		if( ret[k].Score < threshold ) break;			// at least the highest one will pass this filter

		// get topographic distance between keyframes
		const TKeyFrameID to_id = ret[k].Id;

		topo_dist_t	found_distance = numeric_limits<topo_dist_t>::max();

		if( it_from != my_rba_state.spanning_tree.sym.next_edge.end() )
		{
			map<TKeyFrameID,TSpanTreeEntry>::const_iterator it_to_dist = it_from->second.find(to_id);
			if (it_to_dist != it_from->second.end())
				found_distance = it_to_dist->second.distance;
		}
		else
		{
			// *** This shouldn't never happen ***
			cout << "	[ERROR :: Check Loop Closure] 'it_from' is not into the spanning_tree!" << endl;
		}

		// increment distance due to the edge from current KF to last localmap center
		if( found_distance < numeric_limits<topo_dist_t>::max() )
			found_distance++;

		lc_info.similar_kfs.push_back(to_id);
		if( found_distance > stereo_slam_options.lc_distance )
		{
			found_lc = true;
			lc_info.lc_id = ret[k].Id;
		}

		// debug ---------------------------------
		cout << "	[DEBUG :: Check Loop Closure] Topologic dist. from: " << new_kf_id << " to " << to_id << " is: " << found_distance << endl;
		// -------------------------------------------------------

	} // end-for

	/**/
	// if last KF is not inserted --> insert it
	if( lc_info.similar_kfs.end() == std::find(lc_info.similar_kfs.begin(), lc_info.similar_kfs.end(), new_kf_id-1) )
		lc_info.similar_kfs.push_back( new_kf_id-1 );
	/**/
	return found_lc ? lcr_FOUND_LC : lcr_NO_LC;
} // end-checkLoopClosure
