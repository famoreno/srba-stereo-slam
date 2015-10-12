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
#include "CStereoSLAMKF.h"
#include "rba-stereoSLAM-common.h"

#define TRACKING_WITH_F

extern TAppOptions app_options;

// ----------------------------------------------------------
// create
// ----------------------------------------------------------
void CStereoSLAMKF::create( const CObservationStereoImagesPtr & stImgs, const TStereoSLAMOptions & options )
{
    //  :: detect features
    //      -- fills the inner data structures (keypoints and descriptors)
    this->m_detect_features( stImgs, options );

    //  :: match the features
    //      -- checks canonical stereo configuration restrictions
    //      -- fills the inner data structures (matches)
    this->m_match_features( options );
} // end-create

// ----------------------------------------------------------
// performDataAssociation
// ----------------------------------------------------------
void CStereoSLAMKF::performDataAssociation(
							const t_vector_kf					& keyframes,				// INPUT
							const TLoopClosureInfo				& lc_info,					// INPUT
							rso::CStereoOdometryEstimator		& voEngine,					// INPUT
							TVectorKfsDaInfo					& out_da,					// OUTPUT
							const TStereoCamera					& stereo_camera,			// oINPUT
							const TStereoSLAMOptions			& stSLAMOpts,				// oINPUT
							const CPose3DRotVec					& odomPoseFromLastKF )		// oINPUT
{
	out_da.clear();
	out_da.reserve( lc_info.similar_kfs.size() );

	//	:: preliminary : prepare input from THIS KF (this is common for all the KF to test)
	const size_t num_matches = this->m_matches.size();
	cv::Mat curLDesc( num_matches,32, this->m_keyDescLeft.type() );
	cv::Mat curRDesc( num_matches,32, this->m_keyDescRight.type() );

	for(size_t k = 0; k < num_matches; ++k)
	{
		// create matrixes with the proper descriptors: curLDesc and curRDesc
		this->m_keyDescLeft.row( this->m_matches[k].queryIdx ).copyTo( curLDesc.row(k) );
		this->m_keyDescRight.row( this->m_matches[k].trainIdx ).copyTo( curRDesc.row(k) );
	}
	// --------------------------------------------------------------------------

	// check query results and perform data association:
	//	- maximum score should be over an absolute threshold
	//	- non-maximum score should be over an absoulte threshold and over 90% of the maximum one to be considered
	int k = 0;
	for( vector<TKeyFrameID>::const_iterator it = lc_info.similar_kfs.begin(); it != lc_info.similar_kfs.end(); ++it, ++k )
	{
		VERBOSE_LEVEL(1) << "	DA with KF #" << keyframes[*it].m_kfID << endl;

		// insert (if possible) the relative position of this KF wrt the previous one
		// only for the previous one:
		CPose3DRotVec initialPose;
		if( keyframes[*it].m_kfID == this->m_kfID-1 ) // only for the previous one
		{
			initialPose = odomPoseFromLastKF;
			VERBOSE_LEVEL(2) << "		:: Pose estimated by visual odometry: " << initialPose << endl;
		}
		else
		{
			if( !lc_info.similar_kfs_poses[k].empty() )
			{
				CPose3DRotVec aux = stSLAMOpts.camera_pose_on_robot_rvt_inverse + CPose3DRotVec(lc_info.similar_kfs_poses[k]) + stSLAMOpts.camera_pose_on_robot_rvt;
				initialPose = aux+odomPoseFromLastKF;
				VERBOSE_LEVEL(2) << "		:: Pose roughly estimated: " << initialPose << endl;
				VERBOSE_LEVEL(2) << "		:: kfpose: " << aux << " and odom: " << CPose3D(odomPoseFromLastKF) << endl;
			}
			else
			{
				initialPose = CPose3DRotVec();
				VERBOSE_LEVEL(2) << "		:: no initial odometry pose" << endl;
			}
		} // end-else

		rso::CStereoOdometryEstimator::TStereoOdometryResult stOdomResult;
		out_da.push_back( t_kf_da_info() );
		this->internal_performDataAssociation(
									keyframes[*it],			// the other KF
									curLDesc, curRDesc,		// the current descriptors
									*out_da.rbegin(),		// output data association
									voEngine,				// visual odometer (for stage 2)
									stOdomResult,
									stereo_camera,			// stereo camera parameters
									stSLAMOpts,				// stereo slam options
									initialPose );			// initial pose

		// if we already have a higher number of features than the threshold, stop computing da
		if( false && out_da.rbegin()->tracked_matches >= stSLAMOpts.updated_matches_th )
			break;
	}

	if( lc_info.similar_kfs.size() == 0 )	// this shouldn't happen, but just in case
	{
		rso::CStereoOdometryEstimator::TStereoOdometryResult stOdomResult;
		out_da.push_back( t_kf_da_info() );
		// if there isn't any result over the absolute threhold -> perform data association with the last keyframe (even if it is not the closest)
		this->internal_performDataAssociation(
									//*keyframes.rbegin(),			// the other KF
									*(keyframes.rbegin()-1),			// the other KF
									curLDesc, curRDesc,				// the current descriptors
									*out_da.rbegin(),				// output data association
									voEngine,						// visual odometer (for stage 2)
									stOdomResult,
									stereo_camera,					// stereo camera parameters
									stSLAMOpts );					// stereo slam options
	}
} // end-performDataAssociation

// ----------------------------------------------------------
// internal_performDataAssociation
// ----------------------------------------------------------
void CStereoSLAMKF::internal_performDataAssociation(
							const CStereoSLAMKF					& other_kf,					// INPUT  -- The other KF to perform DA with
							const Mat							& curLDesc,					// INPUT
							const Mat							& curRDesc,					// INPUT
							t_kf_da_info						& out_da,					// OUTPUT -- DA information from this KF wrt the other one
							rso::CStereoOdometryEstimator		& voEngine,					// INPUT  -- The visual odometry engine to compute the change in pose
							rso::CStereoOdometryEstimator::TStereoOdometryResult & stOdomResult, 
							const TStereoCamera					& stereo_camera,			// oINPUT -- Stereo camera parameters
							const TStereoSLAMOptions			& stSLAMOpts,				// oINPUT -- Stereo SLAM options
							const CPose3DRotVec					& kf_ini_rel_pose )			// oINPUT -- Initial estimation of the relative pose between this KF and the other one
{
	const size_t this_num_matches  = this->m_matches.size();
	const size_t other_num_matches = other_kf.m_matches.size();

	bool invalid = false;

	//	:: prepare output
	out_da.kf_idx = other_kf.m_kfID;
	out_da.tracked_matches = 0;
	out_da.tracking_info.resize( this_num_matches, make_pair(INVALID_IDX,0.0f) );

	//	:: preliminary : prepare input from the OTHER KF
	cv::Mat preLDesc( other_num_matches, 32, other_kf.m_keyDescLeft.type() );
	cv::Mat preRDesc( other_num_matches, 32, other_kf.m_keyDescRight.type() );
	
	for(size_t k = 0; k < other_num_matches; ++k)
	{
		// create matrixes with the proper descriptors: preLDesc and preRDesc
		other_kf.m_keyDescLeft.row(  other_kf.m_matches[k].queryIdx ).copyTo( preLDesc.row(k) );
		other_kf.m_keyDescRight.row( other_kf.m_matches[k].trainIdx ).copyTo( preRDesc.row(k) );
	}
	
	//	:: create the matcher (bruteforce with Hamming distance)
	BFMatcher matcher( NORM_HAMMING, false );
	vector<DMatch> matL, matR;

	//  :: match between left keypoint descriptors and right keypoint descriptors
	matcher.match( curLDesc /*query*/, preLDesc /*train*/, matL /* size of curLDesc */);
	matcher.match( curRDesc /*query*/, preRDesc /*train*/, matR /* size of curRDesc */);
	
	if( app_options.debug )
	{
		FILE *f = mrpt::system::os::fopen( GENERATE_NAME_WITH_2KF(if_match,other_kf.m_kfID) ,"wt");
		for( vector<DMatch>::iterator it = matL.begin(); it != matL.end(); ++it )
		{
			const size_t idxL = this->m_matches[it->queryIdx].queryIdx;
			const size_t idxR = other_kf.m_matches[it->trainIdx].queryIdx;

			// plu, plv, clu, clv, orb_dist
			mrpt::system::os::fprintf(f,"%.2f %.2f %.2f %.2f %.2f\n", 
				other_kf.m_keyPointsLeft[idxR].pt.x, other_kf.m_keyPointsLeft[idxR].pt.y,
				this->m_keyPointsLeft[idxL].pt.x, this->m_keyPointsLeft[idxL].pt.y,
				it->distance);
		}
		mrpt::system::os::fclose(f);
	}

	//	:: STAGE 1 --> filter out by ORB distance + consistency + uniqueness
	t_vector_pair_idx_distance other_matched( other_num_matches, make_pair(INVALID_IDX, std::numeric_limits<float>::max()) );
	vector<int> this_matched( this_num_matches, INVALID_IDX );
	size_t stage1_counter = 0;
	for( vector<DMatch>::iterator itL = matL.begin(), itR = matR.begin(); itL != matL.end(); ++itL, ++itR )
	{
		if( itL->trainIdx != itR->trainIdx ) continue;																		// consistency test between left and right tracked features
		if( itL->distance > stSLAMOpts.max_orb_distance_da || itR->distance > stSLAMOpts.max_orb_distance_da ) continue;	// orb distance test
		const float mean_distance = 0.5*(itL->distance+itR->distance);
		if( mean_distance > other_matched[itL->trainIdx].second ) continue;
		if( other_matched[itL->trainIdx].first != INVALID_IDX )						// this was already matched but now it's better so ...
		{
			this_matched[ other_matched[itL->trainIdx].first ] = INVALID_IDX;		// ... undo the previous match
			--stage1_counter;
		}
		// we've got a match (or update a previous match)
		other_matched[itL->trainIdx].first  = itL->queryIdx;
		other_matched[itL->trainIdx].second = mean_distance;
		this_matched[itL->queryIdx]			= itL->trainIdx;
		++stage1_counter;
	} // end-for

	VERBOSE_LEVEL(2) << "[iDA " << this->m_kfID << "->" << other_kf.m_kfID << "]: Stage 1 Tracked feats: " << stage1_counter << "/" << matL.size() <<  endl;

	// DEBUG -----------------------------------------------------------
	if( app_options.debug )
	{	//	:: save first stage tracking
		FILE *f1 = os::fopen( GENERATE_NAME_WITH_2KF( cand_stage1_other_tracked, other_kf.m_kfID ), "wt" );
		for( size_t m = 0; m < other_matched.size(); ++m )
			os::fprintf( f1, "%d\n", other_matched[m].first );
		os::fclose(f1);

		FILE *f2 = os::fopen( GENERATE_NAME_WITH_2KF( cand_stage1_this_tracked, other_kf.m_kfID ), "wt" );
		for( size_t m = 0; m < this_matched.size(); ++m )
			os::fprintf( f2, "%d\n", this_matched[m] );
		os::fclose(f2);

		FILE *f3 = os::fopen( GENERATE_NAME_WITH_2KF( stage1_tracked, other_kf.m_kfID ), "wt" );
		os::fprintf( f3, "%% THIS_KF_ID THIS_IDX THIS_ID THIS_UL THIS_VL THIS_UR THIS_VR OTHER_KF_ID OTHER_IDX OTHER_ID OTHER_UL OTHER_VL OTHER_UR OTHER_VR\n" );
		for( size_t m = 0; m < other_matched.size(); ++m )
		{
			if( other_matched[m].first == INVALID_IDX || other_matched[m].first == OUTLIER_ID ) continue;

			// this
			const size_t tm_idx = other_matched[m].first;
			const size_t tm_id  = this->m_matches_ID[tm_idx];
			const cv::KeyPoint & tlkp = this->m_keyPointsLeft[this->m_matches[tm_idx].queryIdx];
			const cv::KeyPoint & trkp = this->m_keyPointsRight[this->m_matches[tm_idx].trainIdx];

			// other
			const size_t om_idx = m;
			const size_t om_id  = other_kf.m_matches_ID[om_idx];
			const cv::KeyPoint & olkp = other_kf.m_keyPointsLeft[other_kf.m_matches[om_idx].queryIdx];
			const cv::KeyPoint & orkp = other_kf.m_keyPointsRight[other_kf.m_matches[om_idx].trainIdx];

			// dist
			const double dist = other_matched[m].second;

			os::fprintf( f3, "%d %d %d %.2f %.2f %.2f %.2f %d %d %d %.2f %.2f %.2f %.2f %.2f\n",
				this->m_kfID, tm_idx, tm_id, tlkp.pt.x, tlkp.pt.y, trkp.pt.x, trkp.pt.y,
				other_kf.m_kfID, om_idx, om_id, olkp.pt.x, olkp.pt.y, orkp.pt.x, orkp.pt.y,
				dist );
		}
		os::fclose(f3);
	}
	// --------------------------------------------------

	size_t outliers_stage2 = 0;
	//	:: STAGE 2 --> either use a fundamental matrix or the minimization residual to remove outliers
	if( stSLAMOpts.da_stage2_method == TStereoSLAMOptions::ST2M_FUNDMATRIX ||
		stSLAMOpts.da_stage2_method == TStereoSLAMOptions::ST2M_BOTH )
	{
		if( stage1_counter < 15 )
		{
			VERBOSE_LEVEL(2) << "[iDA " << this->m_kfID << "->" << other_kf.m_kfID << "]: Stage 2 (F) Not enough input data." << endl;
			invalid = true;
		}
		else
		{
			//	:: using fundamental matrix
			vector<size_t> outliers;

			//	:: detect inliers with fundamental matrix
			m_detect_outliers_with_F(
				other_matched,
				*this,
				other_kf,
				outliers,
				stSLAMOpts );

			FILE *f = NULL;
			if( app_options.debug ) f = os::fopen( GENERATE_NAME_WITH_2KF( fundmat_outliers, other_kf.m_kfID ), "wt" );
			// remove outliers from the fundamental matrix
			// delete from 'other_matched'
			for( size_t k = 0; k < outliers.size(); ++k )
			{
				other_matched[outliers[k]].first = INVALID_IDX;
				if(f) os::fprintf(f,"%d\n",outliers[k]);
			}
			if(f) os::fclose(f);

			outliers_stage2 = outliers.size();

			VERBOSE_LEVEL(2) << "[iDA " << this->m_kfID << "->" << other_kf.m_kfID << "]: Stage 2 (F) Outliers detected: " << outliers_stage2 << endl;
		}
	}

	if( stSLAMOpts.da_stage2_method == TStereoSLAMOptions::ST2M_CHANGEPOSE ||
		stSLAMOpts.da_stage2_method == TStereoSLAMOptions::ST2M_BOTH )
	{
		ASSERT_( stage1_counter >= outliers_stage2 )

		if( stage1_counter - outliers_stage2 < 15 )
		{
			VERBOSE_LEVEL(2) << "[iDA " << this->m_kfID << "->" << other_kf.m_kfID << "]: Stage 2 (CP) Not enough input data." << endl;
			invalid = true;
		}
		else
		{
			//	:: using residual of minimization
			vector<size_t> outliers;

			//	:: detect outliers using the residual of the change in pose optimization process
			m_detect_outliers_with_change_in_pose(
				other_matched,
				*this,
				other_kf,
				voEngine,
				outliers,
				stOdomResult,
				stereo_camera,
				stSLAMOpts,
				kf_ini_rel_pose );

			FILE *f = NULL;
			if( app_options.debug ) f = os::fopen( GENERATE_NAME_WITH_2KF( changepose_outliers, other_kf.m_kfID ), "wt" );
			// remove outliers from the fundamental matrix
			// delete from 'other_matched'
			for( size_t k = 0; k < outliers.size(); ++k )
			{
				other_matched[outliers[k]].first = INVALID_IDX;
				if(f) os::fprintf(f,"%d\n",outliers[k]);
			}
			if(f) os::fclose(f);

			VERBOSE_LEVEL(2) << "[iDA " << this->m_kfID << "->" << other_kf.m_kfID << "]: Stage 2 (CP) Outliers detected: " << outliers.size() << endl;

			outliers_stage2 += outliers.size();
		}
	}

	// DEBUG ------------------------------------------------
	FILE *f2 = NULL;
	if( app_options.debug )
		f2 = mrpt::system::os::fopen( GENERATE_NAME_WITH_2KF(if_match_after, other_kf.m_kfID) ,"wt");
	// ------------------------------------------------------

	if( !invalid )
	{
		//	:: Create output for this KF
		for( size_t k = 0; k < other_matched.size(); ++k )
		{
			if( other_matched[k].first != INVALID_IDX && other_matched[k].first != OUTLIER_ID )
			{
				out_da.tracking_info[other_matched[k].first /*this_idx*/] = make_pair(int(k) /*other_idx*/, other_matched[k].second /*distance*/);
				++out_da.tracked_matches;

				// DEBUG ------------------------------------------------
				if( app_options.debug )
				{
					const size_t idxL = this->m_matches[other_matched[k].first].queryIdx;
					const size_t idxR = other_kf.m_matches[int(k)].queryIdx;

					const double lu = this->m_keyPointsLeft[idxL].pt.x;
					const double lv = this->m_keyPointsLeft[idxL].pt.y;
					const double ru = other_kf.m_keyPointsLeft[idxR].pt.x;
					const double rv = other_kf.m_keyPointsLeft[idxR].pt.y;

					mrpt::system::os::fprintf(f2,"%.2f %.2f %.2f %.2f %.2f %.2f\n", lu, lv, ru, rv, other_matched[k].second, stOdomResult.out_residual[k]);
				}
				// ------------------------------------------------------
			} // end-if
		} // end-for
	} // end-if

	VERBOSE_LEVEL(2) << "[iDA " << this->m_kfID << "->" << other_kf.m_kfID << "]: Total tracked feats: " << out_da.tracked_matches << "/" << matL.size() << endl;

	// DEBUG ------------------------------------------------
	if( app_options.debug )
		mrpt::system::os::fclose(f2);
	// ------------------------------------------------------

} // end-internal_performDataAssociation

void CStereoSLAMKF::m_detect_outliers_with_change_in_pose (
				t_vector_pair_idx_distance			& other_matched,
				const CStereoSLAMKF					& this_kf,
				const CStereoSLAMKF					& other_kf,
				rso::CStereoOdometryEstimator		& voEngine,
				vector<size_t>						& outliers /*output*/,
				rso::CStereoOdometryEstimator::TStereoOdometryResult & result, /*output*/
				const TStereoCamera					& stereo_camera,
				const TStereoSLAMOptions			& stSLAMOpts,
				const CPose3DRotVec					& kf_ini_rel_pose ) /*vector<double>*/
{
	// prepare output
	outliers.clear(); outliers.reserve( other_matched.size() );

	// create a vector with the tracked pairs so far for the visual odometer
	rso::vector_index_pairs_t tracked_pairs;
	tracked_pairs.reserve( this->m_matches.size() );
	for( size_t k = 0; k < other_matched.size(); ++k )
	{
		if( other_matched[k].first != INVALID_IDX )
			tracked_pairs.push_back( make_pair( k /*other_idx*/, size_t(other_matched[k].first) /*this_idx*/) );
	}
	size_t num_tracked_stage1 = tracked_pairs.size();

	// 'getChangeInPose' accepts an initial value for the change in pose between frames.
	// The initial estimation is the a vector form of the pose of the previous frame wrt the current one
	// Here we have the pose of the current frame wrt the previous one, so we need to invert it
	CPose3DRotVec inverseOdomPose = kf_ini_rel_pose.getInverse();

	vector<double> initialPoseVector(6); // [w1,w2,w3,tx,ty,tz]
	initialPoseVector[0] = inverseOdomPose.m_rotvec[0];	initialPoseVector[1] = inverseOdomPose.m_rotvec[1];	initialPoseVector[2] = inverseOdomPose.m_rotvec[2];
	initialPoseVector[3] = inverseOdomPose.m_coords[0];	initialPoseVector[4] = inverseOdomPose.m_coords[1];	initialPoseVector[5] = inverseOdomPose.m_coords[2];

	voEngine.params_least_squares.use_custom_initial_pose = true;
	const bool valid = voEngine.getChangeInPose(
			tracked_pairs,																						// the tracked pairs
			other_kf.m_matches, this->m_matches,																// pre_matches, cur_matches,
			other_kf.m_keyPointsLeft, other_kf.m_keyPointsRight, this->m_keyPointsLeft, this->m_keyPointsRight, // pre_left_feats, pre_right_feats, cur_left_feats, cur_right_feats,
			stereo_camera,
			result,					// output
			initialPoseVector );	// [w1,w2,w3,tx,ty,tz]
	voEngine.params_least_squares.use_custom_initial_pose = false;

	VERBOSE_LEVEL(2) << "Iterations: " << result.num_it << " (initial) and " << result.num_it_final << " (final) " << endl;
	VERBOSE_LEVEL(2) << "Change in pose: " << result.outPose << endl;
	// I already have the outliers (got from the optimizer's first stage) but we may want adjust the threshold here

	if( !result.valid )
	{
		VERBOSE_LEVEL(1) << "	WARNING: Change in pose could not be estimated, skipping this test." << endl;
		if( app_options.debug ) 
		{
			// empty file
			FILE *f = os::fopen( GENERATE_NAME_WITH_KF( posechange_outliers ), "wt" );
			os::fclose(f);
		}
	}

	// remove large outliers
	FILE *f = NULL;
	if( app_options.debug ) f = os::fopen( GENERATE_NAME_WITH_KF( posechange_outliers ), "wt" );
	for( size_t k = 0; k < result.out_residual.size(); ++k )
	{
		if( result.out_residual[k] > stSLAMOpts.residual_th )
		{
			// only set outliers if the change in pose had a valid solution
			if( result.valid )			
				outliers.push_back( tracked_pairs[k].first );

			// in any case, save the "id" and the residual of the outlier for debug purposes
			if(f) os::fprintf( f, "%d %.2f\n", tracked_pairs[k].first, result.out_residual[k] );
		}
	} // end-for

	if(f) os::fclose(f);

	// re-adjust 'results.out_residual' size to match other_matched size
	vector<double> out_residual( other_matched.size(), std::numeric_limits<double>::max() );
	size_t cnt = 0;
	for( size_t k = 0; k < other_matched.size(); ++k )
	{
		if( other_matched[k].first != INVALID_IDX )
			out_residual[k] = result.out_residual[cnt++];
	}
	result.out_residual.swap(out_residual);
} // end -- m_detect_outliers_with_change_in_pose

// ----------------------------------------------------------
// m_detect_outliers_with_F
// ----------------------------------------------------------
void CStereoSLAMKF::m_detect_outliers_with_F (
				const t_vector_pair_idx_distance	& other_matched,
				const CStereoSLAMKF					& this_kf,
				const CStereoSLAMKF					& other_kf,
				vector<size_t>						& outliers /*output*/,
				const TStereoSLAMOptions			& stSLAMOpts )
{
	// prepare output
	outliers.clear(); outliers.reserve( other_matched.size() );

	vector<size_t> tracked_idx; tracked_idx.reserve( other_matched.size() );
	for( size_t k = 0;  k < other_matched.size(); ++k )
	{
		if( other_matched[k].first != INVALID_IDX )
			tracked_idx.push_back(k);
	}
	const size_t num_tracked = tracked_idx.size();

	cv::Mat ppl(num_tracked,2,cv::DataType<float>::type),ppr(num_tracked,2,cv::DataType<float>::type);
	cv::Mat pcl(num_tracked,2,cv::DataType<float>::type),pcr(num_tracked,2,cv::DataType<float>::type);
	for( size_t k = 0; k < tracked_idx.size(); ++k )
	{
		const size_t idx		= tracked_idx[k];
		const size_t preIdxL	= other_kf.m_matches[idx].queryIdx;
		const size_t preIdxR	= other_kf.m_matches[idx].trainIdx;
		const size_t curIdxL	= this_kf.m_matches[other_matched[idx].first].queryIdx;
		const size_t curIdxR	= this_kf.m_matches[other_matched[idx].first].trainIdx;

        ppl.at<float>(k,0) = static_cast<float>(other_kf.m_keyPointsLeft[preIdxL].pt.x);
        ppl.at<float>(k,1) = static_cast<float>(other_kf.m_keyPointsLeft[preIdxL].pt.y);

		ppr.at<float>(k,0) = static_cast<float>(other_kf.m_keyPointsRight[preIdxR].pt.x);
        ppr.at<float>(k,1) = static_cast<float>(other_kf.m_keyPointsRight[preIdxR].pt.y);

		pcl.at<float>(k,0) = static_cast<float>(this_kf.m_keyPointsLeft[curIdxL].pt.x);
        pcl.at<float>(k,1) = static_cast<float>(this_kf.m_keyPointsLeft[curIdxL].pt.y);

		pcr.at<float>(k,0) = static_cast<float>(this_kf.m_keyPointsRight[curIdxR].pt.x);
        pcr.at<float>(k,1) = static_cast<float>(this_kf.m_keyPointsRight[curIdxR].pt.y);
	}

	vector<uchar> left_inliers, right_inliers;
	cv::findFundamentalMat(ppl,pcl,cv::FM_RANSAC,stSLAMOpts.max_y_diff_epipolar,stSLAMOpts.ransac_fit_prob,left_inliers);
	cv::findFundamentalMat(ppr,pcr,cv::FM_RANSAC,stSLAMOpts.max_y_diff_epipolar,stSLAMOpts.ransac_fit_prob,right_inliers);
	for( size_t k = 0; k < left_inliers.size(); ++k )
	{
		if( left_inliers[k] == 0 || right_inliers[k] == 0 )
			outliers.push_back(tracked_idx[k]);
	}
} // end-m_detect_outliers_with_F

// ----------------------------------------------------------
// dumpToConsole
// ----------------------------------------------------------
void CStereoSLAMKF::dumpToConsole()
{
    cout << "KEYFRAME [" << this->m_kfID << "]" << endl
         << "---------------------------------------------" << endl
         << "   :: Camera pose = " << this->m_camPose << endl
         << "   :: Matches [" << this->m_matches.size() << " out of "
         << this->m_keyPointsLeft.size() << "/"
         << this->m_keyPointsRight.size() << "]: ID: left_kp_x,left_kp_y --> right_kp_x,right_kp_y" << endl
         << "   -------------------------------------"
         << endl;

    for( size_t k = 0; k < m_matches.size(); ++k )
    {
        const size_t id1 = m_matches[k].queryIdx;
        const size_t id2 = m_matches[k].trainIdx;

        cout << "   "
             << m_matches_ID[k] << ": "
             << m_keyPointsLeft[id1].pt.x << ","
             << m_keyPointsLeft[id1].pt.y << " --> "
             << m_keyPointsRight[id2].pt.x << ","
             << m_keyPointsRight[id2].pt.y
             << endl;
    } // end-for
    cout << "++++++++++++++++++++++++++++++++++++++++++++++" << endl;
} // end dumpToConsole

// ----------------------------------------------------------
// saveInfoToFiles
// ----------------------------------------------------------
void CStereoSLAMKF::saveInfoToFiles( const string & str_modif )
{
	// prepare output directory
	if( !mrpt::system::directoryExists( app_options.out_dir ) )
		mrpt::system::createDirectory( app_options.out_dir );

	// information
	string my_filename;
	if( str_modif.empty() )
		my_filename = mrpt::format("%s\\info_kf%04d.txt", app_options.out_dir.c_str(), this->m_kfID);
	else
		my_filename = mrpt::format("%s\\%s_info_kf%04d.txt", app_options.out_dir.c_str(), str_modif.c_str(), this->m_kfID);

	FILE *f = mrpt::system::os::fopen( my_filename, "wt");
	if( !f )
		THROW_EXCEPTION( mrpt::format("Output file %s could not be opened", my_filename.c_str()) );

	mrpt::system::os::fprintf(f, "%% [KF_ID] [MATCH_ID] [LEFT_PT{x y}] [RIGHT_PT{x y}] [MATCH_DISTANCE]\n");
	size_t m_count = 0;
	vector<cv::DMatch>::iterator it;
	const size_t n_matches_id_size = this->m_matches_ID.size();
	for( it = this->m_matches.begin(); it != this->m_matches.end(); ++it, ++m_count )
	{
		const cv::KeyPoint & lkp = this->m_keyPointsLeft[it->queryIdx];
		const cv::KeyPoint & rkp = this->m_keyPointsRight[it->trainIdx];
		mrpt::system::os::fprintf( f,"%d %d %.2f %.2f %.2f %.2f %.2f\n",
			this->m_kfID,
			n_matches_id_size > 0 ? this->m_matches_ID[m_count] : 0,
			lkp.pt.x, lkp.pt.y,
			rkp.pt.x, rkp.pt.y,
			it->distance );
	} // end-for
	mrpt::system::os::fclose(f);

	if( str_modif.empty() )
		my_filename = mrpt::format("%s\\info_feats_kf%04d.txt", app_options.out_dir.c_str(), this->m_kfID);
	else
		my_filename = mrpt::format("%s\\%s_info_feats_kf%04d.txt", app_options.out_dir.c_str(), str_modif.c_str(), this->m_kfID);

	f = mrpt::system::os::fopen( my_filename, "wt");
	if( !f )
		THROW_EXCEPTION( mrpt::format("Output file %s could not be opened", my_filename.c_str()) );

	mrpt::system::os::fprintf( f, "%d %d\n", this->m_keyPointsLeft.size(), this->m_keyPointsRight.size() );
	for( vector<cv::KeyPoint>::iterator it = this->m_keyPointsLeft.begin(); it != this->m_keyPointsLeft.end(); ++it )
		mrpt::system::os::fprintf( f, "%.2f %.2f\n", it->pt.x, it->pt.y );
	for( vector<cv::KeyPoint>::iterator it = this->m_keyPointsRight.begin(); it != this->m_keyPointsRight.end(); ++it )
		mrpt::system::os::fprintf( f, "%.2f %.2f\n", it->pt.x, it->pt.y );

	mrpt::system::os::fclose(f);
} // end saveInfoToFiles

// ----------------------------------------------------------
// m_change_structure_binary
// ----------------------------------------------------------
void CStereoSLAMKF::m_change_structure_binary( const Mat & plain, vector<BRIEF::bitset> & out )
{
	out.resize( plain.rows );
	for( unsigned int i = 0; i < static_cast<unsigned int>(plain.rows); i ++ )
	{
		// for each descriptor
		out[i].resize( plain.cols*8 );						// number of bits (256)
		for( unsigned int k = 0; k < static_cast<unsigned int>(plain.cols); ++k )
		{
			const uint8_t val = plain.at<uint8_t>(i,k);
			for(unsigned int m = 0; m < 8; ++m)
				out[i][m+k*8] = (val >> m) & 1;
		}
	}
} // end-changeStructureORB

// ----------------------------------------------------------
// m_detect_features
// ----------------------------------------------------------
void CStereoSLAMKF::m_detect_features( const CObservationStereoImagesPtr & stImgs, const TStereoSLAMOptions & stSLAMOpts )
{
    //  :: detect ORB features --> consider multi-thread
    Mat cvInputLeftImage( stImgs->imageLeft.getAs<IplImage>() );
    Mat cvInputRightImage( stImgs->imageRight.getAs<IplImage>() );

	const size_t n_feats_to_detect = stSLAMOpts.non_maximal_suppression ? 4*stSLAMOpts.n_feats : stSLAMOpts.n_feats;
	ORB orb_detector( n_feats_to_detect, 1.2, stSLAMOpts.n_levels,
			31 /*edgeThreshold*/,
			0 /*firstLevel*/,
			2 /*WTA_K*/,
			ORB::HARRIS_SCORE /*scoreType*/,
			31 /*patchSize*/,
			stSLAMOpts.detect_fast_th );

	// Other approach: FAST + BRIEF (to be considered)
	Mat auxML, auxMR;
	vector<KeyPoint> kpL, kpR;
	if( stSLAMOpts.detect_method == TStereoSLAMOptions::DM_FAST_ORB )
	{
		// opencv's fast detector + orb descriptor
		// cons: no multiscale
		// pros: more control on the number of features with 'stSLAMOpts.detect_fast_th' param
		cv::FastFeatureDetector fast( stSLAMOpts.detect_fast_th );

		fast.detect( cvInputLeftImage, kpL );
		fast.detect( cvInputRightImage, kpR );

		orb_detector( cvInputLeftImage, Mat(), kpL, auxML, true );
		orb_detector( cvInputRightImage, Mat(), kpR, auxMR, true );
	}
	else if( stSLAMOpts.detect_method == TStereoSLAMOptions::DM_ORB_ONLY )
	{
		// opencv's orb detector:
		// cons: fixed FAST threshold of 20 --> there is some control on the number of features detected
		// pros: multiscale FAST detection
		orb_detector( cvInputLeftImage, Mat(), kpL, auxML );
		orb_detector( cvInputRightImage, Mat(), kpR, auxMR );
	}

	if( stSLAMOpts.non_maximal_suppression )
	{
		if( app_options.debug )
		{
			cout << "initial points = " << kpL.size() << endl;
			// save to file before
			FILE *f1 = fopen("before_non_max_supp.txt","wt");
			for( size_t k = 0; k < kpL.size(); ++k )
				fprintf(f1, "%.2f %.2f\n", kpL[k].pt.x, kpL[k].pt.y );
			fclose(f1);
		}

		m_adaptive_non_max_suppression( stSLAMOpts.n_feats, kpL, auxML, this->m_keyPointsLeft, this->m_keyDescLeft );
		m_adaptive_non_max_suppression( stSLAMOpts.n_feats, kpR, auxMR, this->m_keyPointsRight, this->m_keyDescRight );

		if( app_options.debug )
		{
			FILE *f2 = fopen("after_non_max_supp.txt","wt");
			for( size_t k = 0; k < this->m_keyPointsLeft.size(); ++k )
				fprintf(f2, "%.2f %.2f\n", this->m_keyPointsLeft[k].pt.x, this->m_keyPointsLeft[k].pt.y );
			fclose(f2);
		}
	}
	else
	{
		// just copy to inner structure
		kpL.swap(this->m_keyPointsLeft);
		kpR.swap(this->m_keyPointsRight);

		// this should be fast!
		this->m_keyDescLeft = auxML;
		this->m_keyDescRight = auxMR;
	}
} // end -- m_detect_features

// ----------------------------------------------------------
// m_match_features
// ----------------------------------------------------------
void CStereoSLAMKF::m_match_features( const TStereoSLAMOptions & stSLAMOpts )
{
	BFMatcher orb_matcher( NORM_HAMMING, false );

	orb_matcher.match( this->m_keyDescLeft, this->m_keyDescRight, this->m_matches );

	// to do: remove the not matched descriptors
	if( stSLAMOpts.matching_options.enable_robust_1to1_match )
	{
		// for each right feature: 'distance' and 'left idx'
		const size_t right_size = this->m_keyPointsRight.size();
		vector< pair< double, size_t > >  right_cand( right_size, make_pair(-1.0,0) );

		// loop over the matches
		for( size_t k = 0; k < this->m_matches.size(); ++k )
		{
			const size_t idR = this->m_matches[k].trainIdx;
			if( right_cand[idR].first < 0 || right_cand[idR].first > this->m_matches[k].distance )
			{
				right_cand[idR].first  = this->m_matches[k].distance;
				right_cand[idR].second = this->m_matches[k].queryIdx;
			}
		} // end-for

		vector<cv::DMatch>::iterator itMatch;
		for( itMatch = this->m_matches.begin(); itMatch != this->m_matches.end();  )
		{
			if( itMatch->queryIdx != right_cand[ itMatch->trainIdx ].second )
				itMatch = this->m_matches.erase( itMatch );
			else
				++itMatch;
		} // end-for
	} // end-1-to-1 matchings

    //  :: filter out by using epipolar geometry (parallel axis configuration)
    //  :: keep only those that fulfill the epipolar constraint & the ORB distance is below a threshold
	if( stSLAMOpts.matching_options.useEpipolarRestriction && !stSLAMOpts.matching_options.parallelOpticalAxis )
		cout << "WARNING: This method only considers stereo rigs with parallel optical axis. This warning shows because: 'options.parallelOpticalAxis' is false. Set it to true if it's your case. If it's not, don't use this method." << endl;

    vector<DMatch>::iterator itM = this->m_matches.begin();
    while(itM != this->m_matches.end())
    {
        const int diff = this->m_keyPointsLeft[itM->queryIdx].pt.y-this->m_keyPointsRight[itM->trainIdx].pt.y;
		const int disp = this->m_keyPointsLeft[itM->queryIdx].pt.x-this->m_keyPointsRight[itM->trainIdx].pt.x;

		const bool c1 = stSLAMOpts.matching_options.useEpipolarRestriction ? std::abs(diff) <= stSLAMOpts.matching_options.epipolar_TH : true;
		const bool c2 = stSLAMOpts.matching_options.useXRestriction ? disp > 0 : true;
		const bool c3 = stSLAMOpts.matching_options.useDisparityLimits ? disp >= stSLAMOpts.matching_options.min_disp && disp <= stSLAMOpts.matching_options.max_disp : true;

		if( !c1 || !c2 || !c3 || itM->distance > stSLAMOpts.matching_options.maxORB_dist )
		{
			itM = this->m_matches.erase(itM);
		}
		else
		{
			this->m_keyPointsLeft[itM->queryIdx].pt.y = this->m_keyPointsRight[itM->trainIdx].pt.y = .5f*(this->m_keyPointsLeft[itM->queryIdx].pt.y+this->m_keyPointsRight[itM->trainIdx].pt.y);
			++itM;
		}
    } // end-while

} // end -- m_match_features

// ----------------------------------------------------------
// m_adaptive_non_max_suppression
// ----------------------------------------------------------
void CStereoSLAMKF::m_adaptive_non_max_suppression(
					const size_t				& num_out_points,
					const vector<KeyPoint>		& keypoints,
					const Mat					& descriptors,
					vector<KeyPoint>			& out_kp_rad,
					Mat							& out_kp_desc,
					const double				& min_radius_th )
{
	const double CROB = 0.9;
	const size_t N = keypoints.size();
	out_kp_rad.clear();
	out_kp_rad.reserve( N );

	// First: order them by response
	vector<size_t> sorted_indices( N );
	for (size_t i=0;i<N;i++)  sorted_indices[i]=i;
	std::sort( sorted_indices.begin(), sorted_indices.end(), KpResponseSorter(keypoints) );

	// insert the global maximum
	const cv::KeyPoint & strongest_kp = keypoints[sorted_indices[0]];

	// create the radius vector
	vector<double> radius( N );
	radius[sorted_indices[0]] = std::numeric_limits<double>::infinity();	// set the radius for this keypoint to infinity

	// for the rest of the keypoints:
	double min_ri, this_ri;
	for( size_t k1 = 1; k1 < N; ++k1 )
	{
		const cv::KeyPoint & kp1 = keypoints[sorted_indices[k1]];

		// the min_ri is at most the distance to the strongest keypoint
		min_ri = std::fabs( (kp1.pt.x-strongest_kp.pt.x)*(kp1.pt.x-strongest_kp.pt.x)+(kp1.pt.y-strongest_kp.pt.y)*(kp1.pt.y-strongest_kp.pt.y) );; // distance to the strongest

		// compute the ri value for all the previous keypoints
		for( int k2 = k1-1; k2 > 0; --k2 )
		{
			const cv::KeyPoint & kp2 = keypoints[sorted_indices[k2]];

			if( kp1.response < CROB*kp2.response )
			{
				this_ri = std::fabs((kp1.pt.x-kp2.pt.x)*(kp1.pt.x-kp2.pt.x)+(kp1.pt.y-kp2.pt.y)*(kp1.pt.y-kp2.pt.y));
				if( this_ri < min_ri ) min_ri = this_ri;
			}
		} // end-for-k2
		radius[sorted_indices[k1]] = min_ri;
	} // end-for-k1

	// sort again according to the radius
	const double min_radius_th_2 = min_radius_th*min_radius_th;
	for( size_t i = 0; i < N; i++ ) sorted_indices[i] = i;
	std::sort( sorted_indices.begin(), sorted_indices.end(), KpRadiusSorter(radius) );
	for( size_t i = 0; i < num_out_points; i++ )
	{
		if( radius[sorted_indices[i]] > min_radius_th_2 )
		{
			out_kp_rad.push_back( keypoints[sorted_indices[i]] );
			out_kp_desc.push_back( descriptors.row( sorted_indices[i] ) );
		}
	}
} // end-m_adaptive_non_max_suppression
