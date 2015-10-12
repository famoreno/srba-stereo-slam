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

// stereo slam keyframe class
#include "CStereoSLAMKF.h"
#include "rba-stereoSLAM.h"

#define KEEP_MAX(_A,_B) _A = std::max(_A,_B);
#define QUERY_SCORE_TH(_T,_N) _T = _N == UNINITIALIZED_TRACKED_NUMBER ? 0.2 : std::max( 0.15, std::min(0.5, (-0.35/50.0)*(_N-75)+0.15 ) );

inline double updateQueryScoreThreshold( const size_t & numberTrackedFeats )
{
	return UNINITIALIZED_TRACKED_NUMBER ? 0.2 : std::max( 0.15, std::min(0.5, (-0.35/50.0)*(numberTrackedFeats-75)+0.15 ) );
}

enum LCResult { lcr_NO_LC, lcr_FOUND_LC, lcr_BAD_DATA, lcr_NOT_ENOUGH_DATA };

 bool compareKeypointLists( const vector<KeyPoint> & list1, const Mat & desc1, const vector<KeyPoint> & list2, const Mat & desc2 );
 bool compareMatchesLists( const vector<DMatch> & list1, const vector<DMatch> & list2 );
 bool compareOptions( const TStereoSLAMOptions & opt1, const TStereoSLAMOptions & opt2 );
 bool dumpKeyPointsToStream( std::ofstream & stream, const vector<KeyPoint> & keypoints, const Mat & descriptors );
 bool dumpOptionsToStream( std::ofstream & stream, const TStereoSLAMOptions & options );
 bool dumpMatchesToStream( std::ofstream & stream, const vector<DMatch> & matches, const vector<size_t> & matches_ids );

 bool loadOptionsFromStream( std::ifstream & stream, TStereoSLAMOptions & options );
 bool loadKeyPointsFromStream( std::ifstream & stream, vector<KeyPoint> & keypoints, Mat & descriptors );
 bool loadMatchesFromStream( std::ifstream & stream, vector<DMatch> & matches, vector<size_t> & matches_ids );

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
	BriefDatabase					& db );

 bool loadApplicationState( 
	const string 					& filename,						// input file
	size_t							& count,
	int								& last_num_tracked_feats,
	CPose3DRotVec					& current_pose,
	CPose3DRotVec					& last_kf_pose,
	CPose3DRotVec					& incr_pose_from_last_kf,
	CPose3DRotVec					& incr_pose_from_last_check,
	t_vector_kf						& keyframes,
	TStereoSLAMOptions				& stSLAMOpts,
	rso::CStereoOdometryEstimator	& voEngine,
	mySRBA 							& rba,
	TAppOptions						& app_options,
	TStereoSLAMOptions				& stereo_slam_options,
	BriefDatabase					& db );

 void show_kf_numbers( COpenGLScenePtr & scene, const size_t & num_kf, const QueryResults & ret, const double & th = 0.0 );

 LCResult checkLoopClosure( 
	const TKeyFrameID				& new_kf_id,
	const QueryResults				& ret, 
	mySRBA							& rba,
	const TStereoSLAMOptions		& stereo_slam_options, 
	TLoopClosureInfo				& lc_info );

 bool getSimilarKfs( 
	const TKeyFrameID			& newKfId,
	const QueryResults			& dbQueryResults,
	mySRBA						& rba,
	const TStereoSLAMOptions	& stereoSlamOptions,
	TLoopClosureInfo			& out );

 CPose3DRotVec getRelativePose( 
	 const TKeyFrameID		& fromId, 
	 const TKeyFrameID		& toId, 
	 const CPose3DRotVec	& voIncrPose );

 inline TPoint3D projectMatchTo3D(
	 const double			& ul, 
	 const double			& vl, 
	 const double			& ur,
	 const TStereoCamera	& stereoCamera )
 {
	 // camera
	const double & cul		= stereoCamera.leftCamera.cx();
	const double & cvl		= stereoCamera.leftCamera.cy();
	const double & fl		= stereoCamera.leftCamera.fx();
	const double & cur		= stereoCamera.rightCamera.cx();
	const double & fr		= stereoCamera.rightCamera.fx();
	const double & baseline = stereoCamera.rightCameraPose[0];
			
	const double b_d = baseline/(fl*(cur-ur)+fr*(ul-cul));
	return TPoint3D(b_d*fr*(ul-cul),b_d*fr*(vl-cvl),b_d*fl*fr);
 }

 double updateTranslationThreshold( const double x, const double th );
 double updateRotationThreshold( const double x, const double th );
