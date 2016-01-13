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

extern TGeneralOptions general_options;

// ----------------------------------------------------------
// dumpToConsole
// ----------------------------------------------------------
void CStereoSLAMKF::dumpToConsole()
{
    cout << "KEYFRAME [" << this->m_kf_ID << "]" << endl
         << "---------------------------------------------" << endl
         << "   :: Camera pose = " << this->m_camera_pose << endl
         << "   :: Matches [" << this->m_matches.size() << " out of "
         << this->m_keypoints_left.size() << "/"
         << this->m_keypoints_right.size() << "]: ID: left_kp_x,left_kp_y --> right_kp_x,right_kp_y" << endl
         << "   -------------------------------------"
         << endl;

    for( size_t k = 0; k < m_matches.size(); ++k )
    {
        const size_t id1 = m_matches[k].queryIdx;
        const size_t id2 = m_matches[k].trainIdx;

        cout << "   "
             << m_matches_ID[k] << ": "
             << m_keypoints_left[id1].pt.x << ","
             << m_keypoints_left[id1].pt.y << " --> "
             << m_keypoints_right[id2].pt.x << ","
             << m_keypoints_right[id2].pt.y
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
	if( !mrpt::system::directoryExists( general_options.out_dir ) )
		mrpt::system::createDirectory( general_options.out_dir );

	// information
	string my_filename;
	if( str_modif.empty() )
		my_filename = mrpt::format("%s\\info_kf%04d.txt", general_options.out_dir.c_str(), this->m_kf_ID);
	else
		my_filename = mrpt::format("%s\\%s_info_kf%04d.txt", general_options.out_dir.c_str(), str_modif.c_str(), this->m_kf_ID);

	FILE *f = mrpt::system::os::fopen( my_filename, "wt");
	if( !f )
		THROW_EXCEPTION( mrpt::format("Output file %s could not be opened", my_filename.c_str()) );

	mrpt::system::os::fprintf(f, "%% [KF_ID] [MATCH_ID] [LEFT_PT{x y}] [RIGHT_PT{x y}] [MATCH_DISTANCE]\n");
	size_t m_count = 0;
	vector<cv::DMatch>::iterator it;
	const size_t n_matches_id_size = this->m_matches_ID.size();
	for( it = this->m_matches.begin(); it != this->m_matches.end(); ++it, ++m_count )
	{
		const cv::KeyPoint & lkp = this->m_keypoints_left[it->queryIdx];
		const cv::KeyPoint & rkp = this->m_keypoints_right[it->trainIdx];
		mrpt::system::os::fprintf( f,"%d %d %.2f %.2f %.2f %.2f %.2f\n",
			this->m_kf_ID,
			n_matches_id_size > 0 ? this->m_matches_ID[m_count] : 0,
			lkp.pt.x, lkp.pt.y,
			rkp.pt.x, rkp.pt.y,
			it->distance );
	} // end-for
	mrpt::system::os::fclose(f);

	if( str_modif.empty() )
		my_filename = mrpt::format("%s\\info_feats_kf%04d.txt", general_options.out_dir.c_str(), this->m_kf_ID);
	else
		my_filename = mrpt::format("%s\\%s_info_feats_kf%04d.txt", general_options.out_dir.c_str(), str_modif.c_str(), this->m_kf_ID);

	f = mrpt::system::os::fopen( my_filename, "wt");
	if( !f )
		THROW_EXCEPTION( mrpt::format("Output file %s could not be opened", my_filename.c_str()) );

	mrpt::system::os::fprintf( f, "%d %d\n", this->m_keypoints_left.size(), this->m_keypoints_right.size() );
	for( vector<cv::KeyPoint>::iterator it = this->m_keypoints_left.begin(); it != this->m_keypoints_left.end(); ++it )
		mrpt::system::os::fprintf( f, "%.2f %.2f\n", it->pt.x, it->pt.y );
	for( vector<cv::KeyPoint>::iterator it = this->m_keypoints_right.begin(); it != this->m_keypoints_right.end(); ++it )
		mrpt::system::os::fprintf( f, "%.2f %.2f\n", it->pt.x, it->pt.y );

	mrpt::system::os::fclose(f);
} // end saveInfoToFiles
