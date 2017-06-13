/*********************************************************************************
**   				SRBA-Stereo-SLAM    				**
**********************************************************************************
**										**
**	Copyright(c) 2015-2017, Jose Luis Blanco, University of Almeria         **
**	Copyright(c) 2015-2017, Francisco-Angel Moreno 				**
**			MAPIR group, University of Malaga			**
**	All right reserved
**										**
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.				**
**										**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of		**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the		**
**  GNU General Public License for more details.				**
**										**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**										**
**********************************************************************************/
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
