#pragma once

// -- include
// opencv
#include <cv.h>
#include <highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

// mrpt
#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/opengl/CGridPlaneXY.h>
#include <mrpt/opengl/CText.h>
#include <mrpt/opengl/CSimpleLine.h>

#include <mrpt/vision/types.h>
#include <mrpt/vision/CFeatureExtraction.h>

#include <mrpt/hwdrivers/CCameraSensor.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/obs/CObservationStereoImages.h>
#include <mrpt/poses/CPose3DRotVec.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/utils/CFileStream.h>

// visual odometry
#include <libstereo-odometry.h>

// srba
#include <srba.h>

// -- define
#define ENTER_LOGGER( _STR ) if( general_options.enableLogger ) tLog.enter( _STR );
#define LEAVE_LOGGER( _STR ) if( general_options.enableLogger ) tLog.leave( _STR );
#define MY_SQUARE(_X) (_X)*(_X)
#define DUMP_ROTVEC_TO_STREAM( _STREAM, _ROTVEC ) \
		_STREAM.write( (char*)&(_ROTVEC.m_coords[0]), sizeof(double) );\
		_STREAM.write( (char*)&(_ROTVEC.m_coords[1]), sizeof(double) );\
		_STREAM.write( (char*)&(_ROTVEC.m_coords[2]), sizeof(double) );\
		_STREAM.write( (char*)&(_ROTVEC.m_rotvec[0]), sizeof(double) );\
		_STREAM.write( (char*)&(_ROTVEC.m_rotvec[1]), sizeof(double) );\
		_STREAM.write( (char*)&(_ROTVEC.m_rotvec[2]), sizeof(double) );

#define LOAD_ROTVEC_FROM_STREAM( _STREAM, _ROTVEC ) \
		_STREAM.read( (char*)&(_ROTVEC.m_coords[0]), sizeof(double) );\
		_STREAM.read( (char*)&(_ROTVEC.m_coords[1]), sizeof(double) );\
		_STREAM.read( (char*)&(_ROTVEC.m_coords[2]), sizeof(double) );\
		_STREAM.read( (char*)&(_ROTVEC.m_rotvec[0]), sizeof(double) );\
		_STREAM.read( (char*)&(_ROTVEC.m_rotvec[1]), sizeof(double) );\
		_STREAM.read( (char*)&(_ROTVEC.m_rotvec[2]), sizeof(double) );

#define DUMP_VECTORLIKE(_v) \
	if( _v.size() > 0 ) { \
	for(size_t k = 0; k < _v.size()-1; ++k) \
	cout << #_v << "[" << k << "] = " << _v[k] << ", "; \
	cout << #_v << "[" << (_v.size()-1) << "] = " << *(_v.rbegin()) << endl; }
	
#define UNINITIALIZED_TRACKED_NUMBER -1
#define GENERATE_NAME_WITH_KF(STR) mrpt::format("%s\\%s_kf%04d.txt", general_options.out_dir.c_str(), #STR, this->m_kf_ID)
#define GENERATE_NAME_WITH_2KF(STR,OKF_ID) mrpt::format("%s\\%s_kf%04d_with_kf%04d.txt", general_options.out_dir.c_str(), #STR, this->m_kf_ID, OKF_ID)
#define GENERATE_NAME_WITH_KF_OUT(STR,KF) mrpt::format("%s\\%s_kf%04d.txt", general_options.out_dir.c_str(), #STR, KF.m_kf_ID)
#define GENERATE_NAME_WITH_2KF_OUT(_STR,_ID1,_ID2) mrpt::format("%s\\%s_kf%04d_with_kf%04d.txt", general_options.out_dir.c_str(), #_STR, _ID1, _ID2)
#define DUMP_BOOL_VAR_TO_CONSOLE(_MSG,_VAR) cout << _MSG; _VAR ? cout << "Yes " : cout << "No "; cout << endl;
#define VERBOSE_LEVEL(_LEV) if( general_options.verbose_level >= _LEV ) std::cout// System KFs
#define INVALID_KF_ID -1
#define INVALID_IDX -1											// to do: take these two lines to the header file
#define OUTLIER_ID -2

// -- typedef

// -- namespace
using namespace mrpt;
using namespace mrpt::opengl;
using namespace cv;
using namespace srba;