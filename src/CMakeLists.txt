PROJECT(rba-stereoSLAM)

FIND_PACKAGE(SRBA REQUIRED)
INCLUDE_DIRECTORIES(${SRBA_INCLUDE_DIRS})

FIND_PACKAGE(MRPT REQUIRED ${SRBA_REQUIRED_MRPT_MODULES} obs gui opengl vision hwdrivers graphslam)
FIND_PACKAGE(OpenCV REQUIRED)

SET( VO_LIB_DIR 
		"" CACHE PATH 
		"Visual Odometry Library directory" )
SET( VO_INC_DIR 
		"" CACHE PATH 
		"Visual Odometry Include directory" )

SET( BOW2_DIR 
		"" CACHE PATH 
		"BoW2 path")
		
SET( BOOST_INC_DIR 
		"" CACHE PATH 
		"Boost include dir" )

SET( BOW2_LIB_RELEASE_DIR ${BOW2_DIR}/lib )
#SET( BOW2_LIB_REL_UTILS ${BOW2_LIB_RELEASE_DIR}/libDUtils.lib )
#SET( BOW2_LIB_REL_UTILSCV ${BOW2_LIB_RELEASE_DIR}/libDUtilsCV.lib )
#SET( BOW2_LIB_REL_VISION ${BOW2_LIB_RELEASE_DIR}/libDVision.lib )
#SET( BOW2_LIB_REL_BOW2 ${BOW2_LIB_RELEASE_DIR}/libDBoW2.lib )

SET( BOW2_LIB_DEBUG_DIR ${BOW2_DIR}/lib )
#SET( BOW2_LIB_DEB_UTILS ${BOW2_LIB_DEBUG_DIR}/libDUtils.lib )
#SET( BOW2_LIB_DEB_UTILSCV ${BOW2_LIB_DEBUG_DIR}/libDUtilsCV.lib )
#SET( BOW2_LIB_DEB_VISION ${BOW2_LIB_DEBUG_DIR}/libDVision.lib )
#SET( BOW2_LIB_DEB_BOW2 ${BOW2_LIB_DEBUG_DIR}/libDBoW2.lib )

IF(UNIX)
	SET( VO_LIBS_RELEASE ${VO_LIB_DIR}/libstereo-odometry.so )
	SET( VO_LIBS_DEBUG ${VO_LIB_DIR}/libstereo-odometry.so )
ELSE(UNIX)
	SET( VO_LIBS_RELEASE ${VO_LIB_DIR}/Release/stereo-odometry.lib )
	SET( VO_LIBS_DEBUG ${VO_LIB_DIR}/Debug/stereo-odometry.lib )
ENDIF(UNIX)

INCLUDE_DIRECTORIES( 
		${VO_INC_DIR} 
		${BOOST_INC_DIR}
		${BOW2_DIR}/DBoW2 
		${BOW2_DIR}/DUtils 
		${BOW2_DIR}/DUtilsCV 
		${BOW2_DIR}/DVision )

# Declare the target (an executable)
ADD_EXECUTABLE( rba-stereoSLAM
	CStereoSLAMKF.cpp 			CStereoSLAMKF.h
	rba-stereoSLAM_utils.cpp	rba-stereoSLAM_utils.h
	rba-stereoSLAM-common.h
	rba-stereoSLAM_main.cpp		rba-stereoSLAM.h )

# NOTE: It seems crucial the order of the linking libs!! First, our lib, then MRPT ones
#SET( LINK_LIBRARIES_RELEASE optimized 
#		${VO_LIBS_RELEASE} 
#		${BOW2_LIB_REL_UTILS} 
#		${BOW2_LIB_REL_UTILSCV}
#		${BOW2_LIB_REL_VISION}
#		${BOW2_LIB_REL_BOW2} )

#SET( LINK_LIBRARIES_DEBUG debug 
#		${VO_LIBS_DEBUG} 
#		${BOW2_LIB_DEB_UTILS} 
#		${BOW2_LIB_DEB_UTILSCV}
#		${BOW2_LIB_DEB_VISION}
#		${BOW2_LIB_DEB_BOW2} )
IF(UNIX)
	SET( LINK_LIBRARIES_RELEASE 
		optimized ${VO_LIBS_RELEASE} 
		optimized ${BOW2_LIB_RELEASE_DIR}/libDUtils.so 
		optimized ${BOW2_LIB_RELEASE_DIR}/libDUtilsCV.so 
		optimized ${BOW2_LIB_RELEASE_DIR}/libDVision.so 
		optimized ${BOW2_LIB_RELEASE_DIR}/libDBoW2.so )

	SET( LINK_LIBRARIES_DEBUG debug 
		${VO_LIBS_DEBUG} 
		${BOW2_LIB_DEBUG_DIR}/libDUtils.so 
		${BOW2_LIB_DEBUG_DIR}/libDUtilsCV.so 
		${BOW2_LIB_DEBUG_DIR}/libDVision.so 
		${BOW2_LIB_DEBUG_DIR}/libDBoW2.so )
ELSE(UNIX)
	SET( LINK_LIBRARIES_RELEASE 
		optimized ${VO_LIBS_RELEASE} 
		optimized ${BOW2_LIB_RELEASE_DIR}/Release/libDUtils.lib 
		optimized ${BOW2_LIB_RELEASE_DIR}/Release/libDUtilsCV.lib 
		optimized ${BOW2_LIB_RELEASE_DIR}/Release/libDVision.lib 
		optimized ${BOW2_LIB_RELEASE_DIR}/Release/libDBoW2.lib )

	SET( LINK_LIBRARIES_DEBUG debug 
		${VO_LIBS_DEBUG} 
		${BOW2_LIB_DEBUG_DIR}/Debug/libDUtils.lib 
		${BOW2_LIB_DEBUG_DIR}/Debug/libDUtilsCV.lib 
		${BOW2_LIB_DEBUG_DIR}/Debug/libDVision.lib 
		${BOW2_LIB_DEBUG_DIR}/Debug/libDBoW2.lib )
ENDIF(UNIX)

#TARGET_LINK_LIBRARIES( rba-stereoSLAM debug ${VO_LIBS_DEBUG} ${MRPT_LIBS} ${OpenCV_LIBS} ${BOW2_LIB_DEBUG_DIR}/libDUtils.lib ${BOW2_LIB_DEBUG_DIR}/libDUtilsCV.lib ${BOW2_LIB_DEBUG_DIR}/libDVision.lib ${BOW2_LIB_DEBUG_DIR}/libDBoW2.lib )

#TARGET_LINK_LIBRARIES( rba-stereoSLAM optimized ${VO_LIBS_RELEASE} ${MRPT_LIBS} ${OpenCV_LIBS} ${BOW2_LIB_RELEASE_DIR}/libDUtils.lib ${BOW2_LIB_RELEASE_DIR}/libDUtilsCV.lib ${BOW2_LIB_RELEASE_DIR}/libDVision.lib ${BOW2_LIB_RELEASE_DIR}/libDBoW2.lib)

TARGET_LINK_LIBRARIES( 
	rba-stereoSLAM 
	${LINK_LIBRARIES_RELEASE} 
	${LINK_LIBRARIES_DEBUG}
	${MRPT_LIBS}
	${OpenCV_LIBS} )

# Add user supplied extra options (optimization, etc...)
IF(MSVC)
	# For MSVC to avoid the C1128 error about too large object files:
	SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
	SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj")
ENDIF(MSVC)

# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")

