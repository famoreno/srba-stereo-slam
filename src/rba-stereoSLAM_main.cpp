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

#include "rba-stereoSLAM.h"
#include "rba-stereoSLAM_utils.h"
#include "rba-stereoSLAM-common.h"
#include <iterator>

#define DA_METHOD_BOW
#define DEBUG_OUTLIERS 0

#define ENTER_LOGGER( _STR ) if( app_options.enableLogger ) tLog.enter( _STR );
#define LEAVE_LOGGER( _STR ) if( app_options.enableLogger ) tLog.leave( _STR );

//	:: global config file
CConfigFile config;

//	:: time logger
CTimeLogger tLog;
CTimeLogger tLog_define_kf;
vector<double> define_kf_times;

typedef struct TStatsSRBA
{
	double time;
	size_t numberKFs, numberFeatsNew, numberFeatsCommon;

	TStatsSRBA( const double _time, 
				const size_t _numberFeatsNew = 0,
				const size_t _numberFeatsCommon = 0,
				const size_t _numberKFs = 0 ) : 
		time(_time), 
		numberFeatsNew(_numberFeatsNew), 
		numberFeatsCommon(_numberFeatsCommon),
		numberKFs(_numberKFs)
	{}
} TStatsSRBA;

vector<TStatsSRBA> stats;

//	:: app and stslam options
TStereoSLAMOptions stereo_slam_options;
TAppOptions app_options;

//	:: binary vocabulary and database
BriefVocabulary voc;
BriefDatabase db;

vector<CImage> kf_small_imgs;
vector< pair<double,double> > out_pos;

TColor colors[] = {TColor::red,TColor::green,TColor::gray, TColor::black, TColor::blue, TColor::white};
const size_t num_colors = 6;

mySRBA::TOpenGLRepresentationOptions  opengl_params;
mySRBA::TSRBAParameters               srba_params;

size_t CStereoSLAMKF::m_last_match_ID = 0;							// Initial value of the matches ID

/* ------------------------------------------
        main method of this application
 -------------------------------------------- */
static void performStereoSLAM()
{
	opengl_params.span_tree_max_depth			= 1000;//2*rba.parameters.srba.max_tree_depth;
	opengl_params.draw_unknown_feats_ellipses	= false;
	opengl_params.show_unknown_feats_ids		= false;
	opengl_params.draw_unknown_feats			= false;
	opengl_params.draw_kf_hierarchical			= true;

	define_kf_times.reserve(3000);
	stats.reserve(3000);

#if DEBUG_OUTLIERS
	vector<size_t> auxV;
	config.read_vector("DEBUG","outliers_IDs",vector<size_t>(),auxV,false);
	cout << "outliers_IDs size: " << auxV.size() << endl;

	vector<bool> auxVFirstTime;
	auxVFirstTime.resize(auxV.size(),true);
#endif

    // VARIABLE DECLARATION -----------------------------------------
    //  :: bundle adjustement
    mySRBA rba;
	t_vector_kf keyFrames; // keyFrames.reserve(500);

    //  :: visual odometry
    rso::CStereoOdometryEstimator voEngine;
    rso::CStereoOdometryEstimator::TStereoOdometryRequest odom_request;
	rso::CStereoOdometryEstimator::TStereoOdometryResult  odom_result;

	//	:: auxiliary poses
	CPose3DRotVec current_pose;                 // The estimated GLOBAL pose of the camera = last_kf_pose + vo_result + camera_pose_on_robot
	CPose3DRotVec last_kf_pose;					// The GLOBAL pose of the last kf (set from RBA engine at each new KF insertion)
	CPose3DRotVec incr_pose_from_last_kf;		// Accumulated incremental pose from the last KF (computed from vo_results at each time step, reset at new KF insertion)
	CPose3DRotVec incr_pose_from_last_check;	// Accumulated incremental pose from the last check of new insertion KF (computed at each time step, reset at new KF check)

	//	:: stereocamera
	std::vector<std::string> paramSections;
	vector<double> p(6);
	config.read_vector( "GENERAL", "camera_pose_on_robot", vector<double>(6,0), p, false );
	const CPose3D camera_pose_on_robot( p[0],p[1],p[2],DEG2RAD(p[3]),DEG2RAD(p[4]),DEG2RAD(p[5]) );	
	const CPose3D img_to_camera_pose(0,0,0,DEG2RAD(-90),0,DEG2RAD(-90));
	const CPose3D image_pose_on_robot = camera_pose_on_robot+img_to_camera_pose;
	const CPose3DRotVec camera_pose_on_robot_rvt( image_pose_on_robot );
	CPose3DRotVec camera_pose_on_robot_rvt_inverse = camera_pose_on_robot_rvt;
	camera_pose_on_robot_rvt_inverse.inverse();

	stereo_slam_options.camera_pose_on_robot_rvt = camera_pose_on_robot_rvt;
	stereo_slam_options.camera_pose_on_robot_rvt_inverse = camera_pose_on_robot_rvt_inverse;

    //  :: rawlog file
    mrpt::hwdrivers::CCameraSensor myCam;       // The generic image source

    //  :: online visualization variable declaration and view initialization
	CDisplayWindow3DPtr win;
	CImagePtr			imColor;
	size_t				im1_w = 0;

	if( app_options.show3D )
	{
		win = CDisplayWindow3D::Create("RBA resulting map",1024,768);
		{
			COpenGLScenePtr &scene = win->get3DSceneAndLock();
			
			CSetOfObjectsPtr cam = stock_objects::BumblebeeCamera();
			cam->setName("camera");
			scene->insert(cam);

			{	// image viewport
				COpenGLViewportPtr vp = scene->createViewport("image_left");
				vp->setViewportPosition(5,5,497,150);
				vp->setBorderSize(1);
			}

			{	// image viewport
				COpenGLViewportPtr vp = scene->createViewport("image_right");
				vp->setViewportPosition(502,5,497,150);
				vp->setBorderSize(1);
			}

			{	// query results viewport
				COpenGLViewportPtr vp = scene->createViewport("keyframes");
				vp->setViewportPosition(5,160,200,150);
				vp->setCustomBackgroundColor( TColorf( TColor::gray ) );
				vp->setBorderSize(1);
				
				CCamera & cam = vp->getCamera();
				cam.setElevationDegrees( 90 );
				cam.setAzimuthDegrees( 270 );
				cam.setProjectiveModel(false);
				cam.setZoomDistance( 2 );
				cam.setPointingAt(TPoint3D(0.875,0,0));

				for( size_t m = 0; m < 4; ++m )
				{
					CTextPtr score_txt = CText::Create();
					score_txt->setName( mrpt::format("ret%d_score",m) );
					score_txt->setPose( CPoint3D(0.5*m,-0.15,0.0) );
					score_txt->setVisibility(false);
					vp->insert(score_txt);

					CTextPtr id_txt = CText::Create();
					id_txt->setName( mrpt::format("ret%d_id",m) );
					id_txt->setPose( CPoint3D(0.5*m+0.125,-0.4,0.0) );
					id_txt->setVisibility(false);
					vp->insert(id_txt);

					CBoxPtr box = CBox::Create();
					box->setName( mrpt::format("ret%d_box",m) );
					box->setColor( TColorf(1,0,0) );
					box->setVisibility(false);
					vp->insert(box);
				}
				
				CSimpleLinePtr line = CSimpleLine::Create();
				line->setName( "th_line" );
				vp->insert(line);

				CTextPtr txt = CText::Create();
				txt->setName( "th_value" );
				vp->insert(txt);

				vp->insert( CSimpleLine::Create(-0.25,0,0,2.0,0,0,2.0) );
				
				CTextPtr txt2 = CText::Create("DB Query result");
				txt2->setPose( TPose3D(0,-0.75,0,0,0,0) );
				vp->insert( txt2 );
			}

			win->unlockAccess3DScene();
			win->setCameraZoom(4);
		}
		win->forceRepaint();
	}
    // ----------------------------------------- end of variable declaration

	// INITIALIZE STEREO CAMERA & VISUAL ODOMETER -----------------------------------------
    odom_request.stereo_cam = stereo_slam_options.stCamera;

    paramSections.push_back("RECTIFY");
    paramSections.push_back("DETECT");
    paramSections.push_back("MATCH");
    paramSections.push_back("LEAST_SQUARES");
    paramSections.push_back("GUI");
	paramSections.push_back("GENERAL");

    voEngine.loadParamsFromConfigFile(config, paramSections);
	voEngine.setVerbosityLevel( config.read_int("GENERAL","vo_verbosity",0,false) );
	voEngine.dumpToConsole();

    //  :: fill the application options with the same information as the visual odometry
	stereo_slam_options.detect_fast_th					= voEngine.params_detect.initial_FAST_threshold;
	stereo_slam_options.n_levels						= voEngine.params_detect.orb_nlevels;
	stereo_slam_options.n_feats							= voEngine.params_detect.orb_nfeats;
	stereo_slam_options.matching_options.epipolar_TH	= voEngine.params_lr_match.max_y_diff;
	stereo_slam_options.matching_options.maxORB_dist    = voEngine.params_lr_match.orb_max_distance;
	// ----------------------------------------- end of initialize stereo camera & visual odometer

    // INITIALIZE SRBA -----------------------------------------
	//  :: camera
    rba.parameters.sensor.camera_calib.leftCamera       = stereo_slam_options.stCamera.leftCamera;
    rba.parameters.sensor.camera_calib.rightCamera      = stereo_slam_options.stCamera.rightCamera;
    rba.parameters.sensor.camera_calib.rightCameraPose  = stereo_slam_options.stCamera.rightCameraPose;
	rba.parameters.sensor_pose.relative_pose			= image_pose_on_robot; 
    current_pose = camera_pose_on_robot_rvt;							// initial pose of the camera: Z forwards, Y downwards, X to the right

	//  :: topology
	/** In the new version, this cannot be selected from ini file/
	const int edge_policy = config.read_int("SRBA","srba_edge_policy",1,false);
	switch( edge_policy )
	{
		case 0: rba.parameters.srba.edge_creation_policy = mrpt::srba::ecpLinearGraph; break;
		case 1: rba.parameters.srba.edge_creation_policy = mrpt::srba::ecpICRA2013; break;
		default: rba.parameters.srba.edge_creation_policy = mrpt::srba::ecpStarGraph; break;
	}
	/**/
	mrpt::system::pause();
	rba.parameters.srba.max_tree_depth				= config.read_int("SRBA","srba_max_tree_depth",3,false);
	rba.parameters.srba.max_optimize_depth			= config.read_int("SRBA","srba_max_optimize_depth",3,false);

    //  :: other
    rba.setVerbosityLevel( config.read_int("SRBA","srba_verbosity", 0, false) );	// 0: None; 1:Important only; 2:Verbose
	rba.parameters.ecp.submap_size					= config.read_int("SRBA","srba_submap_size",15,false);
	rba.parameters.obs_noise.std_noise_observations = 0.5;							// pixels
	rba.parameters.srba.use_robust_kernel           = config.read_bool("SRBA","srba_use_robust_kernel",true,false);
	rba.parameters.srba.use_robust_kernel_stage1    = config.read_bool("SRBA","srba_use_robust_kernel_stage1",true,false);
	rba.parameters.srba.kernel_param				= config.read_double("SRBA","srba_kernel_param",3.0,false);
	// ----------------------------------------- end of initialize srba

	int last_num_tracked_feats = UNINITIALIZED_TRACKED_NUMBER;
	
	double max_rotation, max_translation, max_rotation_limit, max_translation_limit;
	max_rotation = stereo_slam_options.max_rotation;
	max_translation = stereo_slam_options.max_translation;
	max_rotation_limit = 2*max_rotation;
	max_translation_limit = 2*max_translation;
	
    //	:: 0.1 load images from a camera (or rawlog, or image dir)
    // -----------------------------------------
	string str;
	if( app_options.cap_src == TAppOptions::csRawlog )
	{
		ASSERT_( fileExists( app_options.rawlog_file ) );
		str = string(
			"[CONFIG]\n"
			"grabber_type=rawlog\n"
			"capture_grayscale=false\n"
			"rawlog_file=") + app_options.rawlog_file +
			string("\n");
	}
	else if( app_options.cap_src == TAppOptions::csImgDir )
	{
		str = string(
			"[CONFIG]\n"
			"grabber_type=image_dir\n"
			"image_dir_url=") + app_options.cap_dir_url +
			string("\n left_format=") + app_options.cap_img_left_format +
			string("\n right_format=") + app_options.cap_img_right_format +
			string("\n start_index=") + mrpt::format("%d\n",app_options.cap_img_start_index).c_str() +
			string("\n end_index=") + mrpt::format("%d\n",app_options.cap_img_end_index).c_str();
	}

	myCam.loadConfig( mrpt::utils::CConfigFileMemory(str), "CONFIG" );
    
	//	:: try to start grabbing images: (will raise an exception on any error)
    myCam.initialize();

    mrpt::obs::CObservationPtr obs;
    size_t count = 0;
    size_t kfID  = 0;

	//	:: 0.2 load state
	// -----------------------------------------
	size_t start_iteration = 0;
	if( app_options.load_state_from_file )
	{
		cout << "Loading state from file: " << app_options.state_file << " ... ";
		ASSERT_(fileExists( app_options.state_file ))
		
		loadApplicationState(	app_options.state_file, 
								start_iteration, 
								last_num_tracked_feats,
								current_pose,					// poses
								last_kf_pose,					// poses
								incr_pose_from_last_kf,			// poses
								incr_pose_from_last_check,		// poses
								keyFrames,						// kf
								stereo_slam_options, 
								voEngine,
								rba,
								app_options,
								stereo_slam_options,
								db );

		//	:: update kf identifier counter
		kfID = keyFrames.rbegin()->m_kfID+1;
		
		//	:: build 3d visualization
		if( app_options.show3D )
		{
            COpenGLScenePtr & scene = win->get3DSceneAndLock();

            CRenderizablePtr obj = scene->getByName("srba");
            CSetOfObjectsPtr rba_3d;

            if( obj )
                rba_3d = static_cast<CSetOfObjectsPtr>(obj);
            else
            {
                rba_3d = CSetOfObjects::Create();
                rba_3d->setName("srba");
                scene->insert(rba_3d);
            }
            rba.build_opengl_representation( 0 /* root kf*/, opengl_params /* rendering options*/, rba_3d /* output scene */ );

			// show KF ids
			show_kf_numbers( scene, keyFrames.size(), QueryResults(), 0 );

			win->unlockAccess3DScene();
			win->forceRepaint();
        }
		
		cout << " done. Starting at iteration: " << start_iteration << endl;
	}

	// *************************************************
	// MAIN LOOP: OVER STEREO IMAGE OBSERVATIONS
	// *************************************************
	FILE *fstd = mrpt::system::os::fopen( "std.txt", "wt" );
	FILE *ft = mrpt::system::os::fopen( "da.txt", "wt" );
	FILE *fls = mrpt::system::os::fopen( "ls.txt", "wt" );

	size_t iterations_since_last_vo_check = 0;
	bool end_app = false;
    while ( (obs = myCam.getNextFrame()).present() && !end_app )
    {
		if( app_options.show3D )
		{
			win->get3DSceneAndLock();
			win->addTextMessage( 210, 180, mrpt::format("#Frame: %lu", count ), TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 1);
			win->unlockAccess3DScene();
			win->repaint();
		}

		//	:: load state and/or skip until desired frame
		if( app_options.load_state_from_file && start_iteration > 0 && count < start_iteration )
		{
			cout << ".";
			count++;
			continue;
		}
		else
		{
			if( count < size_t(app_options.from_step) )
			{
				if( count == 0 ) cout << "Skipping frame until " << app_options.from_step << " ";
				cout << ".";
				count++;
				continue;
			}
		}
		cout << endl << endl;
		cout << " >> Frame # " << count << endl;

		//  :: get the images
        CObservationStereoImagesPtr stImgs = static_cast<CObservationStereoImagesPtr>(obs);

		// ----------------------------------------
		//  :: first iteration
		// ----------------------------------------
        if( keyFrames.size() == 0 )
        {
			keyFrames.push_back( CStereoSLAMKF() );			// create new KF
			CStereoSLAMKF & new_kf = keyFrames.back();		// reference to new element
			new_kf.setKFID( kfID++ );

			// DEBUG -----------------------------------------
			if( app_options.debug )										// debug mode: save images, feats positions and stereo matches
			{
				//	:: save images
				stImgs->imageLeft.saveToFile( mrpt::format("%s\\image_left_kf%04d.jpg", app_options.out_dir.c_str(), new_kf.m_kfID) ); 
				stImgs->imageRight.saveToFile( mrpt::format("%s\\image_right_kf%04d.jpg", app_options.out_dir.c_str(), new_kf.m_kfID) ); 
			}
			// -----------------------------------------------

			if( app_options.show3D )
			{
				COpenGLScenePtr & scene = win->get3DSceneAndLock();

				// :: show images in viewport
				scene->getViewport("image_left")->setImageView( stImgs->imageLeft );
				scene->getViewport("image_right")->setImageView( stImgs->imageRight );

				win->unlockAccess3DScene();
				win->repaint();
			}

			//	:: process new stereo pair from VO and get the results
			odom_request.stereo_imgs = stImgs;
			voEngine.processNewImagePair( odom_request, odom_result );
			new_kf.getDataFromVOEngine( voEngine );
			new_kf.insertIntoDB( db );

			//	:: ids management
			new_kf.generateMatchesIDs();
			voEngine.resetIds();

			const size_t nMatches = new_kf.m_matches.size();		// number of matches in this keyframe
			VERBOSE_LEVEL(1) << "	# feats (L/R) = " << new_kf.m_keyPointsLeft.size() 
							 << "/" << new_kf.m_keyPointsRight.size() 
							 << " -- # matches = " << nMatches << endl;

            if( app_options.verbose_level >= 2 ) 
				new_kf.dumpToConsole();

			if( app_options.debug )										// debug mode: save images, feats positions and stereo matches
				new_kf.saveInfoToFiles();

            //  :: insert KF data into SRBA engine
			mySRBA::TNewKeyFrameInfo       newKFInfo;
            mySRBA::new_kf_observations_t  listObs;
            mySRBA::new_kf_observation_t   obsField;

            obsField.is_fixed                   = false;	// landmarks have unknown relative positions (i.e. treat them as unknowns to be estimated)
            obsField.is_unknown_with_init_val   = true;		// we have a guess of the initial LM position

            listObs.resize( nMatches );

            //  :: fill observation fields
            for( size_t m = 0; m < nMatches; ++m )
            {
				const size_t id1 = new_kf.m_matches[m].queryIdx;
                const size_t id2 = new_kf.m_matches[m].trainIdx;

                const KeyPoint & kpLeft  = new_kf.m_keyPointsLeft[id1];
                const KeyPoint & kpRight = new_kf.m_keyPointsRight[id2];

                obsField.obs.feat_id            = new_kf.m_matches_ID[m];
                obsField.obs.obs_data.left_px   = TPixelCoordf( kpLeft.pt.x,  kpLeft.pt.y );
                obsField.obs.obs_data.right_px  = TPixelCoordf( kpRight.pt.x, kpRight.pt.y );

				// initial positions of the landmark
				obsField.setRelPos( camera_pose_on_robot_rvt + projectMatchTo3D( kpLeft.pt.x, kpLeft.pt.y, kpRight.pt.x, stereo_slam_options.stCamera ) );

                listObs[m] = obsField;
            } // end for

            //  :: insert into the SRBA framework
            MRPT_TRY_START
			tLog_define_kf.enter("define_kf");
			rba.define_new_keyframe( listObs,				// list of observations
                                     newKFInfo,				// keyframe info
                                     false );				// not optimize the first time
			tLog_define_kf.leave("define_kf");
			stats.push_back( TStatsSRBA( tLog_define_kf.getMeanTime("define_kf"), listObs.size() ) );
			tLog_define_kf.clear();
			MRPT_TRY_END

			if( app_options.verbose_level >= 2 ) 
				rba.dumpLocalmapCenters();

			mrpt::system::os::fprintf(fls,"%d %.4f\n",newKFInfo.kf_id,newKFInfo.optimize_results_stg1.obs_rmse);

            VERBOSE_LEVEL(1)	<< "-------------------------------------------------------" << endl
								<< "   Created KF #" << newKFInfo.kf_id
								<< " | # kf-to-kf edges created: " <<  newKFInfo.created_edge_ids.size()  << endl
								<< "   Optimization error: " << newKFInfo.optimize_results.total_sqr_error_init << " -> " << newKFInfo.optimize_results.total_sqr_error_final << endl
								<< "-------------------------------------------------------" << endl;

            //  :: and save it in the main vector 
			VERBOSE_LEVEL(2) << "Detected keypoints: Left(" << odom_result.detected_feats.first << ") and Right(" << odom_result.detected_feats.second << ")" << endl
							 << "Matches found: " << odom_result.stereo_matches << endl;

			//	:: set this pose as the last kf 'GLOBAL' pose
			last_kf_pose = new_kf.m_camPose;

            //  :: visualization (quite slow) --> consider a new thread for this
			if( app_options.show3D )
			{
                COpenGLScenePtr & scene = win->get3DSceneAndLock();

                // stereo camera
				CSetOfObjectsPtr cam = static_cast<CSetOfObjectsPtr>(scene->getByName("camera"));
                cam->setPose( CPose3D(current_pose) );

				// srba global map visualization
				CSetOfObjectsPtr rba_3d;
				CRenderizablePtr obj = scene->getByName("srba");
                if(obj) rba_3d = static_cast<CSetOfObjectsPtr>(obj);
                else
                {
                    rba_3d = CSetOfObjects::Create();
                    rba_3d->setName("srba");
                    scene->insert(rba_3d);
                }

                rba.build_opengl_representation( 0 /*Root KF*/, opengl_params /*Rendering options*/, rba_3d /*Output scene*/ );

				// show KF ids
				show_kf_numbers( scene, keyFrames.size(), QueryResults(), 0 );

				win->unlockAccess3DScene();
				win->repaint();
			}

			iterations_since_last_vo_check++;
        } // end first keyframe
		// ----------------------------------------
		//  :: next iterations
		// ----------------------------------------
		else
        {
			if( app_options.save_state_fo_file && count == app_options.save_at_iteration )
			{
				VERBOSE_LEVEL(1) << " Saving state to file at iteration " << app_options.save_at_iteration << " ... ";
				
				//	:: save state and exit
				saveApplicationState(
						app_options.state_file, 
						count,
						last_num_tracked_feats,
						current_pose,
						last_kf_pose,
						incr_pose_from_last_kf,
						incr_pose_from_last_check,
						keyFrames,
						stereo_slam_options,
						voEngine,
						app_options,
						db );
				
				VERBOSE_LEVEL(1) << " done." << endl << "Exiting" << endl;
				
				return;
			}
			//	:: process:
			//		--> perform VO
			//		--> check if the camera has moved enough from last KF OR we've lost too many features OR current images are too different from the rest of the db
			//		--> if it has, create a new KF and perform data association (DA)
			
			//  :: only perform VO
            //  :: track the position of the last keypoints
			CImage imL(UNINITIALIZED_IMAGE), imR(UNINITIALIZED_IMAGE);
			if( app_options.show3D )
			{
				imL.copyFromForceLoad(stImgs->imageLeft);
				imR.copyFromForceLoad(stImgs->imageRight);
			}

			odom_request.stereo_imgs            = stImgs;	// just copy the pointer
            odom_request.use_precomputed_data   = false;
            odom_request.repeat					= false;

			cout << " --> Looking for KF#" << kfID << endl; 
			do
			{
				//	:: perform VO
				VERBOSE_LEVEL(2) << " Processing new stereo pair ... (tstamp: " << stImgs->timestamp <<  ": " << stImgs->imageLeft.getWidth() << "x" << stImgs->imageLeft.getHeight() << ")" << endl;
				voEngine.processNewImagePair( odom_request, odom_result ); // WARNING --> after this call, 'stImgs' contains the previous images; do not use them!
				VERBOSE_LEVEL(2) << "	Detected keypoints: Left(" << odom_result.detected_feats.first << ") and Right(" << odom_result.detected_feats.second << ")" << endl
								 << "	Matches found: " << odom_result.stereo_matches << endl;
				
				if( stereo_slam_options.orb_adaptive_fast_th )
				{
					if( int(odom_result.stereo_matches) < stereo_slam_options.adaptive_th_min_matches )
					{	
						if( !voEngine.isFASTThMin() )
						{
							voEngine.setFASTThreshold( std::max(0,voEngine.getFASTThreshold()-10) );
							odom_request.repeat = true;
							VERBOSE_LEVEL(0) << "Number of stereo matches is too low! (" << odom_result.stereo_matches << ") Repeat detection with a lower FAST threshold: " << voEngine.getFASTThreshold() << endl;
						}
						else if( !voEngine.isORBThMax() )
						{
							// images contain few keypoints, allow worse matches --> increase ORB threshold
							voEngine.setORBThreshold( voEngine.getORBThreshold()+10 );
							odom_request.repeat = true;
							VERBOSE_LEVEL(0) << "Number of stereo matches is still too low! (" << odom_result.stereo_matches << ") Repeat detection with a higher ORB threshold: " << voEngine.getORBThreshold() << endl;
						}
						else if( odom_result.stereo_matches >= 8 )
						{
							// we have reached the limits but we have at least the minimum set of matches --> try to continue
						}
						else
						{
							THROW_EXCEPTION( "The number of found matches is less than the minimum. Aborting" )
						}
					}
					else if(odom_result.stereo_matches < stereo_slam_options.adaptive_th_min_matches*1.2/*stereo_slam_options.n_feats*0.25*/ )
					{	
						// number of stereo matches is low --> reduce FAST threshold for future but continue
						VERBOSE_LEVEL(2) << "Number of stereo matches is low! (" << odom_result.stereo_matches << ") Reduce FAST threshold for the next iteration" << endl;
						if( !voEngine.isFASTThMin() )
							voEngine.setFASTThreshold( std::max(0,voEngine.getFASTThreshold()-5) );
						else if( !voEngine.isORBThMax() )
							voEngine.setORBThreshold( voEngine.getORBThreshold()+5 );
						odom_request.repeat = false;
					}
					else
					{	
						// we are good --> increase the FAST threshold
						voEngine.setFASTThreshold( std::min(stereo_slam_options.detect_fast_th,voEngine.getFASTThreshold()+5) );
						voEngine.resetORBThreshold();
						odom_request.repeat = false;
					}
				} // end-if
			} while( odom_request.repeat ); // end-while

			if( !odom_result.valid )
			{
				VERBOSE_LEVEL(1) << "	[Warnivng - VO Engine] -- Not a valid result! Skipping this frame." << endl;
				count++;
				continue;
			}
			
			//  :: update current 'estimated' pose of the camera:
			//		'incr_pose' = pose of the current stereo frame wrt the previous one
			CPose3DRotVec incr_pose( odom_result.outPose );
            current_pose				+= incr_pose;
			incr_pose_from_last_kf		+= incr_pose;
			incr_pose_from_last_check	+= incr_pose;

			//	:: update camera position in the visualization
			if( app_options.show3D )
			{
                COpenGLScenePtr & scene = win->get3DSceneAndLock();
                CSetOfObjectsPtr cam = static_cast<CSetOfObjectsPtr>( scene->getByName("camera") );
                cam->setPose( CPose3D(current_pose) );

				// :: image viewport
				scene->getViewport("image_left")->setImageView( imL );
				scene->getViewport("image_right")->setImageView( imR );

                win->unlockAccess3DScene();
				win->repaint();
            }

			// VERBOSE ------------------------------------------------------
			VERBOSE_LEVEL(1)	<< "	[VO] # tracked features from last frame: " << odom_result.tracked_feats_from_last_frame << endl
								<< "	[VO] # tracked features from last KF: " << odom_result.tracked_feats_from_last_KF << endl;
			
			VERBOSE_LEVEL(2)	<< "	[VO] Incremental Pose: " << incr_pose << endl
								<< "	[VO] Incremental Pose from last KF: " << incr_pose_from_last_kf << endl
								<< "	[VO] Current pose: " << current_pose << endl;
			// --------------------------------------------------------------

			//	:: check if this frame is far enough from the last KF to force it to be a new KF
			const double incTranslationKf = incr_pose_from_last_kf.m_coords.norm();
			const double incRotationKf = incr_pose_from_last_kf.m_rotvec.norm();
			bool voForceNewKf = incTranslationKf > max_translation_limit || incRotationKf > DEG2RAD( max_rotation_limit );		// it may be modified later

			VERBOSE_LEVEL(1) << "	[VO Check] -- Last KF distance: " << voForceNewKf << " (" << incTranslationKf << " m.," << RAD2DEG(incRotationKf) << "deg) vs Th: (" << max_translation_limit << " m.," << max_rotation_limit << "deg)" << endl;

			//	:: check if visual odometer has lost too many features (force a new KF check)
			const bool voForceCheckTracking = 
				app_options.vo_id_tracking_th == 0 ? 
				false : 
				int(odom_result.tracked_feats_from_last_KF) < app_options.vo_id_tracking_th;

			//	:: check if this frame is far enough from the last check to be a candidate for a new keyframe (and the number of feats we've tracked is low)
			const double incr_translation	= incr_pose_from_last_check.m_coords.norm();
			const double incr_rotation		= incr_pose_from_last_check.m_rotvec.norm();
			const bool voForceCheckDistance = 
				incr_translation > max_translation || 
				incr_rotation > DEG2RAD( max_rotation );		
			
			const bool voForceCheck = voForceCheckTracking || voForceCheckDistance;
			
			// VERBOSE ------------------------------------------------------
			VERBOSE_LEVEL(1) << "	[VO Check] -- Check distance: " << voForceCheckDistance << " (" << incr_translation << " m.," << RAD2DEG(incr_rotation) << "deg) vs Th: (" << max_translation << " m.," << max_rotation << "deg)" << endl
							 << "	[VO Check] -- Feature tracking: " << voForceCheckTracking << " (" << odom_result.tracked_feats_from_last_KF << ") vs Th: (" << app_options.vo_id_tracking_th << ")" << endl;
			// --------------------------------------------------------------
			
			int best_da = 0;
			if( voForceNewKf || voForceCheck )
			{	
				VERBOSE_LEVEL(1) << "	[VO Check] -- Visual odometry forced checking for a new keyframe." << endl;
				
				bool insertNewKf = false;
				
				//	:: reset visual odometer ids
				voEngine.resetIds();
				
				//	:: clear this pose
				incr_pose_from_last_check = CPose3DRotVec();

				//	:: get current number of KFs (before inserting a new one)
				const size_t num_kfs = keyFrames.size();

				//	:: create a temporary keyframe
				keyFrames.push_back( CStereoSLAMKF() );
				CStereoSLAMKF & new_kf = keyFrames.back();
				new_kf.getDataFromVOEngine( voEngine );

				//	:: set the candidate KF ID (it will be discarded if needed)
				new_kf.setKFID( kfID );

				const size_t nMatches = new_kf.m_matches.size();		// number of stereo matches in this KF
				new_kf.m_matches_ID.resize( nMatches, 0 );				// matches Ids will be filled later if needed
				
				//	:: query db
				QueryResults qResults;
				rba.get_time_profiler().enter("queryDB");
				new_kf.queryDB( db, qResults, 4 );
				rba.get_time_profiler().leave("queryDB");

				ASSERTDEB_( qResults.size() > 0 )

				//	:: update query score
				double qScoreTh = stereo_slam_options.query_score_th != 0 ? 
					stereo_slam_options.query_score_th :
					updateQueryScoreThreshold( last_num_tracked_feats );
	
				//	:: analyse query results
				bool confirmedLoopClosure = false;
				TLoopClosureInfo lcInfo;
				rba.get_time_profiler().enter("get_similar_kfs");
				const bool potentialLoopClosure = getSimilarKfs(
								new_kf.m_kfID,
								qResults, 
								rba, 
								stereo_slam_options, 
								lcInfo );
				rba.get_time_profiler().leave("get_similar_kfs");
				if( qResults[0].Score < 0.05 )
					voForceNewKf = true;

				if( qResults[0].Score > 0.5 )	// saves time
				{
					cout << "qResult[0] is significantly large ==> skip the test." << endl;
				}
				else
				{
					VERBOSE_LEVEL(2) << "	Performing data association" << endl;
					//	:: perform data association
					TVectorKfsDaInfo daInfo;
				
					rba.get_time_profiler().enter("performDA");
					new_kf.performDataAssociation( 
								keyFrames, 
								lcInfo, 
								voEngine, 
								daInfo, 
								odom_request.stereo_cam, 
								stereo_slam_options,
								incr_pose_from_last_kf );
					rba.get_time_profiler().leave("performDA");

					const size_t numberSimilarKfs = daInfo.size();
					ASSERT_( numberSimilarKfs > 0 )

					rba.get_time_profiler().enter("confirmLC");
					//	:: check data association results
					//		- variable 'daInfo' has at least size 1 (the best result from the DB query), but it may contain other DA with KFs that are similar to the current one.
					//		- order all the keyframes according to the number of common observed features.
					vector<size_t> sortedIndices( numberSimilarKfs );
					for( size_t i = 0; i < numberSimilarKfs; i++ ) sortedIndices[i] = i;
					if( numberSimilarKfs > 1 )
						std::sort( sortedIndices.begin(), sortedIndices.end(), DATrackedSorter(daInfo) );

					// VERBOSE ------------------------------------------------------
					VERBOSE_LEVEL(1) << "	:: Tracked features" << endl;
					if( app_options.verbose_level >= 1 )
						for( vector<size_t>::iterator it = sortedIndices.begin(); it != sortedIndices.end(); ++it )
							cout << "		with " << daInfo[*it].kf_idx << " -> " << daInfo[*it].tracked_matches << " tracked features." << endl;
					// --------------------------------------------------------------

					const size_t highestNumberTrackedFeats = daInfo[sortedIndices[0]].tracked_matches;
					last_num_tracked_feats = highestNumberTrackedFeats;

					if(!potentialLoopClosure)					
					{
						if( voForceNewKf ) 
						{
							VERBOSE_LEVEL(1) << "	:: VO forced the insertion of a new keyframe." << endl;
							insertNewKf = true;
						}
						else if( voForceCheck )
						{
							if( highestNumberTrackedFeats < stereo_slam_options.updated_matches_th )
							{
								VERBOSE_LEVEL(1) << "	:: Tracked features below the threshold (" << stereo_slam_options.updated_matches_th << ") ==> insert a new KF" << endl;
								insertNewKf = true;
							}
							else
							{
								//	:: the number of tracked feats is still quite big, skip inserting a new kf
								const size_t olimit = stereo_slam_options.updated_matches_th+stereo_slam_options.up_matches_th_plus;
								if( highestNumberTrackedFeats <= olimit )
								{
									//	:: update the dynamic threshold for inserting a new keyframe (decreasing linearly when the number of tracked features is below the threshold 'updated_matches_th'+'up_matches_th_plus')
									max_translation = updateTranslationThreshold( highestNumberTrackedFeats-stereo_slam_options.updated_matches_th, stereo_slam_options.up_matches_th_plus );
									max_rotation = updateRotationThreshold( highestNumberTrackedFeats, olimit );

									VERBOSE_LEVEL(2) << "	New Translation/Rotation thresholds: " << max_translation << " m./" << max_rotation << " deg" << endl;
								} // end-if
							}
						} // end-else-if
					
						rba.loopClosureDetected(false);
				
					} // end-if				
					else
					{
						VERBOSE_LEVEL(2) << "POTENTIAL LOOP CLOSURE" << endl; 
			
						//	:: loop closure is confirmed if the number of tracked feats with the old KF is over a threshold
						size_t loopClosureIdx = 0;
						for (size_t i = 0; !confirmedLoopClosure && i < daInfo.size(); i++)
						{
							confirmedLoopClosure = 
								daInfo[i].kf_idx == lcInfo.lc_id && 
								daInfo[i].tracked_matches > 0.5*highestNumberTrackedFeats;

							if( confirmedLoopClosure ) loopClosureIdx = i;
						} // end-for
					
						if( confirmedLoopClosure )
						{
							rba.loopClosureDetected();
							rba.setLoopClosureOldID( lcInfo.lc_id );
							insertNewKf = true;

							//	:: give priority to the old keyframe
							for (size_t i = 0; i < sortedIndices.size(); ++i )
							{
								if( sortedIndices[i] == loopClosureIdx )
								{
									sortedIndices.erase( sortedIndices.begin()+i );
									break;
								}
							} // end-for
							sortedIndices.insert( sortedIndices.begin(), loopClosureIdx );

							VERBOSE_LEVEL(2) << "Loop closure CONFIRMED" << endl;
						} // end-if
						else
						{
							VERBOSE_LEVEL(2) << "Loop closure NOT CONFIRMED." << endl;
							if( highestNumberTrackedFeats < stereo_slam_options.updated_matches_th )
							{
								VERBOSE_LEVEL(1) <<	"	but feats below a threshold, insert keyframe." << endl;
								insertNewKf = true;
							}
							else
							{
								//	:: the number of tracked feats is still quite big, skip inserting a new kf
								const size_t olimit = stereo_slam_options.updated_matches_th+stereo_slam_options.up_matches_th_plus;
								if( highestNumberTrackedFeats <= olimit )
								{
									//	:: update the dynamic threshold for inserting a new keyframe (decreasing linearly when the number of tracked features is below the threshold 'updated_matches_th'+'up_matches_th_plus')
									max_translation = updateTranslationThreshold( highestNumberTrackedFeats-stereo_slam_options.updated_matches_th, stereo_slam_options.up_matches_th_plus );
									max_rotation = updateRotationThreshold( highestNumberTrackedFeats, olimit );

									VERBOSE_LEVEL(2) << "	New Translation/Rotation thresholds: " << max_translation << " m./" << max_rotation << " deg" << endl;
								} // end-if
							}
						}
					} // end-else

					rba.get_time_profiler().leave("confirmLC");

					// 3D REPRESENTATION ------------------------------------
					if( app_options.show3D )
					{
						COpenGLScenePtr & scene = win->get3DSceneAndLock();
						show_kf_numbers( scene, num_kfs/*keyFrames.size()*/, qResults, qScoreTh );
						win->unlockAccess3DScene();
						win->forceRepaint();
					}
					// ------------------------------------------------------
					// ***************************************
					//	:: THIS IS A NEW KEYFRAME
					//	**************************************
					if( insertNewKf )
					{
						// DEBUG ---------------------------------------------------
						FILE *f = NULL;
						if( app_options.debug )
							f = mrpt::system::os::fopen( GENERATE_NAME_WITH_KF_OUT( da_dist, new_kf ), "wt" );
						// ---------------------------------------------------------

						//	:: set ids for the current keyframe features
						vector<int> vectorNumberTrackedFeats( numberSimilarKfs, 0 );
						size_t numberNewFeats = 0, numberTrackedFeats = 0;
						set<size_t> foundIds;
						for( size_t m = 0; m < nMatches; ++m )
						{
							bool tracked = false;
							for( size_t k = 0; !tracked && k < numberSimilarKfs; ++k )
							{
								const t_kf_da_info & da_info = daInfo[sortedIndices[k]];// shortcut

								if( da_info.tracking_info[m].first != INVALID_IDX )
								{	// tracked feature
									const TKeyFrameID & other_match_idx = da_info.tracking_info[m].first;
									const TKeyFrameID & other_match_id = keyFrames[da_info.kf_idx/*other_kf_idx*/].m_matches_ID[other_match_idx];
									
									if( foundIds.find(other_match_id) != foundIds.end() )
									{
										VERBOSE_LEVEL(2) << "	Feature tracked more than once: kf#" << new_kf.m_kfID << "-id:" << other_match_id << endl;
										// [TODO] check the distance and keep the best match, by now: keep the first one
										break;
									}
									foundIds.insert(other_match_id);

									new_kf.m_matches_ID[m] = other_match_id;
									++vectorNumberTrackedFeats[sortedIndices[k]];
									++numberTrackedFeats;
									tracked = true;

									// DEBUG ---------------------------------------------------
									if( app_options.debug )
										fprintf(f,"%2.f\n",daInfo[sortedIndices[k]].tracking_info[m].second);
									// ---------------------------------------------------------

								} // end-if
							} // end-for
							if( !tracked )
							{	// new feature
								new_kf.m_matches_ID[m] = CStereoSLAMKF::m_last_match_ID++;
								++numberNewFeats;

								// DEBUG ---------------------------------------------------
								if( app_options.debug )
									fprintf(f,"0.00\n");
								// ---------------------------------------------------------
							} // end-if
						} // end-for

						//	:: compute dispersion
						// [TODO]

						//	:: compute number of real tracked keyframes
						size_t realNumberSimilarKfs = 0;
						for( size_t m = 0; m < numberSimilarKfs; ++m )
						{
							if( vectorNumberTrackedFeats[sortedIndices[m]] >= 0.5*highestNumberTrackedFeats )
								realNumberSimilarKfs++;
							else
								break;
						}

						// DEBUG ---------------------------------------------------
						if( app_options.debug )
							mrpt::system::os::fclose(f);
						// ---------------------------------------------------------

						// VERBOSE -------------------------------------------------
						VERBOSE_LEVEL(1)	<< "	Real number of similar kfs: " << realNumberSimilarKfs << "/" << numberSimilarKfs << endl;
						VERBOSE_LEVEL(1)	<< "	Features ----------------------------------" << endl;
						for( size_t k = 0; k < numberSimilarKfs; ++k )
						{
							VERBOSE_LEVEL(1) << "	Tracked with [#" << daInfo[sortedIndices[k]].kf_idx << "]: " << vectorNumberTrackedFeats[sortedIndices[k]] << endl;
							mrpt::system::os::fprintf(ft, "%d %d %d\n", new_kf.m_kfID, daInfo[sortedIndices[k]].kf_idx, vectorNumberTrackedFeats[sortedIndices[k]]);
						}
						VERBOSE_LEVEL(1)	<< "	New features:       " << numberNewFeats << endl
											<< "	-------------------------------------------" << endl
											<< "	TOTAL:              " << numberTrackedFeats + numberNewFeats << endl;
						// ---------------------------------------------------------

						last_num_tracked_feats = UNINITIALIZED_TRACKED_NUMBER;

						VERBOSE_LEVEL(1) << "Inserting new Keyframe " << endl;
						kfID++;	// prepare KF id for the next one

						//	:: restore the original thresholds for inserting a new KF
						max_translation = stereo_slam_options.max_translation;
						max_rotation = stereo_slam_options.max_rotation;

						//  :: perform SRBA
						mySRBA::TNewKeyFrameInfo       newKFInfo;
						mySRBA::new_kf_observations_t  listObs( nMatches );
						size_t obs_idx = 0;
				
						//	:: create list of observations for the srba-slam engine
						mySRBA::new_kf_observation_t   obsField;
						obsField.is_fixed                   = false;
						obsField.is_unknown_with_init_val   = true;

						// DEBUG --------------------------------------------
						FILE* flc = NULL;
						CStereoSLAMKF* okf = NULL;
						if( app_options.debug && confirmedLoopClosure )
						{
							flc = mrpt::system::os::fopen( mrpt::format("%s\\loop_closure_info_%d.txt", app_options.out_dir.c_str(), count).c_str() ,"wt" );
							okf = &(keyFrames[qResults[0].Id]);
						}
						//----------------------------------------------------

						//	:: Create the input for the SRBA
						size_t outCounter = 0;
						vector<size_t>::iterator it_id = new_kf.m_matches_ID.begin();
						vector<cv::DMatch>::iterator it_ma = new_kf.m_matches.begin();
						while( it_id != new_kf.m_matches_ID.end() )
						{
#if DEBUG_OUTLIERS
							// check if the ID is within the manual outlier list and this is not the first time it is found
							vector<size_t>::iterator it = std::lower_bound( auxV.begin(), auxV.end(), *it_id );
							if( it != auxV.end() && !(*it_id < *it) )
							{
								const size_t pos = it-auxV.begin();	// position where it is found
								
								// if this is the first time it is found ...
								if( auxVFirstTime[pos] )
									auxVFirstTime[pos] = false;	// toggle indicator
								else
								{
									cout << "kk kf " << new_kf.m_kfID << " deleting obs: " << *it_id << endl;
									outCounter++;

									// delete match
									it_id = new_kf.m_matches_ID.erase(it_id);
									it_ma = new_kf.m_matches.erase(it_ma);
								
									continue;
								} // end-else
							} // end-if
#endif
							// else
							// add info to the new KF input
							obsField.obs.feat_id			= *it_id;
							KeyPoint & kpLeft				= new_kf.m_keyPointsLeft[it_ma->queryIdx];
							KeyPoint & kpRight				= new_kf.m_keyPointsRight[it_ma->trainIdx];

							obsField.obs.obs_data.left_px   = TPixelCoordf( kpLeft.pt.x,  kpLeft.pt.y );
							obsField.obs.obs_data.right_px  = TPixelCoordf( kpRight.pt.x, kpRight.pt.y );
							obsField.setRelPos( camera_pose_on_robot_rvt + projectMatchTo3D( kpLeft.pt.x, kpLeft.pt.y, kpRight.pt.x, stereo_slam_options.stCamera ) );
							listObs[obs_idx++]				= obsField;

							// increment iterators
							++it_id;
							++it_ma;

							if( app_options.debug && confirmedLoopClosure )
							{	
								KeyPoint *olfeat = NULL, *orfeat = NULL;
								for( size_t mm = 0; mm < okf->m_matches_ID.size(); ++mm )
								{
									if( obsField.obs.feat_id != okf->m_matches_ID[mm] )
										continue;
								
									olfeat = & (okf->m_keyPointsLeft[  okf->m_matches[mm].queryIdx ]);
									orfeat = & (okf->m_keyPointsRight[ okf->m_matches[mm].trainIdx ]);
								}
							
								if( !olfeat || !orfeat )
									continue;

								// save loop closure information
								mrpt::system::os::fprintf( flc,"%d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", 
									obsField.obs.feat_id,
									kpLeft.pt.x, kpLeft.pt.y,
									kpRight.pt.x, kpRight.pt.y,
									olfeat->pt.x, olfeat->pt.y,
									orfeat->pt.x, orfeat->pt.y );
							} // end-if
						} // end-while
	#if DEBUG_OUTLIERS
						listObs.resize( listObs.size() - outCounter );
	#endif
						// DEBUG --------------------------------------------
						if( app_options.debug )
						{	//	:: debug: save current images and the new KF info
							imL.saveToFile(mrpt::format("%s\\image_left_kf%04d.jpg", app_options.out_dir.c_str(), new_kf.m_kfID));
							imR.saveToFile(mrpt::format("%s\\image_right_kf%04d.jpg", app_options.out_dir.c_str(), new_kf.m_kfID));
							new_kf.saveInfoToFiles();
						}
						
						// --------------------------------------------------
						if( app_options.debug && confirmedLoopClosure )
						{
							mrpt::system::os::fclose(flc);
							okf = NULL;
						}

						//  :: dump all the content of the new KF to the console
						if( app_options.verbose_level >= 2 )
							new_kf.dumpToConsole();

						// DEBUG -------------------------------------------------------------
						if( app_options.debug )
						{	//	:: debug: save current data association to a file (equals the input of the srba-slam)
							ofstream fstr( mrpt::format( "%s\\da_info_%04d.txt", app_options.out_dir.c_str(), new_kf.m_kfID ).c_str() );
							fstr.setf( std::ios::fixed, std::ios::floatfield );
							fstr.precision(2);
				
							for( size_t kobs = 0; kobs < listObs.size(); ++kobs )
							{
								fstr << listObs[kobs].obs.feat_id << " "
									 << listObs[kobs].obs.obs_data.left_px.x << " "
									 << listObs[kobs].obs.obs_data.left_px.y << " " 
									 << listObs[kobs].obs.obs_data.right_px.x << " "
									 << listObs[kobs].obs.obs_data.right_px.y << endl;
							}
							fstr.close();
						}
						// -------------------------------------------------------------

						const size_t auxsize = listObs.size();
                
						//  :: insert the new observations into the rba-slam framework
						try
						{
#if USE_SE2
							CPose2D inputPose;
							if( app_options.useInitialPose )
								inputPose = CPose2D(CPose3D(incr_pose_from_last_kf));
							else
								inputPose = CPose2D();
#else
							CPose3D inputPose;
							if( app_options.useInitialPose )
								inputPose = CPose3D(incr_pose_from_last_kf);
							else
								inputPose = CPose3D();
#endif
							rba.setInitialKFPose( inputPose );

							VERBOSE_LEVEL(1) << "	Inserting " << listObs.size() << " observations in srba-slam engine" << endl;
							tLog_define_kf.enter("define_kf");
							rba.define_new_keyframe( listObs,
													 newKFInfo,
													 true );
							tLog_define_kf.leave("define_kf");
							stats.push_back( TStatsSRBA( tLog_define_kf.getMeanTime("define_kf"), listObs.size() ) );
							tLog_define_kf.clear();
							cout << "inserted stat #" << stats.size() << endl;
							mrpt::system::os::fprintf(fls,"%d %.4f\n",newKFInfo.kf_id,newKFInfo.optimize_results_stg1.obs_rmse);
						}
						catch (exception& e)
						{
							cout << "Standard exception: " << e.what() << endl;
						}
						catch (...)
						{
							cout << "EXCEPTION" << endl;
							
							if( app_options.debug )
								rba.save_graph_as_dot( mrpt::format("%s\\graph_at_exception.dot", app_options.out_dir.c_str() ) );
							
							CDisplayWindow3D final_win("Final global map");

							COpenGLScenePtr & scene = final_win.get3DSceneAndLock();
							scene->insert(CGridPlaneXY::Create(-100,100,-100,100,0,1));

							CRenderizablePtr obj = scene->getByName("srba");
							CSetOfObjectsPtr rba_3d;

							if( obj )
								rba_3d = static_cast<CSetOfObjectsPtr>(obj);
							else
							{
								rba_3d = CSetOfObjects::Create();
								rba_3d->setName("srba");
								scene->insert(rba_3d);
							}
							rba.build_opengl_representation(
								0,              // Root KF,
								opengl_params,  // Rendering options
								rba_3d          // Output scene
							);

							final_win.unlockAccess3DScene();
							final_win.forceRepaint();
							final_win.waitForKey();

							if( app_options.debug )
								scene->saveToFile( mrpt::format("%s\\exception_map.3Dscene", app_options.out_dir.c_str() ) );						
							
							cout << "exception caught" << endl;

							FILE *ftime = mrpt::system::os::fopen("time_new_kf.txt","wt");
							for (size_t i = 0; i < stats.size(); i++)
								mrpt::system::os::fprintf(ftime, "%.3f %d\n", 1000.0*stats[i].time, stats[i].numberFeatsNew);
							mrpt::system::os::fclose(ftime);
							return;
						}
						if( app_options.debug && confirmedLoopClosure )
						{
							cout << "saving graph at loop closure" << endl;
							rba.save_graph_as_dot( mrpt::format("%s\\graph_at_loopclosure.dot", app_options.out_dir.c_str() ) );
						}

						VERBOSE_LEVEL(1) << "	-------------------------------------------------------" << endl
										 << "		Created KF #" << newKFInfo.kf_id
										 << "| # kf-to-kf edges created:" <<  newKFInfo.created_edge_ids.size()  << endl
										 << "		Optimization error: " << newKFInfo.optimize_results.total_sqr_error_init << " -> " << newKFInfo.optimize_results.total_sqr_error_final << endl
										 << "	-------------------------------------------------------" << endl;
						
						// --------------------------------------------------------------------------------
						// Show 3D view of the resulting map:
						// --------------------------------------------------------------------------------
                
						//	:: get output info
						size_t num_lms = 0;
						if( app_options.verbose_level >= 1 )
						{
							myRBAProblemState &myRBAState = rba.get_rba_state();
							num_lms = myRBAState.unknown_lms.size();
							cout << "RBA Problem info: [" << num_lms << " landmarks]" << endl;
						}
				
						//	:: add the keyframe to the list of KFs
						if( app_options.show3D )
						{
							COpenGLScenePtr & scene = win->get3DSceneAndLock();
							CRenderizablePtr obj    = scene->getByName("srba");
							CSetOfObjectsPtr rba_3d;

							if( obj )
								rba_3d = static_cast<CSetOfObjectsPtr>(obj);
							else
							{
								rba_3d = CSetOfObjects::Create();
								rba_3d->setName("srba");
								scene->insert(rba_3d);
							}
							rba.build_opengl_representation(
								0,              // Root KF,
								opengl_params,  // Rendering options
								rba_3d          // Output scene
								);

							string aux_str = mrpt::format("Query DB Results for candidate KF %d: ", new_kf.m_kfID);
							for(size_t k = 0; k < qResults.size(); ++k )
								aux_str += mrpt::format("%d(%.2f) ", qResults[k].Id, qResults[k].Score);
							win->addTextMessage( 210, 200, aux_str, TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 3);

							//	:: update text
							win->addTextMessage( 210, 160,
								mrpt::format("#KF: %lu | #LM: %lu", keyFrames.size(), num_lms ),
								TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 
								0);

							win->unlockAccess3DScene();
							win->forceRepaint();

							//	:: get last kf pose from 3d representation (it used 'create_complete_spanning_tree' within)
							CRenderizablePtr aux_obj = rba_3d->getByName( mrpt::format("%d",new_kf.m_kfID ).c_str() );
							if( aux_obj )
								new_kf.m_camPose = last_kf_pose = CPose3DRotVec( CPose3D(aux_obj->getPose()) );
							else {
								cout << "no aux_obj" << endl; }
						}
						else
						{
							mySRBA::frameid2pose_map_t  spantree;
							rba.create_complete_spanning_tree(0, spantree, rba.parameters.srba.max_tree_depth );
							for( mySRBA::frameid2pose_map_t::const_iterator itP = spantree.begin(); itP != spantree.end(); ++itP )
							{
								if( itP->first != new_kf.m_kfID ) continue;
								new_kf.m_camPose = last_kf_pose = CPose3DRotVec( itP->second.pose );
							}
						}

						//	:: insert kf into the database
						new_kf.insertIntoDB( db );

						//	:: update the poses
						current_pose				= last_kf_pose + camera_pose_on_robot_rvt;
						incr_pose_from_last_kf		= CPose3DRotVec();							// set the incremental poses to zero

					} // end insert a new keyframe
					else
						keyFrames.resize(keyFrames.size()-1);
				} // end else
            } // end if( voForceNewKf || voForceCheck )
        } // end perform only visual odometry

		count++;

		// check stop conditions
		if( (app_options.max_num_kfs > 0 && keyFrames.size() == app_options.max_num_kfs) || 
			(app_options.to_step != 0 && count >= size_t(app_options.to_step)) )
			end_app = true;

		if( app_options.pause_at_each_iteration )
			mrpt::system::pause();
	} // end while crawlog

	// save kf creation times
	FILE *ftime = mrpt::system::os::fopen("time_new_kf.txt","wt");
	for (size_t i = 0; i < stats.size(); i++)
		mrpt::system::os::fprintf(ftime, "%.3f %d\n", 1000.0*stats[i].time, stats[i].numberFeatsNew);
	mrpt::system::os::fclose(ftime);

	// perform pose-graph
	mrpt::graphs::CNetworkOfPoses3D poseGraph;
	rba.get_global_graphslam_problem(poseGraph);

	mrpt::graphslam::TResultInfoSpaLevMarq out_info;
	mrpt::utils::TParametersDouble extra_params;

	mrpt::graphslam::optimize_graph_spa_levmarq(	
		poseGraph, 
		out_info,
		NULL, /* in_nodes_to_optimize, NULL=all */
		extra_params
		); // run optimization

	// render resulting graph
	mrpt::gui::CDisplayWindow3D win2("Global optimized map",640,480);
	{
		mrpt::opengl::COpenGLScenePtr &scene = win2.get3DSceneAndLock();

		mrpt::utils::TParametersDouble render_params;   // See docs for mrpt::opengl::graph_tools::graph_visualize()
		render_params["show_ID_labels"] = 1;	

		// Get opengl representation of the graph:
		mrpt::opengl::CSetOfObjectsPtr gl_global_map = mrpt::opengl::graph_tools::graph_visualize( poseGraph,render_params );
		scene->insert(gl_global_map);

		win2.unlockAccess3DScene();
		win2.repaint();

		scene->saveToFile("final_global_path.3DScene");

		// save kfs_poses
		{
			FILE *fposes = mrpt::system::os::fopen("out_kf_poses.txt","wt");
			unsigned int kf = 0;
			for( mrpt::graphs::CNetworkOfPoses3D::global_poses_t::iterator it = poseGraph.nodes.begin(); it != poseGraph.nodes.end(); ++it, ++kf )
			{
				CPose3D pose = it->second.getPoseMean();
				mrpt::system::os::fprintf(fposes,"%d %.3f %.3f %.3f %.3f %.3f %.3f\n",
					kf,pose[0],pose[1],pose[2],pose[3],pose[4],pose[5]);
			}
			mrpt::system::os::fclose(fposes);
		}

		// display final map with landmarks
		CDisplayWindow3D final_win("Final global map");
		{
			COpenGLScenePtr & scene = final_win.get3DSceneAndLock();
			// scene->insert(CGridPlaneXY::Create(-100,100,-100,100,0,1));

			opengl_params.span_tree_max_depth			= 1000; // 10*rba.parameters.srba.max_tree_depth;
			opengl_params.draw_unknown_feats_ellipses	= false;
			opengl_params.show_unknown_feats_ids		= false;
			opengl_params.draw_kf_hierarchical			= true;

			CRenderizablePtr obj = scene->getByName("srba");
			CSetOfObjectsPtr rba_3d;

			if( obj )
				rba_3d = static_cast<CSetOfObjectsPtr>(obj);
			else
			{
				rba_3d = CSetOfObjects::Create();
				rba_3d->setName("srba");
				scene->insert(rba_3d);
			}
			rba.build_opengl_representation(
				0,              // Root KF,
				opengl_params,  // Rendering options
				rba_3d          // Output scene
			);
		}
			
		final_win.unlockAccess3DScene();
		final_win.forceRepaint();
		final_win.waitForKey();

		// save time profiler
		rba.get_time_profiler().saveToCSVFile("profiler.csv");

	}

	mrpt::system::os::fclose( fstd );
	mrpt::system::os::fclose( ft );
	mrpt::system::os::fclose( fls );

	if( app_options.show3D ) 
	{
		COpenGLScenePtr & scene = win->get3DSceneAndLock();

        CRenderizablePtr obj = scene->getByName("srba");
        CSetOfObjectsPtr rba_3d;

        if( obj )
            rba_3d = static_cast<CSetOfObjectsPtr>(obj);
        else
        {
            rba_3d = CSetOfObjects::Create();
            rba_3d->setName("srba");
            scene->insert(rba_3d);
        }
        rba.build_opengl_representation(
			0,              // Root KF,
			opengl_params,  // Rendering options
			rba_3d          // Output scene
        );
		
		if( app_options.debug )
			scene->saveToFile( mrpt::format("%s\\final_map.3Dscene", app_options.out_dir.c_str() ) );
		
		win->waitForKey();
	}

	// show final results
	if( !app_options.show3D )
	{
		CDisplayWindow3D final_win("Final global map");

		COpenGLScenePtr & scene = final_win.get3DSceneAndLock();
		scene->insert(CGridPlaneXY::Create(-100,100,-100,100,0,1));

        CRenderizablePtr obj = scene->getByName("srba");
        CSetOfObjectsPtr rba_3d;

        if( obj )
            rba_3d = static_cast<CSetOfObjectsPtr>(obj);
        else
        {
            rba_3d = CSetOfObjects::Create();
            rba_3d->setName("srba");
            scene->insert(rba_3d);
        }
        rba.build_opengl_representation(
			0,              // Root KF,
			opengl_params,  // Rendering options
			rba_3d          // Output scene
        );

		final_win.unlockAccess3DScene();
		final_win.forceRepaint();
		final_win.waitForKey();

		if( app_options.debug )
			scene->saveToFile( mrpt::format("%s\\final_map.3Dscene", app_options.out_dir.c_str() ) );
	}
	
	// save final graph
	if( app_options.debug )
		rba.save_graph_as_dot( mrpt::format("%s\\final_graph.dot", app_options.out_dir.c_str() ) );

} // end-performStereoSLAM

// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------
int main(int argc, char **argv)
{
	try
	{
	    if(argc < 2)
	    {
	        cout << "Use: rba-stereoSLAM iniFile" << endl;
	        return -1;
	    }

		std::string INI_FILENAME(argv[1]);
		ASSERT_FILE_EXISTS_( INI_FILENAME )
		config.setFileName( INI_FILENAME );

		// set vocabulary and create db
		const string VOC_FILENAME = config.read_string("GENERAL","voc_filename","",true);
		ASSERT_FILE_EXISTS_( VOC_FILENAME )
		voc.load( VOC_FILENAME );
		db.setVocabulary( voc, true, 5 );

	    // get application parameters
		app_options.loadFromConfigFile( config );				// general app options
		stereo_slam_options.loadFromConfigFile( config );		// specific stereo-slam options

		if(app_options.from_step != 0 && app_options.to_step != 0 && app_options.to_step < app_options.from_step )
			THROW_EXCEPTION( "Parameter 'toStep' is lower than 'fromStep'" );

	    app_options.dumpToConsole();
		stereo_slam_options.dumpToConsole();
		performStereoSLAM();
	    cout << "performStereoSLAM done!" << endl;

		return 0;
	} catch (exception &e)
	{
		cout << "MRPT exception caught: " << e.what() << endl;
		return -1;
	}
	catch (...)
	{
		printf("Untyped exception!!");
		return -1;
	}
}
