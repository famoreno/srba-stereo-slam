#include "CSRBAStereoSLAMEstimator.h"
#include <mrpt/graphslam.h>				// For global map recovery only
#include <mrpt/opengl/graph_tools.h>	// To render the global map

extern TGeneralOptions	general_options; // global variable

void CSRBAStereoSLAMEstimator::performStereoSLAM()
{
	mrpt::obs::CObservationPtr obs;
	bool end_app = false;
	size_t count = 0, iterations_since_last_vo_check = 0;

	FILE *ft = NULL, *fls = NULL, *fstd = NULL;
	if( general_options.debug )
	{
		fstd = system::os::fopen( "std.txt", "wt" );
		ft = system::os::fopen( "da.txt", "wt" );
		fls = system::os::fopen( "ls.txt", "wt" );
	}
	
	// main loop
	while( (obs = m_myCam.getNextFrame()).present() && !end_app )
    {
		if( general_options.show3D )
		{
			m_win->get3DSceneAndLock();
			m_win->addTextMessage( 210, 180, mrpt::format("#Frame: %lu", count ), TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 1);
			m_win->unlockAccess3DScene();
			m_win->repaint();
		}
		
		//	load state and/or skip until desired frame
		/** /
		if( general_options.load_state_from_file && start_iteration > 0 && count < start_iteration )
		{
			cout << ".";
			count++;
			continue;
		}
		else
		{
			if( count < size_t(general_options.from_step) )
			{
				if( count == 0 ) cout << "Skipping frame until " << general_options.from_step << " ";
				cout << ".";
				count++;
				continue;
			}
		}
		/**/
		cout << endl << endl;
		cout << " >> Frame # " << count << endl;
		
		//  :: get the images
        CObservationStereoImagesPtr stImgs = static_cast<CObservationStereoImagesPtr>(obs);
		
		// ----------------------------------------
		//  :: first iteration
		// ----------------------------------------
        if( m_keyframes.size() == 0 )
        {
			m_keyframes.push_back( CStereoSLAMKF() );			// create new KF
			CStereoSLAMKF & new_kf = m_keyframes.back();		// reference to new element
			new_kf.setKFID( m_last_kf_ID++ );

			// DEBUG -----------------------------------------
			if( general_options.debug )										// debug mode: save images, feats positions and stereo matches
			{
				//	:: save images
				stImgs->imageLeft.saveToFile( mrpt::format("%s\\image_left_kf%04d.jpg", general_options.out_dir.c_str(), new_kf.m_kf_ID) ); 
				stImgs->imageRight.saveToFile( mrpt::format("%s\\image_right_kf%04d.jpg", general_options.out_dir.c_str(), new_kf.m_kf_ID) ); 
			}
			// -----------------------------------------------			
			
			// gui stuff
			if( general_options.show3D )
			{
				COpenGLScenePtr & scene = m_win->get3DSceneAndLock();

				// :: show images in viewport
				scene->getViewport("image_left")->setImageView( stImgs->imageLeft );
				scene->getViewport("image_right")->setImageView( stImgs->imageRight );

				m_win->unlockAccess3DScene();
				m_win->repaint();
			}
			
			// visual odometry process and copy information from visual odometer engine
			m_odom_request.stereo_imgs = stImgs;
			m_voEngine.processNewImagePair( m_odom_request, m_odom_result );
			m_voEngine.setThisFrameAsKF();					// <- create a virtual KF in the visual odometer to keep track of the IDs
			new_kf.getDataFromVOEngine( m_voEngine );		// <- this will copy ORB features, ORB descriptors and stereo matches and IDs
			m_last_match_ID = new_kf.m_matches_ID.back();	// <- set maximum ID match

			// insert new kf into bow databse
			m_bow_manager.insertIntoDB( new_kf );
			
			// ids management
			// m_voEngine.resetIds();	// <-- ??
			
			const size_t nMatches = new_kf.m_matches.size();				// number of matches in this keyframe
			VERBOSE_LEVEL(1) << "	# feats (L/R) = " << new_kf.m_keypoints_left.size() 
							 << "/" << new_kf.m_keypoints_right.size() 
							 << " -- # matches = " << nMatches << endl;
							 
			if( general_options.verbose_level >= 2 ) 
				new_kf.dumpToConsole();

			if( general_options.debug )										// debug mode: save images, feats positions and stereo matches
				new_kf.saveInfoToFiles();
				
            // insert KF data into SRBA engine
			mySRBA::TNewKeyFrameInfo       newKFInfo;
            mySRBA::new_kf_observations_t  listObs;
            mySRBA::new_kf_observation_t   obsField;
				
			obsField.is_fixed                   = false;	// landmarks have unknown relative positions (i.e. treat them as unknowns to be estimated)
            obsField.is_unknown_with_init_val   = true;		// we have a guess of the initial LM position

			listObs.resize( nMatches );

            // fill observation fields
            for( size_t m = 0; m < nMatches; ++m )
            {
				const size_t id1 = new_kf.m_matches[m].queryIdx;
                const size_t id2 = new_kf.m_matches[m].trainIdx;

                const KeyPoint & kpLeft  = new_kf.m_keypoints_left[id1];
                const KeyPoint & kpRight = new_kf.m_keypoints_right[id2];

                obsField.obs.feat_id            = new_kf.m_matches_ID[m];
                obsField.obs.obs_data.left_px   = TPixelCoordf( kpLeft.pt.x,  kpLeft.pt.y );
                obsField.obs.obs_data.right_px  = TPixelCoordf( kpRight.pt.x, kpRight.pt.y );

				// initial positions of the landmark
				obsField.setRelPos( srba_options.camera_pose_on_robot_rvt + projectMatchTo3D( kpLeft.pt.x, kpLeft.pt.y, kpRight.pt.x, srba_options.stereo_camera ) );

                listObs[m] = obsField;
            } // end for
			
			// insert KF into the SRBA framework
            MRPT_TRY_START
			m_time_logger_define_kf.enter("define_kf");
			rba.define_new_keyframe( listObs,				// list of observations
                                     newKFInfo,				// keyframe info
                                     false );				// not optimize the first time
			m_time_logger_define_kf.leave("define_kf");
			m_stats.push_back( TStatsSRBA( m_time_logger_define_kf.getMeanTime("define_kf"), listObs.size() ) );
			m_time_logger_define_kf.clear();
			MRPT_TRY_END
			
			if( general_options.verbose_level >= 2 ) 
				rba.dumpLocalmapCenters();			// <-- ??
				
            VERBOSE_LEVEL(1)	<< "-------------------------------------------------------" << endl
								<< "   Created KF #" << newKFInfo.kf_id
								<< " | # kf-to-kf edges created: " <<  newKFInfo.created_edge_ids.size()  << endl
								<< "   Optimization error: " << newKFInfo.optimize_results.total_sqr_error_init << " -> " << newKFInfo.optimize_results.total_sqr_error_final << endl
								<< "-------------------------------------------------------" << endl;

			// set this pose as the last kf 'GLOBAL' pose
			m_last_kf_pose = new_kf.m_camera_pose;		// <-- ??
			
			// visualization (quite slow) --> consider a new thread for this
			if( general_options.show3D )
			{
                COpenGLScenePtr & scene = m_win->get3DSceneAndLock();

                // stereo camera
				CSetOfObjectsPtr cam = static_cast<CSetOfObjectsPtr>(scene->getByName("camera"));
                cam->setPose( CPose3D(m_current_pose) );

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
				show_kf_numbers( scene, m_keyframes.size(), QueryResults(), 0 );

				m_win->unlockAccess3DScene();
				m_win->repaint();
			} // end-if-show3D
			
			iterations_since_last_vo_check++;
		}
		// ----------------------------------------
		//  :: next iterations
		// ----------------------------------------
		else
        {
			/** /
			if( general_options.save_state_to_file && count == general_options.save_at_iteration )
			{
				VERBOSE_LEVEL(1) << " Saving state to file at iteration " << general_options.save_at_iteration << " ... ";
				
				//	:: save state and exit
				saveApplicationState(
						general_options.state_file, 
						count,
						m_last_num_tracked_feats,
						m_current_pose,
						m_last_kf_pose,
						m_incr_pose_from_last_kf,
						m_incr_pose_from_last_check,
						m_keyframes,
						srba_options,
						m_voEngine,
						general_options,
						db );
				
				VERBOSE_LEVEL(1) << " done." << endl << "Exiting" << endl;
				
				return;
			}
			/**/
			VERBOSE_LEVEL(1) << " --> Looking for KF#" << m_last_kf_ID << endl; 
			
			// for the gui
			CImage imL(UNINITIALIZED_IMAGE), imR(UNINITIALIZED_IMAGE);
			if( general_options.show3D )
			{
				imL.copyFromForceLoad(stImgs->imageLeft);
				imR.copyFromForceLoad(stImgs->imageRight);
			}
			
            // peform VO -- track the position of the last keypoints
			m_odom_request.stereo_imgs            = stImgs;	// just copy the pointer
            m_odom_request.use_precomputed_data   = false;
            m_odom_request.repeat					= false;
			
			const size_t octave = 0;
			do
			{
				// perform VO
				VERBOSE_LEVEL(2) << " Processing new stereo pair ... (tstamp: " << stImgs->timestamp <<  ": " << stImgs->imageLeft.getWidth() << "x" << stImgs->imageLeft.getHeight() << ")" << endl;
				m_voEngine.processNewImagePair( m_odom_request, m_odom_result ); // WARNING --> after this call, 'stImgs' contains the previous images; do not use them!
				VERBOSE_LEVEL(2) << "	Detected keypoints: Left(" << m_odom_result.detected_feats[octave].first << ") and Right(" << m_odom_result.detected_feats[octave].second << ")" << endl
								 << "	Matches found: " << m_odom_result.stereo_matches[octave] << endl;
				
				if( srba_options.orb_adaptive_fast_th )
				{
					if( int(m_odom_result.stereo_matches[octave]) < srba_options.adaptive_th_min_matches )
					{	
						if( !m_voEngine.isFASTThMin() )
						{
							m_voEngine.setFASTThreshold( std::max(0,m_voEngine.getFASTThreshold()-10) );
							m_odom_request.repeat = true;
							VERBOSE_LEVEL(0) << "Number of stereo matches is too low! (" << m_odom_result.stereo_matches[octave] << ") Repeat detection with a lower FAST threshold: " << m_voEngine.getFASTThreshold() << endl;
						}
						else if( !m_voEngine.isORBThMax() )
						{
							// images contain few keypoints, allow worse matches --> increase ORB threshold
							m_voEngine.setORBThreshold( m_voEngine.getORBThreshold()+10 );
							m_odom_request.repeat = true;
							VERBOSE_LEVEL(0) << "Number of stereo matches is still too low! (" << m_odom_result.stereo_matches[octave] << ") Repeat detection with a higher ORB threshold: " << m_voEngine.getORBThreshold() << endl;
						}
						else if( m_odom_result.stereo_matches[octave] >= 8 )
						{
							// we have reached the limits but we have at least the minimum set of matches --> try to continue
						}
						else
						{
							THROW_EXCEPTION( "The number of found matches is less than the minimum. Aborting" )
						}
					}
					else if(m_odom_result.stereo_matches[octave] < srba_options.adaptive_th_min_matches*1.2/*srba_options.n_feats*0.25*/ )
					{	
						// number of stereo matches is low --> reduce FAST threshold for future but continue
						VERBOSE_LEVEL(2) << "Number of stereo matches is low! (" << m_odom_result.stereo_matches[octave] << ") Reduce FAST threshold for the next iteration" << endl;
						if( !m_voEngine.isFASTThMin() )
							m_voEngine.setFASTThreshold( std::max(0,m_voEngine.getFASTThreshold()-5) );
						else if( !m_voEngine.isORBThMax() )
							m_voEngine.setORBThreshold( m_voEngine.getORBThreshold()+5 );
						m_odom_request.repeat = false;
					}
					else
					{	
						// we are good --> increase the FAST threshold
						m_voEngine.setFASTThreshold( std::min(srba_options.detect_fast_th,m_voEngine.getFASTThreshold()+5) );
						m_voEngine.resetORBThreshold();
						m_odom_request.repeat = false;
					}
				} // end-if
			} while( m_odom_request.repeat ); // end-while
			
			// check vo result
			if( !m_odom_result.valid )
			{
				VERBOSE_LEVEL(1) << "	[Warning - VO Engine] -- Not a valid result! Skipping this frame." << endl;
				count++;
				continue;
			} // end-if

			// update current 'estimated' pose of the camera:
			// 'incr_pose' = pose of the current stereo frame wrt the previous one	<-- ?? Check this
			CPose3DRotVec incr_pose( m_odom_result.outPose );
            m_current_pose					+= incr_pose;
			m_incr_pose_from_last_kf		+= incr_pose;
			m_incr_pose_from_last_check		+= incr_pose;
			
			// get number of tracked feats from last KF
			/** /
			size_t tracked_feats_from_last_KF = 0;
			const vector<size_t> & c_ids = m_voEngine.getRefCurrentIDs(0); // octave
			for( size_t i = 0; i < c_ids.size(); ++i )
			{
				if( c_ids[i] <= m_last_match_ID )
					tracked_feats_from_last_KF++;
			}
			/**/

			// update camera position in the visualization
			if( general_options.show3D )
			{
                COpenGLScenePtr & scene = m_win->get3DSceneAndLock();
                CSetOfObjectsPtr cam = static_cast<CSetOfObjectsPtr>( scene->getByName("camera") );
                cam->setPose( CPose3D(m_current_pose) );

				// image viewport	// <-- ?? Show points
				scene->getViewport("image_left")->setImageView( imL );
				scene->getViewport("image_right")->setImageView( imR );

                m_win->unlockAccess3DScene();
				m_win->repaint();
            } // end-if-gui
			
			// show info
			VERBOSE_LEVEL(1)	<< "	[VO] # tracked features from last frame: " << m_odom_result.tracked_feats_from_last_frame << endl
								<< "	[VO] # tracked features from last KF: " << m_odom_result.tracked_feats_from_last_KF << endl;
			
			VERBOSE_LEVEL(2)	<< "	[VO] Incremental Pose: " << incr_pose << endl
								<< "	[VO] Incremental Pose from last KF: " << m_incr_pose_from_last_kf << endl
								<< "	[VO] Current pose: " << m_current_pose << endl;
			
			// check if this frame is far enough from the last KF to force it to be a new KF
			const double incTranslationKf = m_incr_pose_from_last_kf.m_coords.norm();
			const double incRotationKf = m_incr_pose_from_last_kf.m_rotvec.norm();
			bool voForceNewKf = incTranslationKf > m_max_translation_limit || incRotationKf > DEG2RAD( m_max_rotation_limit );		// it may be modified later

			// show info
			VERBOSE_LEVEL(1) << "	[VO Check] -- Last KF distance: " << voForceNewKf << " (" << incTranslationKf << " m.," << RAD2DEG(incRotationKf) << "deg) vs Th: (" << m_max_translation_limit << " m.," << m_max_rotation_limit << "deg)" << endl;

			// check if visual odometer has lost too many features (force a new KF check)
			const bool voForceCheckTracking =
				srba_options.vo_id_tracking_th == 0 ? 
				false : 
				int(m_odom_result.tracked_feats_from_last_KF) < srba_options.vo_id_tracking_th;
				
			// check if this frame is far enough from the last check to be a candidate for a new keyframe (and the number of feats we've tracked is low)
			const double incr_translation	= m_incr_pose_from_last_check.m_coords.norm();
			const double incr_rotation		= m_incr_pose_from_last_check.m_rotvec.norm();
			const bool voForceCheckDistance = 
				incr_translation > m_max_translation || 
				incr_rotation > DEG2RAD( m_max_rotation );		
			
			const bool voForceCheck = voForceCheckTracking || voForceCheckDistance;	

			// show info
			VERBOSE_LEVEL(1) << "	[VO Check] -- Check distance: " << voForceCheckDistance << " (" << incr_translation << " m.," << RAD2DEG(incr_rotation) << "deg) vs Th: (" << m_max_translation << " m.," << m_max_rotation << "deg)" << endl
							 << "	[VO Check] -- Feature tracking: " << voForceCheckTracking << " (" << m_odom_result.tracked_feats_from_last_KF << ") vs Th: (" << srba_options.vo_id_tracking_th << ")" << endl;
			
			bool insertNewKf = false;
			if( voForceNewKf || voForceCheck )
			{	
				VERBOSE_LEVEL(1) << "	[VO Check] -- Visual odometry asked for CHECKING for a new keyframe." << endl;
				
				if( voForceCheckTracking )
				{
					// reset visual odometer ids to avoid consecutive checks because of this
					// this will define previous frame as a virtual KF for the visual odometry system so that
					// it can start a new record of tracked matches from last KF
					m_voEngine.resetIds();
				}
				
				// clear this pose
				m_incr_pose_from_last_check = CPose3DRotVec();

				// get current number of KFs (before inserting a new one)
				const size_t num_kfs = m_keyframes.size();

				// create a temporary keyframe (will be deleted if necessary)
				m_keyframes.push_back( CStereoSLAMKF() );
				CStereoSLAMKF & new_kf = m_keyframes.back();
				new_kf.getDataFromVOEngine( m_voEngine );

				// set the candidate KF ID (it will be discarded if needed)
				new_kf.setKFID( m_last_kf_ID );

				const size_t nMatches = new_kf.m_matches.size();		// number of stereo matches in this KF
				//new_kf.m_matches_ID.resize( nMatches, 0 );				// matches Ids will be filled later if needed
				
				// query db
				QueryResults qResults;
				rba.get_time_profiler().enter("queryDB");
				m_bow_manager.queryDB( new_kf, qResults, 4 );
				rba.get_time_profiler().leave("queryDB");

				ASSERTDEB_( qResults.size() > 0 )

				// update query score
				// --> probably unused, remove
				double qScoreTh = srba_options.query_score_th != 0 ? 
					srba_options.query_score_th :
					updateQueryScoreThreshold( m_last_num_tracked_feats );
	
				//if( qResults[0].Score > 0.5 )		// this KF is too similar to the most similar one
				//{
				//	VERBOSE_LEVEL(2) << "qResult[0] is significantly large --> skip the complete test." << endl;
				//	if( voForceCheckTracking )
				//		// m_reset_voEngine(); // set current frame IDs (in VO) 
				//}
				//else
				//{
				// analyse query results
				bool confirmedLoopClosure = false;
				TLoopClosureInfo similar_kfs_info;
				rba.get_time_profiler().enter("get_similar_kfs");
				const bool potentialLoopClosure = m_get_similar_kfs( new_kf.m_kf_ID, qResults, similar_kfs_info );
				rba.get_time_profiler().leave("get_similar_kfs");
				
				// result is too low, force the creation of a new KF
				if( qResults[0].Score < 0.05 )
					insertNewKf = true;
					
				VERBOSE_LEVEL(2) << "	Performing data association" << endl;
					
				// perform data association
				TVectorKfsDaInfo daInfo;
				rba.get_time_profiler().enter("performDA");
				m_data_association( 					// this includes: ORB inter-frame matching, fundamental matrix filter, change in pose filter
							new_kf,						// this KF
							similar_kfs_info,			// information about the similar kf
							daInfo );						// output data association info
				rba.get_time_profiler().leave("performDA");
					
				const size_t numberSimilarKfs = daInfo.size();	// should be equal to 'similar_kfs_info.size()'
				ASSERT_( numberSimilarKfs > 0 )
					
				rba.get_time_profiler().enter("confirmLC");
					
				// check data association results
				//		- variable 'daInfo' has at least size 1 (the best result from the DB query), but it may contain other DA with KFs that are similar to the current one.
				//		- order all the keyframes according to the number of common observed features.
				vector<size_t> sortedIndices( numberSimilarKfs );
				for( size_t i = 0; i < numberSimilarKfs; i++ ) sortedIndices[i] = i;
				if( numberSimilarKfs > 1 )
					std::sort( sortedIndices.begin(), sortedIndices.end(), DATrackedSorter(daInfo) );
						
				// VERBOSE ------------------------------------------------------
				VERBOSE_LEVEL(1) << "	:: Tracked features" << endl;
				if( general_options.verbose_level >= 1 )
					for( vector<size_t>::iterator it = sortedIndices.begin(); it != sortedIndices.end(); ++it )
						cout << "		with " << daInfo[*it].kf_idx << " -> " << daInfo[*it].tracked_matches << " tracked features." << endl;
				// --------------------------------------------------------------
					
				const size_t highestNumberTrackedFeats = daInfo[sortedIndices[0]].tracked_matches;
				m_last_num_tracked_feats = highestNumberTrackedFeats;

				if( voForceNewKf ) 
				{
					VERBOSE_LEVEL(1) << "	[VO Check] -- Visual odometry FORCED the insertion of a new keyframe." << endl;
					insertNewKf = true;
				}

				if( potentialLoopClosure )					
				{
					VERBOSE_LEVEL(2) << "POTENTIAL LOOP CLOSURE" << endl; 
			
					//	:: loop closure is confirmed if the number of tracked feats with the old KF is over a threshold
					size_t loopClosureIdx = 0;
					for( size_t i = 0; !confirmedLoopClosure && i < daInfo.size(); i++ )
					{
						confirmedLoopClosure = 
							daInfo[i].kf_idx == similar_kfs_info.lc_id && 
							daInfo[i].tracked_matches > 0.5*highestNumberTrackedFeats;

						if( confirmedLoopClosure ) loopClosureIdx = i;
					} // end-for
					
					if( confirmedLoopClosure )
					{
						rba.loopClosureDetected();
						rba.setLoopClosureOldID( similar_kfs_info.lc_id );
						insertNewKf = true;

						//	:: give priority to the old keyframe
						for( size_t i = 0; i < sortedIndices.size(); ++i )
						{
							if( sortedIndices[i] == loopClosureIdx )
							{
								sortedIndices.erase( sortedIndices.begin()+i );
								break;
							}
						} // end-for
						sortedIndices.insert( sortedIndices.begin(), loopClosureIdx );

						VERBOSE_LEVEL(2) << "	Loop closure confirmed" << endl;
					}
					else 
					{
						VERBOSE_LEVEL(2) << "	Loop closure NOT confirmed" << endl;
						rba.loopClosureDetected(false);
					}
				} // end-if-potential-loop-closure
				else
				{
					if( highestNumberTrackedFeats < srba_options.updated_matches_th )
					{
						VERBOSE_LEVEL(1) << "	:: Tracked features below the threshold (" << srba_options.updated_matches_th << ") ==> insert a new KF" << endl;
						insertNewKf = true;
					}
					else
					{
						// the number of tracked feats is still quite big, skip inserting a new kf
						const size_t olimit = srba_options.updated_matches_th+srba_options.up_matches_th_plus;
						if( highestNumberTrackedFeats <= olimit )
						{
							// update the dynamic threshold for inserting a new keyframe (decreasing linearly when the number of tracked features is below the threshold 'updated_matches_th'+'up_matches_th_plus')
							m_max_translation = updateTranslationThreshold( highestNumberTrackedFeats-srba_options.updated_matches_th, srba_options.up_matches_th_plus );
							m_max_rotation = updateRotationThreshold( highestNumberTrackedFeats, olimit );

							VERBOSE_LEVEL(2) << "	New Translation/Rotation thresholds: " << m_max_translation << " m./" << m_max_rotation << " deg" << endl;
						} // end-if
					}
					rba.loopClosureDetected(false);
				} // end-else-potential-loop-closure
				rba.get_time_profiler().leave("confirmLC");
					
				// gui stuff
				if( general_options.show3D )
				{
					COpenGLScenePtr & scene = m_win->get3DSceneAndLock();
					show_kf_numbers( scene, num_kfs/*m_keyframes.size()*/, qResults, qScoreTh );
					m_win->unlockAccess3DScene();
					m_win->forceRepaint();
				}
					
				// ***************************************
				//	:: INSERT A NEW KEYFRAME?
				//	**************************************
				if( !insertNewKf )
				{
					// undo the insertion
					m_keyframes.resize(m_keyframes.size()-1);
				}
				else
				{
					// DEBUG ---------------------------------------------------
					FILE *f = NULL;
					if( general_options.debug )
						f = mrpt::system::os::fopen( GENERATE_NAME_WITH_KF_OUT( da_dist, new_kf ), "wt" );
					// ---------------------------------------------------------

					// set ids for the current keyframe matches
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
								const TKeyFrameID & other_match_id = m_keyframes[da_info.kf_idx/*other_kf_idx*/].m_matches_ID[other_match_idx];
									
								if( foundIds.find(other_match_id) != foundIds.end() )
								{
									VERBOSE_LEVEL(2) << "	Feature tracked more than once: kf#" << new_kf.m_kf_ID << "-id:" << other_match_id << endl;
									// [TODO] check the distance and keep the best match, by now: keep the first one
									break;
								}
								foundIds.insert(other_match_id);

								new_kf.m_matches_ID[m] = other_match_id;
								vectorNumberTrackedFeats[sortedIndices[k]]++;
								numberTrackedFeats++;
								tracked = true;

								// DEBUG ---------------------------------------------------
								if( general_options.debug )
									fprintf(f,"%2.f\n",daInfo[sortedIndices[k]].tracking_info[m].second);
								// ---------------------------------------------------------

							} // end-if
						} // end-for
						if( !tracked )
						{	// new feature
							new_kf.m_matches_ID[m] = ++m_last_match_ID;
							numberNewFeats++;

							// DEBUG ---------------------------------------------------
							if( general_options.debug )
								fprintf(f,"0.00\n");
							// ---------------------------------------------------------
						} // end-if
					} // end-for

					//	:: set IDs in the visual odometry engine
					m_voEngine.resetIds();
					// m_voEngine.setIds( new_kf.m_matches_ID );
					// m_voEngine.setMaxMatchID( m_last_match_ID+1 );

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
					if( general_options.debug )
						mrpt::system::os::fclose(f);
					// ---------------------------------------------------------

					// VERBOSE -------------------------------------------------
					VERBOSE_LEVEL(1)	<< "	Real number of similar kfs: " << realNumberSimilarKfs << "/" << numberSimilarKfs << endl;
					VERBOSE_LEVEL(1)	<< "	Features ----------------------------------" << endl;
					for( size_t k = 0; k < numberSimilarKfs; ++k )
					{
						VERBOSE_LEVEL(1) << "	Tracked with [#" << daInfo[sortedIndices[k]].kf_idx << "]: " << vectorNumberTrackedFeats[sortedIndices[k]] << endl;
						if( general_options.debug )
							mrpt::system::os::fprintf(ft, "%d %d %d\n", new_kf.m_kf_ID, daInfo[sortedIndices[k]].kf_idx, vectorNumberTrackedFeats[sortedIndices[k]]);
					}
					VERBOSE_LEVEL(1)	<< "	New features:       " << numberNewFeats << endl
										<< "	-------------------------------------------" << endl
										<< "	TOTAL:              " << numberTrackedFeats + numberNewFeats << endl;
					// ---------------------------------------------------------

					m_last_num_tracked_feats = UNINITIALIZED_TRACKED_NUMBER;

					VERBOSE_LEVEL(1) << "Inserting new Keyframe " << endl;
					m_last_kf_ID++;	// prepare KF id for the next one

					//	:: restore the original thresholds for inserting a new KF
					m_max_translation = srba_options.max_translation;
					m_max_rotation = srba_options.max_rotation;

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
					if( general_options.debug && confirmedLoopClosure )
					{
						flc = mrpt::system::os::fopen( mrpt::format("%s\\loop_closure_info_%d.txt", general_options.out_dir.c_str(), count).c_str() ,"wt" );
						okf = &(m_keyframes[qResults[0].Id]);
					}
					//----------------------------------------------------

					//	:: Create the input for the SRBA
					size_t outCounter = 0;
					vector<size_t>::iterator it_id = new_kf.m_matches_ID.begin();
					vector<cv::DMatch>::iterator it_ma = new_kf.m_matches.begin();
					while( it_id != new_kf.m_matches_ID.end() )
					{
						// add info to the new KF input
						obsField.obs.feat_id			= *it_id;
						KeyPoint & kpLeft				= new_kf.m_keypoints_left[it_ma->queryIdx];
						KeyPoint & kpRight				= new_kf.m_keypoints_right[it_ma->trainIdx];

						obsField.obs.obs_data.left_px   = TPixelCoordf( kpLeft.pt.x,  kpLeft.pt.y );
						obsField.obs.obs_data.right_px  = TPixelCoordf( kpRight.pt.x, kpRight.pt.y );
						obsField.setRelPos( srba_options.camera_pose_on_robot_rvt + projectMatchTo3D( kpLeft.pt.x, kpLeft.pt.y, kpRight.pt.x, srba_options.stereo_camera ) );
						listObs[obs_idx++]				= obsField;

						// increment iterators
						++it_id;
						++it_ma;

						if( general_options.debug && confirmedLoopClosure )
						{	
							KeyPoint *olfeat = NULL, *orfeat = NULL;
							for( size_t mm = 0; mm < okf->m_matches_ID.size(); ++mm )
							{
								if( obsField.obs.feat_id != okf->m_matches_ID[mm] )
									continue;
								
								olfeat = & (okf->m_keypoints_left[  okf->m_matches[mm].queryIdx ]);
								orfeat = & (okf->m_keypoints_right[ okf->m_matches[mm].trainIdx ]);
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
					// DEBUG --------------------------------------------
					if( general_options.debug )
					{	//	:: debug: save current images and the new KF info
						imL.saveToFile(mrpt::format("%s\\image_left_kf%04d.jpg", general_options.out_dir.c_str(), new_kf.m_kf_ID));
						imR.saveToFile(mrpt::format("%s\\image_right_kf%04d.jpg", general_options.out_dir.c_str(), new_kf.m_kf_ID));
						new_kf.saveInfoToFiles();
					}
						
					// --------------------------------------------------
					if( general_options.debug && confirmedLoopClosure )
					{
						mrpt::system::os::fclose(flc);
						okf = NULL;
					}

					//  :: dump all the content of the new KF to the console
					if( general_options.verbose_level >= 2 )
						new_kf.dumpToConsole();

					// DEBUG -------------------------------------------------------------
					if( general_options.debug )
					{	//	:: debug: save current data association to a file (equals the input of the srba-slam)
						ofstream fstr( mrpt::format( "%s\\da_info_%04d.txt", general_options.out_dir.c_str(), new_kf.m_kf_ID ).c_str() );
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
						CPose3D inputPose;
						if( srba_options.use_initial_pose )
							inputPose = CPose3D(m_incr_pose_from_last_kf);
						else
							inputPose = CPose3D();

						rba.setInitialKFPose( inputPose );

						VERBOSE_LEVEL(1) << "	Inserting " << listObs.size() << " observations in srba-slam engine" << endl;
						m_time_logger_define_kf.enter("define_kf");
						rba.define_new_keyframe( listObs,
													newKFInfo,
													true );
						m_time_logger_define_kf.leave("define_kf");
						m_stats.push_back( TStatsSRBA( m_time_logger_define_kf.getMeanTime("define_kf"), listObs.size() ) );
						m_time_logger_define_kf.clear();
						VERBOSE_LEVEL(2) << "inserted stat #" << m_stats.size() << endl;
						if( general_options.debug ) 
							mrpt::system::os::fprintf(fls,"%d %.4f\n",newKFInfo.kf_id,newKFInfo.optimize_results_stg1.obs_rmse);
					}
					catch (exception& e)
					{
						cout << "Standard exception: " << e.what() << endl;
					}
					catch (...)
					{
						cout << "EXCEPTION" << endl;
							
						if( general_options.debug )
							rba.save_graph_as_dot( mrpt::format("%s\\graph_at_exception.dot", general_options.out_dir.c_str() ) );
							
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

						if( general_options.debug )
							scene->saveToFile( mrpt::format("%s\\exception_map.3Dscene", general_options.out_dir.c_str() ) );						
							
						cout << "exception caught" << endl;

						FILE *ftime = mrpt::system::os::fopen("time_new_kf.txt","wt");
						for (size_t i = 0; i < m_stats.size(); i++)
							mrpt::system::os::fprintf(ftime, "%.3f %d\n", 1000.0*m_stats[i].time, m_stats[i].numberFeatsNew);
						mrpt::system::os::fclose(ftime);
						return;
					}
					if( general_options.debug && confirmedLoopClosure )
					{
						cout << "saving graph at loop closure" << endl;
						rba.save_graph_as_dot( mrpt::format("%s\\graph_at_loopclosure.dot", general_options.out_dir.c_str() ) );
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
					if( general_options.verbose_level >= 1 )
					{
						myRBAProblemState &myRBAState = rba.get_rba_state();
						num_lms = myRBAState.unknown_lms.size();
						cout << "RBA Problem info: [" << num_lms << " landmarks]" << endl;
					}
				
					//	:: add the keyframe to the list of KFs
					if( general_options.show3D )
					{
						COpenGLScenePtr & scene = m_win->get3DSceneAndLock();
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

						string aux_str = mrpt::format("Query DB Results for candidate KF %d: ", new_kf.m_kf_ID);
						for(size_t k = 0; k < qResults.size(); ++k )
							aux_str += mrpt::format("%d(%.2f) ", qResults[k].Id, qResults[k].Score);
						m_win->addTextMessage( 210, 200, aux_str, TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 3);

						//	:: update text
						m_win->addTextMessage( 210, 160,
							mrpt::format("#KF: %lu | #LM: %lu", m_keyframes.size(), num_lms ),
							TColorf(1,1,1),"mono",10, mrpt::opengl::FILL, 
							0);

						m_win->unlockAccess3DScene();
						m_win->forceRepaint();

						//	:: get last kf pose from 3d representation (it used 'create_complete_spanning_tree' within)
						CRenderizablePtr aux_obj = rba_3d->getByName( mrpt::format("%d",new_kf.m_kf_ID ).c_str() );
						if( aux_obj )
							new_kf.m_camera_pose = m_last_kf_pose = CPose3DRotVec( CPose3D(aux_obj->getPose()) );
						else {
							cout << "no aux_obj" << endl; }
					}
					else
					{
						mySRBA::frameid2pose_map_t  spantree;
						rba.create_complete_spanning_tree(0, spantree, rba.parameters.srba.max_tree_depth );
						for( mySRBA::frameid2pose_map_t::const_iterator itP = spantree.begin(); itP != spantree.end(); ++itP )
						{
							if( itP->first != new_kf.m_kf_ID ) continue;
							new_kf.m_camera_pose = m_last_kf_pose = CPose3DRotVec( itP->second.pose );
						}
					}

					//	:: insert kf into the database
					m_bow_manager.insertIntoDB(new_kf);

					//	:: update the poses
					m_current_pose				= m_last_kf_pose + srba_options.camera_pose_on_robot_rvt;
					m_incr_pose_from_last_kf		= CPose3DRotVec();							// set the incremental poses to zero
				} // end insert a new keyframe
			} // end-if-force-check
		} // end-else-first-iteration		
		count ++;
		
		// check stop conditions
		if( (general_options.max_num_kfs > 0 && m_keyframes.size() == general_options.max_num_kfs) || 
			(general_options.to_step != 0 && count >= size_t(general_options.to_step)) )
			end_app = true;
		
		if( general_options.pause_at_each_iteration )
			mrpt::system::pause();

	} // end-while
	
	// save kf creation times
	FILE *ftime = mrpt::system::os::fopen("time_new_kf.txt","wt");
	for (size_t i = 0; i < m_stats.size(); i++)
		mrpt::system::os::fprintf(ftime, "%.3f %d\n", 1000.0*m_stats[i].time, m_stats[i].numberFeatsNew);
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

			opengl_params.span_tree_max_depth			= 1000;
			opengl_params.draw_unknown_feats_ellipses	= false;
			opengl_params.show_unknown_feats_ids		= false;
			opengl_params.draw_kf_hierarchical			= false;

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

	if( general_options.debug )
	{
		mrpt::system::os::fclose( fstd );
		mrpt::system::os::fclose( ft );
		mrpt::system::os::fclose( fls );
	}

	if( general_options.show3D ) 
	{
		COpenGLScenePtr & scene = m_win->get3DSceneAndLock();

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
		
		if( general_options.debug )
			scene->saveToFile( mrpt::format("%s\\final_map.3Dscene", general_options.out_dir.c_str() ) );
		
		m_win->waitForKey();
	}

	// show final results
	if( !general_options.show3D )
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

		if( general_options.debug )
			scene->saveToFile( mrpt::format("%s\\final_map.3Dscene", general_options.out_dir.c_str() ) );
	}
	
	// save final graph
	if( general_options.debug )
		rba.save_graph_as_dot( mrpt::format("%s\\final_graph.dot", general_options.out_dir.c_str() ) );
		
} // end--performStereoSLAM

void CSRBAStereoSLAMEstimator::initialize( const CConfigFile & config )
{
	// bag of words
 	m_bow_manager.loadVocabularyFromConfigFile(config, "SRBA_GENERAL", "voc_filename" ); // will raise an exception if not file not found
	
	// stereo camera
	vector<double> p(6);
	config.read_vector( "GENERAL", "camera_pose_on_robot", vector<double>(6,0), p, false );
	const CPose3D camera_pose_on_robot( p[0],p[1],p[2],DEG2RAD(p[3]),DEG2RAD(p[4]),DEG2RAD(p[5]) );	
	const CPose3D img_to_camera_pose(0,0,0,DEG2RAD(-90),0,DEG2RAD(-90));
	CPose3D image_pose_on_robot; image_pose_on_robot.composeFrom(camera_pose_on_robot,img_to_camera_pose);
	const CPose3DRotVec camera_pose_on_robot_rvt( image_pose_on_robot );
	CPose3DRotVec camera_pose_on_robot_rvt_inverse = camera_pose_on_robot_rvt;
	camera_pose_on_robot_rvt_inverse.inverse();
	
	m_current_pose = camera_pose_on_robot_rvt;			// initial pose of the camera: Z forwards, Y downwards, X to the right
		
	// srba_options -- load options from config file
	srba_options.loadFromConfigFile( config );
	srba_options.camera_pose_on_robot_rvt 			= camera_pose_on_robot_rvt;
	srba_options.camera_pose_on_robot_rvt_inverse	= camera_pose_on_robot_rvt_inverse;

	// visual odometry engine
	std::vector<std::string> paramSections;
    paramSections.push_back("RECTIFY");
    paramSections.push_back("DETECT");
    paramSections.push_back("MATCH");
    paramSections.push_back("IF-MATCH");
    paramSections.push_back("LEAST_SQUARES");
    paramSections.push_back("GUI");
	paramSections.push_back("GENERAL");

    m_voEngine.loadParamsFromConfigFile(config, paramSections);
	// m_voEngine.setVerbosityLevel( config.read_int("GENERAL","vo_verbosity",0,false) );
	m_voEngine.setVerbosityLevel( general_options.verbose_level );

	// force some visual odometry paremeters to those suitable for this application
    m_voEngine.params_detect.detect_method				= rso::CStereoOdometryEstimator::TDetectParams::dmORB;
    m_voEngine.params_lr_match.match_method				= rso::CStereoOdometryEstimator::TLeftRightMatchParams::smDescRbR;
    m_voEngine.params_if_match.ifm_method				= rso::CStereoOdometryEstimator::TInterFrameMatchingParams::ifmDescBF;

	// overwrite some visual odometry parameters with SRBA application options 
	m_voEngine.params_detect.initial_FAST_threshold		= srba_options.detect_fast_th;
	m_voEngine.params_detect.orb_nlevels				= srba_options.n_levels;
	m_voEngine.params_detect.orb_nfeats					= srba_options.n_feats;

	m_voEngine.dumpToConsole();
	
	// main member RBA
	rba.parameters.sensor.camera_calib.leftCamera       = srba_options.stereo_camera.leftCamera;
    rba.parameters.sensor.camera_calib.rightCamera      = srba_options.stereo_camera.rightCamera;
    rba.parameters.sensor.camera_calib.rightCameraPose  = srba_options.stereo_camera.rightCameraPose;
	rba.parameters.sensor_pose.relative_pose			= image_pose_on_robot; 

	rba.parameters.srba.max_tree_depth				= config.read_int("SRBA_GENERAL","srba_max_tree_depth",3,false);
	rba.parameters.srba.max_optimize_depth			= config.read_int("SRBA_GENERAL","srba_max_optimize_depth",3,false);

    // rba.setVerbosityLevel( config.read_int("SRBA_GENERAL","srba_verbosity", 0, false) );	// 0: None; 1:Important only; 2:Verbose
	rba.setVerbosityLevel( general_options.verbose_level );	// 0: None; 1:Important only; 2:Verbose
	rba.parameters.ecp.submap_size					= config.read_int("SRBA_GENERAL","srba_submap_size",15,false);
	rba.parameters.obs_noise.std_noise_observations = 0.5;							// pixels
	rba.parameters.srba.use_robust_kernel           = config.read_bool("SRBA_GENERAL","srba_use_robust_kernel",true,false);
	rba.parameters.srba.use_robust_kernel_stage1    = config.read_bool("SRBA_GENERAL","srba_use_robust_kernel_stage1",true,false);
	rba.parameters.srba.kernel_param				= config.read_double("SRBA_GENERAL","srba_kernel_param",3.0,false);
	
	// set limits for checking new KF
	m_max_rotation_limit = srba_options.max_rotation, m_max_translation_limit = srba_options.max_translation,
	m_max_rotation = 2*m_max_rotation_limit, m_max_translation = 2*m_max_translation_limit;
			
	// odometry request
	m_odom_request.stereo_cam = srba_options.stereo_camera;
	
	// initialize input stream (camera, rawlog or image dir)
	/** /
	string str;
	if( general_options.cap_src == TGeneralOptions::csRawlog )
	{
		ASSERT_( fileExists( general_options.rawlog_file ) );
		str = string(
			"[CONFIG]\n"
			"grabber_type=rawlog\n"
			"capture_grayscale=false\n"
			"rawlog_file=") + general_options.rawlog_file +
			string("\n");
	}
	else if( general_options.cap_src == TGeneralOptions::csImgDir )
	{
		str = string(
			"[CONFIG]\n"
			"grabber_type=image_dir\n"
			"image_dir_url=") + general_options.image_dir_url +
			string("\n left_format=") + general_options.left_format +
			string("\n right_format=") + general_options.right_format +
			string("\n start_index=") + mrpt::format("%d\n",general_options.start_index).c_str() +
			string("\n end_index=") + mrpt::format("%d\n",general_options.end_index).c_str();
	}
	/**/
	m_myCam.loadConfig( config, "IMG_SOURCE" );
    
	// try to start grabbing images: (will raise an exception on any error)
    m_myCam.initialize();	
	
	// load state (if desired)
	/** /
	if( general_options.load_state_from_file )
	{
		cout << "Loading state from file: " << general_options.state_file << " ... ";
		ASSERT_(fileExists( general_options.state_file ))
		
		m_load_state(	general_options.state_file, 
								start_iteration, 
								m_last_num_tracked_feats,
								m_current_pose,					// poses
								m_last_kf_pose,					// poses
								m_incr_pose_from_last_kf,			// poses
								m_incr_pose_from_last_check,		// poses
								m_keyframes,						// kf
								srba_options, 
								m_voEngine,
								rba,
								general_options,
								srba_options,
								db );

		//	:: update kf identifier counter
		kfID = m_keyframes.rbegin()->m_kf_ID+1;
		
		//	:: build 3d visualization
		if( general_options.show3D )
		{
            COpenGLScenePtr & scene = m_win->get3DSceneAndLock();

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

			// opengl
			opengl_params.span_tree_max_depth			= 1000;
			opengl_params.draw_unknown_feats_ellipses	= false;
			opengl_params.show_unknown_feats_ids		= false;
			opengl_params.draw_unknown_feats			= false;
			opengl_params.draw_kf_hierarchical			= true;

            rba.build_opengl_representation( 0, opengl_params, rba_3d ); // root kf, rendering options, output scene

			// show KF ids
			show_kf_numbers( scene, m_keyframes.size(), QueryResults(), 0 );

			m_win->unlockAccess3DScene();
			m_win->forceRepaint();
        }
		
		cout << " done. Starting at iteration: " << start_iteration << endl;
	} // end-if-load_state_from_file
	/**/

	// gui
	if( general_options.show3D )
	{
		m_win = CDisplayWindow3D::Create("SRBA resulting map",1024,768);
		{
			COpenGLScenePtr &scene = m_win->get3DSceneAndLock();
			
			//	:: insert camera
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
				cam.setPointingAt(mrpt::math::TPoint3D(0.875,0,0));

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
				txt2->setPose( mrpt::math::TPose3D(0,-0.75,0,0,0,0) );
				vp->insert( txt2 );
			}

			m_win->unlockAccess3DScene();
			m_win->setCameraZoom(4);
		}
		m_win->forceRepaint();
	} // end-if-show3D
 }; // end--initialize

void CSRBAStereoSLAMEstimator::m_data_association(
		const CStereoSLAMKF					& kf,						// INPUT
		const TLoopClosureInfo				& lc_info,					// INPUT
		TVectorKfsDaInfo					& out_da )					// OUTPUT
{
	out_da.clear();
	out_da.reserve( lc_info.similar_kfs.size() );

	//	:: preliminary : prepare input from THIS KF (this is common for all the KF to test)
	const size_t num_matches = kf.m_matches.size();
	cv::Mat curLDesc( num_matches,32, kf.m_descriptors_left.type() );
	cv::Mat curRDesc( num_matches,32, kf.m_descriptors_right.type() );

	for(size_t k = 0; k < num_matches; ++k)
	{
		// create matrixes with the proper descriptors: curLDesc and curRDesc
		kf.m_descriptors_left.row( kf.m_matches[k].queryIdx ).copyTo( curLDesc.row(k) );
		kf.m_descriptors_right.row( kf.m_matches[k].trainIdx ).copyTo( curRDesc.row(k) );
	}
	// --------------------------------------------------------------------------

	// check query results and perform data association:
	//	- maximum score should be over an absolute threshold
	//	- non-maximum score should be over an absoulte threshold and over 90% of the maximum one to be considered
	for( size_t k = 0; k < lc_info.similar_kfs.size(); ++k )
	{
		const size_t idx = lc_info.similar_kfs[k];
		VERBOSE_LEVEL(1) << "	DA with KF #" << m_keyframes[idx].m_kf_ID << endl;

		// insert (if possible) the relative position of this KF wrt the previous one
		CPose3DRotVec initialPose = CPose3DRotVec();
		if( srba_options.da_stage2_method == TSRBAStereoSLAMOptions::ST2M_CHANGEPOSE ||
			srba_options.da_stage2_method == TSRBAStereoSLAMOptions::ST2M_BOTH )
		{
			initialPose = CPose3DRotVec(lc_info.similar_kfs_poses[k]);
			if( m_keyframes[idx].m_kf_ID == kf.m_kf_ID-1 ) // only for the previous one
			{
				VERBOSE_LEVEL(2) << "		:: Pose estimated by visual odometry: " << initialPose << endl;
			}
			else
			{
				VERBOSE_LEVEL(2) << "		:: Pose roughly estimated: " << initialPose << endl;
			}
		}

		rso::CStereoOdometryEstimator::TStereoOdometryResult stOdomResult;
		out_da.push_back( t_kf_da_info() );
		m_internal_data_association(
				kf,						// this KF
				m_keyframes[idx],		// the other KF
				curLDesc, curRDesc,		// the current descriptors
				*out_da.rbegin(),		// output data association
				initialPose);			// inkial pose of other KF camera wrt this one KF camera

		// if we already have a higher number of features than the threshold, stop computing data association <-- likely to be deleted
		// if( false && out_da.rbegin()->tracked_matches >= srba_options.updated_matches_th )
		//	break;
	}

	if( lc_info.similar_kfs.size() == 0 )	// this shouldn't happen, but just in case
	{
		rso::CStereoOdometryEstimator::TStereoOdometryResult stOdomResult;
		out_da.push_back( t_kf_da_info() );
		// if there isn't any result over the absolute threhold -> perform data association with the last keyframe (even if it is not the closest)
		m_internal_data_association(
				kf,							// this KF
				*(m_keyframes.rbegin()-1),	// the other KF (the last one)
				curLDesc, curRDesc,			// the current descriptors
				*out_da.rbegin());			// output data association
	}
} // end-performDataAssociation

void CSRBAStereoSLAMEstimator::m_internal_data_association( 
		const CStereoSLAMKF					& this_kf,					// INPUT  -- One KF to perform DA
		const CStereoSLAMKF					& other_kf,					// INPUT  -- The other KF to perform DA with
		const Mat							& this_left_desc,			// INPUT  -- Left image descriptors from current KF
		const Mat							& this_right_desc,			// INPUT  -- Right image descriptors from current KF
		t_kf_da_info						& out_da,					// OUTPUT -- DA information from this KF wrt the other one
		const CPose3DRotVec					& kf_ini_rel_pose )			// oINPUT -- Initial estimation of the relative pose between the KFs
{
	const size_t this_num_matches  = this_kf.m_matches.size();
	const size_t other_num_matches = other_kf.m_matches.size();

	bool invalid = false;

	//	:: prepare output
	out_da.kf_idx = other_kf.m_kf_ID;
	out_da.tracked_matches = 0;
	out_da.tracking_info.resize( this_num_matches, make_pair(INVALID_IDX,0.0f) );

	//	:: preliminary : prepare input from the OTHER KF
	cv::Mat preLDesc( other_num_matches, 32, other_kf.m_descriptors_left.type() );
	// cv::Mat preRDesc( other_num_matches, 32, other_kf.m_descriptors_right.type() );
	
	for(size_t k = 0; k < other_num_matches; ++k)
	{
		// create matrixes with the proper descriptors: preLDesc and preRDesc
		other_kf.m_descriptors_left.row(  other_kf.m_matches[k].queryIdx ).copyTo( preLDesc.row(k) );
		// other_kf.m_descriptors_right.row( other_kf.m_matches[k].trainIdx ).copyTo( preRDesc.row(k) );
	}
	
	//	:: create the matcher (bruteforce with Hamming distance)
	BFMatcher matcher( NORM_HAMMING, false );
	vector<DMatch> matL/*, matR*/;

	//  :: match between left keypoint descriptors
	matcher.match( this_left_desc /*query*/, preLDesc /*train*/, matL /* size of this_left_desc */);
	// matcher.match( this_right_desc /*query*/, preRDesc /*train*/, matR /* size of this_right_desc */);
	
	if( general_options.debug )
	{
		string s = GENERATE_NAME_WITH_2KF_OUT( if_match, this_kf.m_kf_ID, other_kf.m_kf_ID ); 
		FILE *f_if =  mrpt::system::os::fopen( s.c_str(),"wt");
		mrpt::system::os::fprintf(f_if,"%% OTHER_LX OTHER_LY THIS_LX THIS_LY DISTANCE\n");
		for( vector<DMatch>::iterator it = matL.begin(); it != matL.end(); ++it )
		{
			const size_t idxL = this_kf.m_matches[it->queryIdx].queryIdx;
			const size_t idxR = other_kf.m_matches[it->trainIdx].queryIdx;

			// plu, plv, clu, clv, orb_dist
			mrpt::system::os::fprintf(f_if,"%.2f %.2f %.2f %.2f %.2f\n", 
				other_kf.m_keypoints_left[idxR].pt.x, other_kf.m_keypoints_left[idxR].pt.y,
				this_kf.m_keypoints_left[idxL].pt.x, this_kf.m_keypoints_left[idxL].pt.y,
				it->distance);
		}
		mrpt::system::os::fclose(f_if);
	}

	// NEW **********************************************
	deque<TDAMatchInfo> this_matches(this_num_matches);
	// NEW **********************************************

	//	:: STAGE 1 --> filter out by ORB distance + consistency + uniqueness
	t_vector_pair_idx_distance other_matched( other_num_matches, make_pair(INVALID_IDX, std::numeric_limits<float>::max()) );
	vector<int> this_matched( this_num_matches, INVALID_IDX );
	size_t stage1_counter = 0;
	size_t st1_wrong_consistency = 0, st1_wrong_orb_distance = 0;
	for( vector<DMatch>::iterator itL = matL.begin()/*, itR = matR.begin()*/; itL != matL.end(); ++itL/*, ++itR */)
	{
		// consistency check between left and right tracked features
		/*if( false && itL->trainIdx != itR->trainIdx )
		{ 
			st1_wrong_consistency++; 
			continue; 
		}*/																			
		
		// orb distance
		if( itL->distance > srba_options.max_orb_distance_da/* || itR->distance > srba_options.max_orb_distance_da*/ ) 
		{
			this_matches[itL->queryIdx].status = TDAMatchInfo::sREJ_ORB;
			st1_wrong_orb_distance++;
			continue;
		}
		
		// the other 'idx' was already matched but that match was better than this one
		if( itL->distance > other_matched[itL->trainIdx].second )
		{
			this_matches[itL->queryIdx].status = TDAMatchInfo::sNON_TRACKED;
			continue;
		}

		// the other 'idx' was already matched! 
		if( other_matched[itL->trainIdx].first != INVALID_IDX )	
		{
			this_matches[ other_matched[itL->trainIdx].first ].status = TDAMatchInfo::sNON_TRACKED; // undo
			--stage1_counter;
		}
		
		// tracked
		other_matched[itL->trainIdx].first		= itL->queryIdx;
		other_matched[itL->trainIdx].second		= 
		this_matches[itL->queryIdx].distance	= itL->distance;
		this_matches[itL->queryIdx].other_idx	= itL->trainIdx;
		
		++stage1_counter;
#if 0

		// const float mean_distance = 0.5*(itL->distance+itR->distance);
		const float mean_distance = itL->distance;
		if( mean_distance > other_matched[itL->trainIdx].second ) 
		{
			continue;																// this distance is larger than the previous one, continue
		}

		if( other_matched[itL->trainIdx].first != INVALID_IDX )						// this was already matched but now it's better so ...
		{
			this_matched[other_matched[itL->trainIdx].first ] = INVALID_IDX;		// ... undo the previous match
			--stage1_counter;
		}

		// we've got a match (or update a previous match)
		other_matched[itL->trainIdx].first  = itL->queryIdx;
		other_matched[itL->trainIdx].second = mean_distance;
		this_matched[itL->queryIdx]			= itL->trainIdx;
		++stage1_counter;
#endif

	} // end-for

	VERBOSE_LEVEL(2) << "[iDA " << this_kf.m_kf_ID << "->" << other_kf.m_kf_ID << "]: Stage 1 Tracked feats: " << stage1_counter << "/" << matL.size() <<  endl;
	VERBOSE_LEVEL(2) << "	Rejected: " << st1_wrong_consistency << " matches for L-R inconsistency and " << st1_wrong_orb_distance << " matches for larger ORB distance (th=" << srba_options.max_orb_distance_da << ")." <<  endl;

	// DEBUG -----------------------------------------------------------
	/** /
	if( general_options.debug )
	{	
		//	:: save first stage tracking

		FILE *f1 = os::fopen( GENERATE_NAME_WITH_2KF_OUT( cand_stage1_other_tracked, this_kf.m_kf_ID, other_kf.m_kf_ID ), "wt" );
		for( size_t m = 0; m < other_matched.size(); ++m )
			os::fprintf( f1, "%d\n", other_matched[m].first );
		os::fclose(f1);

		FILE *f2 = os::fopen( GENERATE_NAME_WITH_2KF_OUT( cand_stage1_this_tracked, this_kf.m_kf_ID, other_kf.m_kf_ID ), "wt" );
		for( size_t m = 0; m < this_matched.size(); ++m )
			os::fprintf( f2, "%d\n", this_matched[m] );
		os::fclose(f2);

		FILE *f3 = os::fopen( GENERATE_NAME_WITH_2KF_OUT( stage1_tracked, this_kf.m_kf_ID, other_kf.m_kf_ID ), "wt" );
		os::fprintf( f3, "%% THIS_KF_ID THIS_IDX THIS_ID THIS_UL THIS_VL THIS_UR THIS_VR OTHER_KF_ID OTHER_IDX OTHER_ID OTHER_UL OTHER_VL OTHER_UR OTHER_VR\n" );
		for( size_t m = 0; m < other_matched.size(); ++m )
		{
			if( other_matched[m].first == INVALID_IDX ) continue;

			// this
			const size_t tm_idx = other_matched[m].first;
			const size_t tm_id  = this_kf.m_matches_ID[tm_idx];
			const cv::KeyPoint & tlkp = this_kf.m_keypoints_left[this_kf.m_matches[tm_idx].queryIdx];
			const cv::KeyPoint & trkp = this_kf.m_keypoints_right[this_kf.m_matches[tm_idx].trainIdx];

			// other
			const size_t om_idx = m;
			const size_t om_id  = other_kf.m_matches_ID[om_idx];
			const cv::KeyPoint & olkp = other_kf.m_keypoints_left[other_kf.m_matches[om_idx].queryIdx];
			const cv::KeyPoint & orkp = other_kf.m_keypoints_right[other_kf.m_matches[om_idx].trainIdx];

			// dist
			const double dist = other_matched[m].second;

			os::fprintf( f3, "%d %d %d %.2f %.2f %.2f %.2f %d %d %d %.2f %.2f %.2f %.2f %.2f\n",
				this_kf.m_kf_ID, tm_idx, tm_id, tlkp.pt.x, tlkp.pt.y, trkp.pt.x, trkp.pt.y,
				other_kf.m_kf_ID, om_idx, om_id, olkp.pt.x, olkp.pt.y, orkp.pt.x, orkp.pt.y,
				dist );
		}
		os::fclose(f3);
	}
	/**/
	// --------------------------------------------------

	size_t outliers_stage2 = 0;
	//	:: STAGE 2 --> either use a fundamental matrix or the minimization residual to remove outliers
	if( srba_options.da_stage2_method == TSRBAStereoSLAMOptions::ST2M_FUNDMATRIX ||
		srba_options.da_stage2_method == TSRBAStereoSLAMOptions::ST2M_BOTH )
	{
		if( stage1_counter < 15 )
		{
			VERBOSE_LEVEL(2) << "[iDA " << this_kf.m_kf_ID << "->" << other_kf.m_kf_ID << "]: Stage 2 (F) Not enough input data." << endl;
			invalid = true;
		}
		else
		{
			//	:: using fundamental matrix
			// vector<size_t> outliers;

			//	:: detect inliers with fundamental matrix
			m_detect_outliers_with_F(
				this_matches,
				stage1_counter,
				this_kf,
				other_kf);

			/** /
			m_detect_outliers_with_F(
				other_matched,
				this_kf,
				other_kf,
				outliers );
			/**/
#if 0
			FILE *f = NULL;
			if( general_options.debug ) f = os::fopen( GENERATE_NAME_WITH_2KF_OUT( fundmat_outliers, this_kf.m_kf_ID, other_kf.m_kf_ID ), "wt" );
			// remove outliers from the fundamental matrix
			// delete from 'other_matched'
			for( size_t k = 0; k < outliers.size(); ++k )
			{
				other_matched[outliers[k]].first = INVALID_IDX;
				if(f) os::fprintf(f,"%d\n",outliers[k]);
			}
			if(f) os::fclose(f);

			outliers_stage2 = outliers.size();

			VERBOSE_LEVEL(2) << "[iDA " << this_kf.m_kf_ID << "->" << other_kf.m_kf_ID << "]: Stage 2 (F) Outliers detected: " << outliers_stage2 << endl;
#endif
		}
	}

	if( srba_options.da_stage2_method == TSRBAStereoSLAMOptions::ST2M_CHANGEPOSE ||
		srba_options.da_stage2_method == TSRBAStereoSLAMOptions::ST2M_BOTH )
	{
		ASSERT_( stage1_counter >= outliers_stage2 )

		if( stage1_counter - outliers_stage2 < 15 )
		{
			VERBOSE_LEVEL(2) << "[iDA " << this_kf.m_kf_ID << "->" << other_kf.m_kf_ID << "]: Stage 2 (CP) Not enough input data." << endl;
			invalid = true;
		}
		else
		{
			//	:: using residual of minimization
			//vector<size_t> outliers;

			//	:: detect outliers using the residual of the change in pose optimization process
			m_detect_outliers_with_change_in_pose(
				this_matches,
				this_kf,
				other_kf,
				kf_ini_rel_pose );

			/** /
			m_detect_outliers_with_change_in_pose(
				other_matched,
				this_kf,
				other_kf,
				outliers,
				kf_ini_rel_pose );
			/**/
#if 0
			FILE *f = NULL;
			if( general_options.debug ) f = os::fopen( GENERATE_NAME_WITH_2KF_OUT( changepose_outliers, this_kf.m_kf_ID, other_kf.m_kf_ID ), "wt" );
			// remove outliers from the fundamental matrix
			// delete from 'other_matched'
			for( size_t k = 0; k < outliers.size(); ++k )
			{
				other_matched[outliers[k]].first = INVALID_IDX;
				if(f) os::fprintf(f,"%d\n",outliers[k]);
			}
			if(f) os::fclose(f);

			VERBOSE_LEVEL(2) << "[iDA " << this_kf.m_kf_ID << "->" << other_kf.m_kf_ID << "]: Stage 2 (CP) Outliers detected: " << outliers.size() << endl;

			outliers_stage2 += outliers.size();
#endif
		}
	}

	if( !invalid )
	{
		// DEBUG ------------------------------------------------
		FILE *f2 = NULL;
		if( general_options.debug )
			f2 = mrpt::system::os::fopen( GENERATE_NAME_WITH_2KF_OUT(if_match_after, this_kf.m_kf_ID, other_kf.m_kf_ID) ,"wt");
		// ------------------------------------------------------
		for( size_t k = 0; k < this_matched.size(); ++k )
		{
			// .status
			// .other_id
			// .other_index
			// .distance
			if( this_matches[k].status == TDAMatchInfo::sTRACKED )
			{
				out_da.tracking_info[k] = make_pair( this_matches[k].other_idx, this_matches[k].distance );
				out_da.tracked_matches++;
			}
			if( general_options.debug )
			{
				const size_t idxL = this_kf.m_matches[matL[k].queryIdx].queryIdx;
				const size_t idxR = other_kf.m_matches[matL[k].trainIdx].queryIdx;

				const double tlu = this_kf.m_keypoints_left[idxL].pt.x;
				const double tlv = this_kf.m_keypoints_left[idxL].pt.y;

				const double olu = other_kf.m_keypoints_left[idxR].pt.x;
				const double olv = other_kf.m_keypoints_left[idxR].pt.y;

				mrpt::system::os::fprintf(f2,"%d %.2f %.2f %.2f %.2f %.2f\n", 
					this_matches[k].status, 
					tlu, tlv, olu, olv, 
					this_matches[k].distance );
			}
		} // end--for
#if 0		
		//	:: Create output for this KF
		for( size_t k = 0; k < other_matched.size(); ++k )
		{
			if( other_matched[k].first != INVALID_IDX )
			{
				out_da.tracking_info[other_matched[k].first /*this_idx*/] = make_pair(int(k) /*other_idx*/, other_matched[k].second /*distance*/);
				out_da.tracked_matches++;

				// DEBUG ------------------------------------------------
				if( general_options.debug )
				{
					const size_t idxL = this_kf.m_matches[other_matched[k].first].queryIdx;
					const size_t idxR = other_kf.m_matches[int(k)].queryIdx;

					const double lu = this_kf.m_keypoints_left[idxL].pt.x;
					const double lv = this_kf.m_keypoints_left[idxL].pt.y;

					const double ru = other_kf.m_keypoints_left[idxR].pt.x;
					const double rv = other_kf.m_keypoints_left[idxR].pt.y;

					mrpt::system::os::fprintf(f2,"%.2f %.2f %.2f %.2f %.2f\n", lu, lv, ru, rv, other_matched[k].second);
				}
				// ------------------------------------------------------
			} // end-if
		} // end-for
#endif
		// DEBUG ------------------------------------------------
		if( general_options.debug )
			mrpt::system::os::fclose(f2);
		// ------------------------------------------------------
	} // end-if

	VERBOSE_LEVEL(2) << "[iDA " << this_kf.m_kf_ID << "->" << other_kf.m_kf_ID << "]: Total tracked feats: " << out_da.tracked_matches << "/" << matL.size() << endl;

} // end-internal_performDataAssociation

/** /
 * Determines those KFs which are similar to this one according to the query DB results
 * Criteria -- a KF is considered similar if either:
 *	- it is the previous one
 *	- its score is over 80% of the maximum score -- 0.8*dbQueryResults[0].score
 *	- its score is over an absolute minimum (0.05) and it is far away from the current KF (potential loop closure)
 * Returns: potential loop closure detected (bool)
/**/
bool CSRBAStereoSLAMEstimator::m_get_similar_kfs( 
		const TKeyFrameID			& newKfId,
		const QueryResults			& dbQueryResults,
		TLoopClosureInfo			& out )
{
	const size_t qSize = dbQueryResults.size();
	if( qSize == 0 )
		THROW_EXCEPTION( "Parameter 'dbQueryResults' contains no results. This method should not be called here." );

	VERBOSE_LEVEL(2) << "dbQueryResults: " << dbQueryResults << endl;

	if( dbQueryResults[0].Score < 0.04 /* TODO: absoluteDbQueryThreshold */ )
	{
		SHOW_WARNING( "Best result in 'dbQueryResults' is below a minimum threshold. Lost camera?" );
	}

	// prepare output
	out.similar_kfs.clear();
	out.similar_kfs.reserve( qSize+1 );

	// always insert last kf as a similar one
	out.similar_kfs.push_back( newKfId-1 );

	if( qSize == 1 )	// we have already added the last one, set the pose (if needed) and we are done here
	{
		if( srba_options.da_stage2_method == TSRBAStereoSLAMOptions::ST2M_CHANGEPOSE ||
		srba_options.da_stage2_method == TSRBAStereoSLAMOptions::ST2M_BOTH )
		{
			out.similar_kfs_poses.push_back(m_incr_pose_from_last_kf.getInverse());	// pose of previous KF (camera to camera) wrt this one
		}
		return false;
	}

	// we have more than one result, analyze them
	out.lc_id = INVALID_KF_ID;
	bool found_potential_loop_closure = false;

	// we've got enough good data, loop closure? -- we'll detect a LC if in the list there is any far KF with a score large enough
	mySRBA::rba_problem_state_t & myRbaState = rba.get_rba_state();
	
	// get id of the last localmap center -- if last inserted kf is a base, then use it, if not, use the previous one
	const size_t SUBMAP_SIZE = rba.parameters.ecp.submap_size;		// In # of KFs
	const TKeyFrameID fromIdBase = SUBMAP_SIZE*((newKfId-1)/SUBMAP_SIZE);
	// const TKeyFrameID fromIdBase = rba.isKFLocalmapCenter( newKfId-1 ) ? newKfId-1 : rba.getLocalmapCenterID( newKfId-1 );						

	mySRBA::rba_problem_state_t::TSpanningTree::next_edge_maps_t::const_iterator itFrom =
		myRbaState.spanning_tree.sym.next_edge.find( fromIdBase );	// get spanning tree for the current localmap center

	// check the results
	const double add_similar_kf_th = 0.8*dbQueryResults[0].Score;
	for( size_t i = 0; i < dbQueryResults.size(); ++i )
	{
		const TKeyFrameID toId = dbQueryResults[i].Id;

		if( toId == newKfId-1 )	// already inserted
			continue;

		if( dbQueryResults[i].Score > add_similar_kf_th )
		{
			out.similar_kfs.push_back( toId );
			continue;
		}

		// compute topologic distance from current localmap center 'itFrom' to target KF 'toId'
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
				THROW_EXCEPTION("[ERROR :: Check Loop Closure] 'itFrom' is not into the spanning_tree!");
			}
		}

		bool insertKf = false;
		if( out.lc_id == INVALID_KF_ID && topoDistance > srba_options.lc_distance && dbQueryResults[i].Score > 0.05 )
		{
			// only set the lc with the first KF found
			out.lc_id = toId;
			found_potential_loop_closure = true;
			VERBOSE_LEVEL(1) << "		FOUND POTENTIAL LOOP CLOSURE " << endl;

			//	add this KF as similar
			out.similar_kfs.push_back( toId );
		}

		VERBOSE_LEVEL(2) << "		Distance from " << fromIdBase << " to " << toId << " : " << topoDistance << endl;
	} // end-for

	// For all similar KFs, get a rough estimation of all the KF poses (camera to camera) wrt to this one (only if change in pose is going to be used as a filter)
	// getChangeInPose needs as input the initial estimation of pose of other camera wrt the current one, so we have:
	//	-- the inverse of the 'm_incr_pose_from_last_kf' for the previous KF
	//	-- the inverse of the 'm_incr_pose_from_last_kf' composed by the sequence K^(-1)*pose_other_wrt_this*K, with K = camera_pose_on_robot
	out.similar_kfs_poses.resize( out.similar_kfs.size(),CPose3D() );
	
	if( srba_options.da_stage2_method == TSRBAStereoSLAMOptions::ST2M_CHANGEPOSE ||
		srba_options.da_stage2_method == TSRBAStereoSLAMOptions::ST2M_BOTH )
	{
		CPose3DRotVec incr_pose_from_last_kf_inverse = m_incr_pose_from_last_kf.getInverse(); // camera to camera

		// search along the spantree for the poses: 
		mySRBA::frameid2pose_map_t  spantree;
		rba.create_complete_spanning_tree(newKfId-1, spantree, rba.parameters.srba.max_tree_depth );
		for( size_t k = 0; k < out.similar_kfs.size(); ++k )
		{
			if( out.similar_kfs[k] == newKfId-1 )
				out.similar_kfs_poses[k] = incr_pose_from_last_kf_inverse;	// pose of previous KF (camera to camera) wrt this one
			else
			{
				mySRBA::frameid2pose_map_t::const_iterator itP = spantree.find( out.similar_kfs[k] );
				if( itP != spantree.end() )
				{
					CPose3DRotVec aux_pose_rvt = 
						srba_options.camera_pose_on_robot_rvt_inverse + 
						CPose3DRotVec(itP->second.pose) + 
						srba_options.camera_pose_on_robot_rvt;

					out.similar_kfs_poses[k] = CPose3D(incr_pose_from_last_kf_inverse+aux_pose_rvt);
				}
			} // end-else
		} // end-for
	} // end-if

	// DEBUG ------------------------------------------
	if( general_options.verbose_level >= 2 )
	{
		DUMP_VECTORLIKE( out.similar_kfs )
	}
	// ------------------------------------------------

	return found_potential_loop_closure;
} // end--m_get_similar_kfs

/** Compute the fundamental matrix between the left images and also between the right ones and find outliers */
void CSRBAStereoSLAMEstimator::m_detect_outliers_with_F ( 
		deque<TDAMatchInfo>		& this_matches, 
		const size_t			& num_tracked,
		const CStereoSLAMKF		& this_kf, 
		const CStereoSLAMKF		& other_kf )
{
	cv::Mat p_this_left(num_tracked,2,cv::DataType<float>::type);
	cv::Mat p_other_left(num_tracked,2,cv::DataType<float>::type);
	for( size_t k = 0, k0 = 0; k < this_matches.size(); ++k )
	{
		if( this_matches[k].status == TDAMatchInfo::sTRACKED )
		{
			const size_t this_idx_left  = this_kf.m_matches[k].queryIdx;
			p_this_left.at<float>(k0,0) = static_cast<float>(this_kf.m_keypoints_left[this_idx_left].pt.x);
			p_this_left.at<float>(k0,1) = static_cast<float>(this_kf.m_keypoints_left[this_idx_left].pt.y);

			const size_t other_idx_left	= other_kf.m_matches[this_matches[k].other_idx].queryIdx;
			p_other_left.at<float>(k0,0) = static_cast<float>(other_kf.m_keypoints_left[other_idx_left].pt.x);
			p_other_left.at<float>(k0,1) = static_cast<float>(other_kf.m_keypoints_left[other_idx_left].pt.y);
			k0++;
		}
	} // end-for

	// compute fundamental matrix
	vector<uchar> left_inliers;
	cv::findFundamentalMat( p_other_left, p_this_left, cv::FM_RANSAC, srba_options.max_y_diff_epipolar, srba_options.ransac_fit_prob, left_inliers );
	
	// update da info
	for( size_t k = 0, k0 = 0; k < this_matches.size(); ++k )
	{
		if( this_matches[k].status == TDAMatchInfo::sTRACKED )
		{
			if( left_inliers[k0++] == 0 )
				this_matches[k].status = TDAMatchInfo::sREJ_FUND_MATRIX;
		}
	} // end--for
} // end--m_detect_outliers_with_F

void CSRBAStereoSLAMEstimator::m_detect_outliers_with_F ( 
		const t_vector_pair_idx_distance	& other_matched, 
		const CStereoSLAMKF					& this_kf, 
		const CStereoSLAMKF					& other_kf, 
		vector<size_t>						& outliers )
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

	cv::Mat ppl(num_tracked,2,cv::DataType<float>::type)/*,ppr(num_tracked,2,cv::DataType<float>::type)*/;
	cv::Mat pcl(num_tracked,2,cv::DataType<float>::type)/*,pcr(num_tracked,2,cv::DataType<float>::type)*/;
	for( size_t k = 0; k < tracked_idx.size(); ++k )
	{
		const size_t idx		= tracked_idx[k];
		const size_t preIdxL	= other_kf.m_matches[idx].queryIdx;
		const size_t preIdxR	= other_kf.m_matches[idx].trainIdx;
		const size_t curIdxL	= this_kf.m_matches[other_matched[idx].first].queryIdx;
		const size_t curIdxR	= this_kf.m_matches[other_matched[idx].first].trainIdx;

        ppl.at<float>(k,0) = static_cast<float>(other_kf.m_keypoints_left[preIdxL].pt.x);
        ppl.at<float>(k,1) = static_cast<float>(other_kf.m_keypoints_left[preIdxL].pt.y);

		/*ppr.at<float>(k,0) = static_cast<float>(other_kf.m_keypoints_right[preIdxR].pt.x);
        ppr.at<float>(k,1) = static_cast<float>(other_kf.m_keypoints_right[preIdxR].pt.y);*/

		pcl.at<float>(k,0) = static_cast<float>(this_kf.m_keypoints_left[curIdxL].pt.x);
        pcl.at<float>(k,1) = static_cast<float>(this_kf.m_keypoints_left[curIdxL].pt.y);

		/*pcr.at<float>(k,0) = static_cast<float>(this_kf.m_keypoints_right[curIdxR].pt.x);
        pcr.at<float>(k,1) = static_cast<float>(this_kf.m_keypoints_right[curIdxR].pt.y);*/
	}

	vector<uchar> left_inliers, right_inliers;
	cv::findFundamentalMat( ppl, pcl, cv::FM_RANSAC, srba_options.max_y_diff_epipolar, srba_options.ransac_fit_prob, left_inliers );
	//cv::findFundamentalMat( ppr, pcr, cv::FM_RANSAC, srba_options.max_y_diff_epipolar, srba_options.ransac_fit_prob, right_inliers );
	for( size_t k = 0; k < left_inliers.size(); ++k )
	{
		if( left_inliers[k] == 0/* || right_inliers[k] == 0*/ )
			outliers.push_back(tracked_idx[k]);	// output: 'idx' from other_matched that has been undetected outliers up to now
	}
} // end-m_detect_outliers_with_F

void CSRBAStereoSLAMEstimator::m_detect_outliers_with_change_in_pose ( 
		deque<TDAMatchInfo>					& this_matches, 
		const CStereoSLAMKF					& this_kf, 
		const CStereoSLAMKF					& other_kf, 
		const CPose3DRotVec					& kf_ini_rel_pose )
{
	// create a vector with the tracked pairs so far for the visual odometer
	rso::vector_index_pairs_t tracked_pairs;
	tracked_pairs.reserve( this_kf.m_matches.size() );
	for( size_t k = 0; k < this_matches.size(); ++k )
	{
		if( this_matches[k].status == TDAMatchInfo::sTRACKED )
			tracked_pairs.push_back( make_pair( this_matches[k].other_idx /*other_idx*/, k /*this_idx*/) );
	}
	size_t num_tracked_stage1 = tracked_pairs.size();

	// 'getChangeInPose' accepts an initial value for the change in pose between frames.
	// The initial estimation is the a vector form of the pose of the previous frame wrt the current one
	vector<double> initialPoseVector(6); // [w1,w2,w3,tx,ty,tz]
	initialPoseVector[0] = kf_ini_rel_pose.m_rotvec[0];	initialPoseVector[1] = kf_ini_rel_pose.m_rotvec[1];	initialPoseVector[2] = kf_ini_rel_pose.m_rotvec[2];
	initialPoseVector[3] = kf_ini_rel_pose.m_coords[0];	initialPoseVector[4] = kf_ini_rel_pose.m_coords[1];	initialPoseVector[5] = kf_ini_rel_pose.m_coords[2];

	rso::CStereoOdometryEstimator::TStereoOdometryResult  odometry_result;
	m_voEngine.params_least_squares.use_custom_initial_pose = true;
	const bool valid = m_voEngine.getChangeInPose(
			tracked_pairs,											// the tracked pairs
			other_kf.m_matches, this_kf.m_matches,					// pre_matches, cur_matches,
			other_kf.m_keypoints_left, other_kf.m_keypoints_right,	// pre_left_feats, pre_right_feats, 
			this_kf.m_keypoints_left, this_kf.m_keypoints_right,	// cur_left_feats, cur_right_feats,
			srba_options.stereo_camera,
			odometry_result,					// output
			initialPoseVector );				// [w1,w2,w3,tx,ty,tz]
	m_voEngine.params_least_squares.use_custom_initial_pose = false;

	VERBOSE_LEVEL(2) << "Iterations: " << odometry_result.num_it << " (initial) and " << odometry_result.num_it_final << " (final) " << endl;
	VERBOSE_LEVEL(2) << "Change in pose: " << odometry_result.outPose << endl;
	// I already have the outliers (got from the optimizer's first stage) but we may want adjust the threshold here

	if( !odometry_result.valid )
	{
		VERBOSE_LEVEL(1) << "	WARNING: Change in pose could not be estimated, all points are set to outliers" << endl;
		for( size_t k = 0; k < this_matches.size(); ++k )
		{
			if( this_matches[k].status == TDAMatchInfo::sTRACKED )
				this_matches[k].status = TDAMatchInfo::sREJ_CHANGE_POSE;
		}
		return;
	}

	// remove points with large residuals
	for( size_t k = 0, k0 = 0; k < this_matches.size(); ++k )
	{
		if( this_matches[k].status == TDAMatchInfo::sTRACKED )
		{
			// we've got a residual for this
			if( odometry_result.out_residual[k0] > srba_options.residual_th )
				this_matches[k].status = TDAMatchInfo::sREJ_CHANGE_POSE;
			k0++;
		}
	} // end-for
} // end--m_detect_outliers_with_change_in_pose

void CSRBAStereoSLAMEstimator::m_detect_outliers_with_change_in_pose ( 
		t_vector_pair_idx_distance			& other_matched, 
		const CStereoSLAMKF					& this_kf, 
		const CStereoSLAMKF					& other_kf, 
		vector<size_t>						& outliers,						// OUTPUT
		const CPose3DRotVec					& kf_ini_rel_pose )
{
	// prepare output
	outliers.clear(); outliers.reserve( other_matched.size() );

	// create a vector with the tracked pairs so far for the visual odometer
	rso::vector_index_pairs_t tracked_pairs;
	tracked_pairs.reserve( this_kf.m_matches.size() );
	for( size_t k = 0; k < other_matched.size(); ++k )
	{
		if( other_matched[k].first != INVALID_IDX )
			tracked_pairs.push_back( make_pair( k /*other_idx*/, size_t(other_matched[k].first) /*this_idx*/) );
	}
	size_t num_tracked_stage1 = tracked_pairs.size();

	// 'getChangeInPose' accepts an initial value for the change in pose between frames.
	// The initial estimation is the a vector form of the pose of the previous frame wrt the current one
	vector<double> initialPoseVector(6); // [w1,w2,w3,tx,ty,tz]
	initialPoseVector[0] = kf_ini_rel_pose.m_rotvec[0];	initialPoseVector[1] = kf_ini_rel_pose.m_rotvec[1];	initialPoseVector[2] = kf_ini_rel_pose.m_rotvec[2];
	initialPoseVector[3] = kf_ini_rel_pose.m_coords[0];	initialPoseVector[4] = kf_ini_rel_pose.m_coords[1];	initialPoseVector[5] = kf_ini_rel_pose.m_coords[2];

	rso::CStereoOdometryEstimator::TStereoOdometryResult  odometry_result;
	m_voEngine.params_least_squares.use_custom_initial_pose = true;
	const bool valid = m_voEngine.getChangeInPose(
			tracked_pairs,											// the tracked pairs
			other_kf.m_matches, this_kf.m_matches,					// pre_matches, cur_matches,
			other_kf.m_keypoints_left, other_kf.m_keypoints_right,	// pre_left_feats, pre_right_feats, 
			this_kf.m_keypoints_left, this_kf.m_keypoints_right,	// cur_left_feats, cur_right_feats,
			srba_options.stereo_camera,
			odometry_result,					// output
			initialPoseVector );				// [w1,w2,w3,tx,ty,tz]
	m_voEngine.params_least_squares.use_custom_initial_pose = false;

	VERBOSE_LEVEL(2) << "Iterations: " << odometry_result.num_it << " (initial) and " << odometry_result.num_it_final << " (final) " << endl;
	VERBOSE_LEVEL(2) << "Change in pose: " << odometry_result.outPose << endl;
	// I already have the outliers (got from the optimizer's first stage) but we may want adjust the threshold here

	if( !odometry_result.valid )
	{
		VERBOSE_LEVEL(1) << "	WARNING: Change in pose could not be estimated, skipping this test." << endl;
		if( general_options.debug ) 
		{
			// empty file
			FILE *f = os::fopen( GENERATE_NAME_WITH_KF_OUT( posechange_outliers, this_kf ), "wt" );
			os::fclose(f);
		}
	}

	// remove large outliers
	FILE *f = NULL;
	if( general_options.debug ) f = os::fopen( GENERATE_NAME_WITH_KF_OUT( posechange_outliers, this_kf ), "wt" );
	for( size_t k = 0; k < odometry_result.out_residual.size(); ++k )
	{
		if( odometry_result.out_residual[k] > srba_options.residual_th )
		{
			// only set outliers if the change in pose had a valid solution
			if( odometry_result.valid )			
				outliers.push_back( tracked_pairs[k].first );

			// in any case, save the "id" and the residual of the outlier for debug purposes
			if(f) os::fprintf( f, "%d %.2f\n", tracked_pairs[k].first, odometry_result.out_residual[k] );
		}
	} // end-for

	if(f) os::fclose(f);

	// re-adjust 'results.out_residual' size to match other_matched size
	vector<double> out_residual( other_matched.size(), std::numeric_limits<double>::max() );
	size_t cnt = 0;
	for( size_t k = 0; k < other_matched.size(); ++k )
	{
		if( other_matched[k].first != INVALID_IDX )
			out_residual[k] = odometry_result.out_residual[cnt++];
	}
	odometry_result.out_residual.swap(out_residual);
} // end-m_detect_outliers_with_change_in_pose

#if 0
// ---------------------------------------------------
// load application state
// ---------------------------------------------------
 bool CSRBAStereoSLAMEstimator::m_load_state(
	const string 					& filename,					// input file
	size_t							& count,
	int								& m_last_num_tracked_feats,
	CPose3DRotVec					& m_current_pose,
	CPose3DRotVec					& m_last_kf_pose,
	CPose3DRotVec					& m_incr_pose_from_last_kf,
	CPose3DRotVec					& m_incr_pose_from_last_check,
	t_vector_kf						& keyframes,
	TSRBAStereoSLAMOptions				& stSLAMOpts,
	rso::CStereoOdometryEstimator	& m_voEngine,
	mySRBA /*myRBAEngine*/			& rba,
	TGeneralOptions						& general_options,
	TSRBAStereoSLAMOptions				& stereo_options,
	BriefDatabase					& db )
{
	string n_filename = mrpt::format("%s\\%s", general_options.out_dir.c_str(), filename.c_str() );
	std::ifstream state_file_stream( n_filename.c_str(), ios::in | ios::binary );	// read
	if( !state_file_stream.is_open() )
		return false;

	// global parameters
	//	-- iteration number
	state_file_stream.read( (char*)&count, sizeof(size_t) );

	//	-- last match ID
	state_file_stream.read(reinterpret_cast<char*>(&CStereoSLAMKF::m_last_match_ID),sizeof(size_t)); // >> aux;							// matches last id

	//	-- important poses
	LOAD_ROTVEC_FROM_STREAM( state_file_stream, m_current_pose )
	LOAD_ROTVEC_FROM_STREAM( state_file_stream, m_last_kf_pose )
	LOAD_ROTVEC_FROM_STREAM( state_file_stream, m_incr_pose_from_last_kf )
	LOAD_ROTVEC_FROM_STREAM( state_file_stream, m_incr_pose_from_last_check )

	if(!m_load_options_from_stream( state_file_stream, stSLAMOpts ))
	{
		cout << "ERROR while loading the state -- options could not be loaded. Closed stream?" << endl;
		return false;
	}

	// last number tracked feats
	state_file_stream.read( (char*)&m_last_num_tracked_feats, sizeof(int) );

	// kfs
	size_t num_kfs;
	state_file_stream.read(reinterpret_cast<char*>(&num_kfs),sizeof(size_t));
	keyframes.resize( num_kfs );

	VERBOSE_LEVEL(1) << "	" << num_kfs << " keyframes to load: " << endl;

	TSRBAStereoSLAMOptions opt;
	for( size_t k = 0; k < num_kfs; ++k )
	{
		CStereoSLAMKF & kf = keyframes[k];

		// kf ID
		size_t kf_id;
		state_file_stream.read( (char*)&kf_id, sizeof(size_t) );
		kf.setKFID( kf_id );

		LOAD_ROTVEC_FROM_STREAM( state_file_stream, kf.m_camera_pose )

		// kf features
		if( !m_load_keypoints_from_stream( state_file_stream, kf.m_keypoints_left, kf.m_keyDescLeft ) )
		{
			cout << "ERROR while loading the state -- left keypoints could not be loaded. Closed stream?" << endl;
			return false;
		}
		if( !m_load_keypoints_from_stream( state_file_stream, kf.m_keypoints_right, kf.m_keyDescRight ) )
		{
			cout << "ERROR while loading the state -- right keypoints could not be loaded. Closed stream?" << endl;
			return false;
		}

		// kf matches
		if( !m_load_matches_from_stream( state_file_stream, kf.m_matches, kf.m_matches_ID ) )
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

			const KeyPoint & kpLeft  		= kf.m_keypoints_left[id1];
			const KeyPoint & kpRight 		= kf.m_keypoints_right[id2];

			obsField.obs.feat_id            = kf.m_matches_ID[m];
			obsField.obs.obs_data.left_px   = TPixelCoordf( kpLeft.pt.x,  kpLeft.pt.y );
			obsField.obs.obs_data.right_px  = TPixelCoordf( kpRight.pt.x, kpRight.pt.y );
			obsField.setRelPos( kf.projectMatchTo3D( m, stereo_options ) );

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

	if( !m_voEngine.loadStateFromFile( vo_filename ) )
	{
		cout << "ERROR while saving the state -- vodometry could not be saved. Closed stream?" << endl;
		return false;
	}
	state_file_stream.close();

	// load bag of words database
	db.load( mrpt::format("%s\\db_saved.gz", general_options.out_dir.c_str() ) );

	return true;
} // end-load_state_from_file
#endif
// ---------------------------------------------------
// load methods
// ---------------------------------------------------
 bool CSRBAStereoSLAMEstimator::m_load_options_from_stream( std::ifstream & stream, TSRBAStereoSLAMOptions & options )
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

 bool CSRBAStereoSLAMEstimator::m_load_keypoints_from_stream( std::ifstream & stream, TKeyPointList & keypoints, Mat & descriptors )
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

 bool CSRBAStereoSLAMEstimator::m_load_matches_from_stream( std::ifstream & stream, TDMatchList & matches, vector<size_t> & matches_ids )
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
// dumping methods
// ---------------------------------------------------
 bool CSRBAStereoSLAMEstimator::m_dump_keypoints_to_stream( std::ofstream & stream, const TKeyPointList & keypoints, const Mat & descriptors )
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

 bool CSRBAStereoSLAMEstimator::m_dump_options_to_stream( std::ofstream & stream, const TSRBAStereoSLAMOptions & options )
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

 bool CSRBAStereoSLAMEstimator::m_dump_matches_to_stream( std::ofstream & stream, const TDMatchList & matches, const vector<size_t> & matches_ids )
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

#if 0
// ---------------------------------------------------
// save application state
// ---------------------------------------------------
 bool CSRBAStereoSLAMEstimator::m_save_state(
	const string 					& filename,						// output file
	const size_t					& count,
	const int						& m_last_num_tracked_feats,
	const CPose3DRotVec				& m_current_pose,
	const CPose3DRotVec				& m_last_kf_pose,
	const CPose3DRotVec				& m_incr_pose_from_last_kf,
	const CPose3DRotVec				& m_incr_pose_from_last_check,
	const t_vector_kf				& keyframes,
	const TSRBAStereoSLAMOptions		& stSLAMOpts,
	rso::CStereoOdometryEstimator	& m_voEngine,
	TGeneralOptions						& general_options,
	BriefDatabase					& db )
{
	/*	FORMAT:
		- starting iteration
		- last match id
		- map of reference ids
		- important poses: m_current_pose, incr_pose??, m_last_kf_pose, m_incr_pose_from_last_kf
		- # of kfs
		- options*
		- kf ID
		- kf camera pose (zeros by now)
		- left features*
		- right features*
		- matches*
	*/
	string n_filename = mrpt::format("%s\\%s", general_options.out_dir.c_str(), filename.c_str() );
	std::ofstream state_file_stream( n_filename.c_str(), ios::out | ios::binary );			// write
	if( !state_file_stream.is_open() )
		return false;

	// global parameters
	//	-- iteration number
	state_file_stream.write( (char*)&count, sizeof(size_t) );

	//	-- last match ID
	state_file_stream.write( reinterpret_cast<char*>(&CStereoSLAMKF::m_last_match_ID), sizeof(size_t) );

	//	-- important poses
	DUMP_ROTVEC_TO_STREAM( state_file_stream, m_current_pose )
	DUMP_ROTVEC_TO_STREAM( state_file_stream, m_last_kf_pose )
	DUMP_ROTVEC_TO_STREAM( state_file_stream, m_incr_pose_from_last_kf )
	DUMP_ROTVEC_TO_STREAM( state_file_stream, m_incr_pose_from_last_check )

	//	-- options
	if( !dumpOptionsToStream( state_file_stream, stSLAMOpts ) )	// save options (only for the first kf, they are all the same)
	{
		cout << "ERROR while saving the state -- options could not be saved. Closed stream?" << endl;
		return false;
	}

	//	-- last number of tracked features
	state_file_stream.write( (char*)(&m_last_num_tracked_feats), sizeof(int) );

	//	-- kfs
	size_t num_kf = keyframes.size();
	state_file_stream.write( reinterpret_cast<char*>(&num_kf), sizeof(size_t) );											// number of kfs

	for( size_t k = 0; k < keyframes.size(); ++k )
	{
		const CStereoSLAMKF & kf = keyframes[k];									// shortcut

		state_file_stream.write( (char*)&(kf.m_kf_ID), sizeof(size_t) );

		DUMP_ROTVEC_TO_STREAM( state_file_stream, kf.m_camera_pose )

		// kf features
		if( !dumpKeyPointsToStream( state_file_stream, kf.m_keypoints_left, kf.m_keyDescLeft ) )
		{
			cout << "ERROR while saving the state -- left keypoints could not be saved. Closed stream?" << endl;
			return false;
		}
		if( !dumpKeyPointsToStream( state_file_stream, kf.m_keypoints_right, kf.m_keyDescRight ) )
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

	if( !m_voEngine.saveStateToFile( vo_filename ) )
	{
		cout << "ERROR while saving the state -- vodometry could not be saved. Closed stream?" << endl;
		return false;
	}

	// global parameters
	state_file_stream.close();

	// save bag of words database
	db.save( mrpt::format("%s\\db_saved.gz", general_options.out_dir.c_str() ) );

	return true;
} // end-method
#endif
