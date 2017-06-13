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
#include "srba-stereo-slam_utils.h"
#include "srba-stereo-slam_common.h"
#include "srba-stereo-slam.h"

extern TGeneralOptions general_options;

// These are static methods (only available when the header file is included)
// ---------------------------------------------------
// comparison (auxiliary methods)
// ---------------------------------------------------
bool compareKeypointLists( 
	const TKeyPointList	& list1, 
	const Mat 			& desc1, 
	const TKeyPointList & list2, 
	const Mat 			& desc2 )
{
	if( list1.size() != list2.size() )
		return false;

	if( desc1.size() != desc2.size() )
		return false;

	// keyp
	TKeyPointList::const_iterator it1, it2;
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

bool compareMatchesLists( 
	const TDMatchList 	& list1, 
	const TDMatchList 	& list2 )
{
	if( list1.size() != list2.size() )
		return false;

	TDMatchList::const_iterator it1, it2;
	for( it1 = list1.begin(), it2 = list2.begin(); it1 != list1.end(); ++it1, ++it2 )
	{
		if( it1->queryIdx != it2->queryIdx || it1->trainIdx != it2->trainIdx || it1->distance != it2->distance || it1->imgIdx != it2->imgIdx )
			return false;
	}

	return true;
}

bool compareOptions( 
	const TSRBAStereoSLAMOptions 	& opt1, 
	const TSRBAStereoSLAMOptions 	& opt2 )
{
	return 	opt1.n_levels == opt2.n_levels && opt1.n_feats == opt2.n_feats && opt1.min_ORB_distance == opt2.min_ORB_distance &&
			opt1.matching_options == opt2.matching_options &&
			opt1.max_y_diff_epipolar == opt2.max_y_diff_epipolar &&
			opt1.max_orb_distance_da == opt2.max_orb_distance_da &&
			opt1.ransac_fit_prob == opt2.ransac_fit_prob &&
			opt1.max_translation == opt2.max_translation && opt1.max_rotation == opt2.max_rotation &&
			opt1.residual_th == opt2.residual_th && opt1.non_maximal_suppression == opt2.non_maximal_suppression &&
			opt1.updated_matches_th == opt2.updated_matches_th && opt1.up_matches_th_plus == opt2.up_matches_th_plus &&
			opt1.detect_method == opt2.detect_method && opt1.detect_fast_th == opt2.detect_fast_th && 
			opt1.non_max_supp_method == opt2.non_max_supp_method;
}

// ---------------------------------------------------
// show kf information
// ---------------------------------------------------
void show_kf_numbers( 
	COpenGLScenePtr 			& scene, 
	const size_t 				& num_kf, 
	const DBoW2::QueryResults 	& ret, 
	const double 				& th )
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
			box->setBoxCorners(mrpt::math::TPoint3D(0.5*k,0,0.0),mrpt::math::TPoint3D(0.5*k+0.25,ret[k].Score,0.0));
			box->setColor(mrpt::utils::TColorf(1-3*ret[k].Score,3*ret[k].Score,0));
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

double updateTranslationThreshold( 
	const double x, 
	const double th )
{
	double newTh = 0.02 + (0.25/th)*x;
	newTh = newTh < 0.02 ? 0.02 : newTh;
	newTh = newTh > 0.3 ? 0.3 : newTh;
	return newTh;
} // end -- updateTranslationThreshold

double updateRotationThreshold( 
	const double x, 
	const double th )
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
	const TKeyFrameID				& newKfId,
	const DBoW2::QueryResults		& dbQueryResults,
	mySRBA							& rba,
	const TSRBAStereoSLAMOptions	& stereoSlamOptions,
	TLoopClosureInfo				& out )
{
	if( general_options.verbose_level >= 2 )
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
	/** /
	const TKeyFrameID fromIdBase =
		rba.isKFLocalmapCenter( newKfId-1 ) ?
		newKfId-1 :
		rba.getLocalmapCenterID( newKfId-1 );						// get id of the last localmap center
	/**/
	const size_t SUBMAP_SIZE = rba.parameters.ecp.submap_size;		// In # of KFs
	const TKeyFrameID fromIdBase = SUBMAP_SIZE*((newKfId-1)/SUBMAP_SIZE);

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
	if( general_options.verbose_level >= 2 )
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
		const DBoW2::QueryResults	& ret,
		mySRBA						& rba,
		const TSRBAStereoSLAMOptions	& stereo_slam_options,
		TLoopClosureInfo			& lc_info )
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
