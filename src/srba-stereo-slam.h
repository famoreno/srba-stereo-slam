#pragma once
#include "srba-stereo-slam_common.h"
using namespace srba;
using namespace mrpt::opengl;
using namespace mrpt::poses;

// the options for the RBA Engine
struct my_srba_options : public RBA_OPTIONS_DEFAULT
{
	//typedef options::observation_noise_identity			obs_noise_matrix_t;     // The sensor noise matrix is the same for all observations and equal to \sigma * I(identity)
	typedef options::sensor_pose_on_robot_se3			sensor_pose_on_robot_t;	
	typedef options::solver_LM_schur_dense_cholesky     solver_t;					// Solver algorithm
	//typedef ecps::local_areas_var_size				edge_creation_policy_t;  
	typedef ecps::local_areas_fixed_size				edge_creation_policy_t;  
};

// define the RBA Problem state for this application
typedef TRBA_Problem_state <
	kf2kf_poses::SE3,
    landmarks::Euclidean3D,
    observations::StereoCamera,
    my_srba_options
> myRBAProblemState;

// define the RBA Engine for this application
typedef RbaEngine <
	kf2kf_poses::SE3,				// 6D movement
	landmarks::Euclidean3D,			// {X,Y,Z} 3D landmarks
    observations::StereoCamera,		// Observations are stereo: o^i = {ul^i,vl^i,ur^i,vr^i}
    my_srba_options
    >
myRBAEngine;

/*********************************************
CLASS: Customized SRBA engine
**********************************************/
class mySRBA : public myRBAEngine
{
private : 
	bool									m_lc;									//!< Indicates if a loop closure has been detected
	size_t									m_lc_old_kf_id;							//!< ID of the old KF for the loop closure
	deque<TKeyFrameID> 						m_localmap_center_ids;					//!< Contains the IDs of the localmap centers for each KF
	map<TKeyFrameID,size_t> 				m_submap_kfs_from_localmap_center;		//!< Contains the number of KF that a certain localmap contains
	
	deque< set<TKeyFrameID> >				m_kf_localmap_center_ID;						//!< Contains the IDs of the localmap centers for each KF
	map< TKeyFrameID, set<TKeyFrameID> >	m_localmap_kf_IDs;							//!< Contains the number of KF that a certain localmap contains
	
	pose_t m_initial_kf_pose;														//!< Initial estimation of the pose of the added KF wrt the previous one

public: 
	/** Constructor */
	mySRBA() : 
		m_lc(false), 
		m_lc_old_kf_id(0), 
		m_localmap_center_ids(), 
		m_submap_kfs_from_localmap_center(), 
		m_kf_localmap_center_ID(), 
		m_localmap_kf_IDs(),
		m_initial_kf_pose(pose_t()) 
	{}

	/** Sets/unsets flag for loop closure */
	void loopClosureDetected( bool _lc = true ) { 
		m_lc = _lc; 
	}
	
	/** Sets ID of the old KF for a loop closure */
	void setLoopClosureOldID( size_t _id ) { 
		m_lc_old_kf_id = _id; 
	}

	/** Sets the initial pose of the KF to be added */
	inline void setInitialKFPose( const pose_t & pose ) { 
		m_initial_kf_pose = pose; 
	}

	/** Indicates if the input KF ID correspond to a localmap center */
	inline bool isKFLocalmapCenter( const TKeyFrameID & kf_id ) const { 
		return m_localmap_kf_IDs.find(kf_id) != m_localmap_kf_IDs.end(); 
	} // end -- isKFLocalmapCenter

	/** Gets the ID of the localmap center for the input KF ID */
	inline TKeyFrameID getLocalmapCenterID( const TKeyFrameID & kf_id ) const { 
		ASSERT_( kf_id < m_kf_localmap_center_ID.size() ); 
		return *m_kf_localmap_center_ID[kf_id].rbegin(); 
	} // end -- getLocalmapCenterID

	/** Prints the number of KFs in each localmap */
	inline void dumpKfsInLocalmaps() {
		cout << "Kfs within localmaps:" << endl;
		for( map< TKeyFrameID, set<TKeyFrameID> >::iterator it = m_localmap_kf_IDs.begin(); it != m_localmap_kf_IDs.end(); ++it )
		{
			cout << "	Localmap " << it->first << "(" << it->second.size() << " kfs)" << endl;
			for( set<TKeyFrameID>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 )
				cout << "		--> " << *it2 << endl; 
		} // end-for
	} // end -- dumpNumberOfKFs

	/** Prints the localmap centers for all the defined KFs */
	inline void dumpLocalmapCenters() {
		cout << "Localmap centers:" << endl;
		for( size_t i = 0; i < m_kf_localmap_center_ID.size(); i++ )
		{
			cout << "	Kf " << i << endl;
			for( set<TKeyFrameID>::iterator it = m_kf_localmap_center_ID[i].begin(); it != m_kf_localmap_center_ID[i].end(); ++it )
				cout << "		--> " << *it << endl;
		}
	} // end -- dumpLocalmapCenters

	/** Gets the number of KFs for a certain localmap ID */
	inline size_t getNumberOfKFsForLocalMap( const TKeyFrameID & kf_localmap_center_id ) 
	{ 
		return m_localmap_kf_IDs.find(kf_localmap_center_id) != m_localmap_kf_IDs.end() ? 
			m_localmap_kf_IDs[kf_localmap_center_id].size() :
			-1;
	} // end -- getNumberOfKFsForLocalMap

private:
	/******************************* MAIN METHOD  *******************************/
	/** Implements the edge-creation policy, by default depending on "parameters.edge_creation_policy" if the user doesn't re-implement
	this virtual method. * See tutorials for examples of how to implement custom policies. */
	virtual void edge_creation_policy(
		const TKeyFrameID							new_kf_id,
		const traits_t::new_kf_observations_t		& obs,
		std::vector<TNewEdgeInfo>					& new_k2k_edge_ids )
	{
		//	:: this method should not be called for the first KF
		ASSERT_( new_kf_id >= 1 )

		//	:: set KF#0 to have itself as its base
		if( new_kf_id == 1 )
		{
			set<TKeyFrameID> aux; aux.insert(0);
			m_kf_localmap_center_ID.push_back( aux );
			m_localmap_kf_IDs[0] = set<TKeyFrameID>();
		}

		//	:: get sRBA state
		rba_problem_state_t & my_rba_state = this->get_rba_state();

		const size_t MINIMUM_OBS_TO_LOOP_CLOSURE = parameters.ecp.min_obs_to_loop_closure;
		const size_t SUBMAP_SIZE = parameters.ecp.submap_size; // In # of KFs
		// although submaps have variable sizes from the time a loop is closed, by default they have SUBMAP_SIZE

		//	:: get the current localmap base id
		TKeyFrameID currentLocalmapBaseId = 
				isKFLocalmapCenter(new_kf_id-1) ? 
				new_kf_id-1 : 
				getLocalmapCenterID(new_kf_id-1);

		//	:: two cases: 
		const size_t NUM_KFS_LOCALMAP = getNumberOfKFsForLocalMap( currentLocalmapBaseId );
		ASSERT_( NUM_KFS_LOCALMAP >= 0 )
		if( NUM_KFS_LOCALMAP < SUBMAP_SIZE-1 )
		{
			//	:: not a localmap base --> just add the new kf to the current localmap
			//	:: set the current localmap center as the center for the new kf
			set<TKeyFrameID> aux; aux.insert(currentLocalmapBaseId);
			m_kf_localmap_center_ID.push_back( aux );

			//	:: create the edge
			TNewEdgeInfo nei;
			nei.has_approx_init_val = false; // Filled in below
			nei.id = this->create_kf2kf_edge( new_kf_id, TPairKeyFrameID( currentLocalmapBaseId, new_kf_id ), obs );

			if( NUM_KFS_LOCALMAP == 0 )
			{
				// This is the first KF after a new center, so if we add an edge to it we must be very close:
#ifdef SRBA_WORKAROUND_MSVC9_DEQUE_BUG
				my_rba_state.k2k_edges[nei.id]->inv_pose = m_initial_kf_pose;	//pose_t();
#else
				my_rba_state.k2k_edges[nei.id].inv_pose = m_initial_kf_pose;	// pose_t();
#endif
			}			
			else
			{
				// Idea: the new KF should be close to the last one.
#ifdef SRBA_WORKAROUND_MSVC9_DEQUE_BUG
				my_rba_state.k2k_edges[nei.id]->inv_pose = my_rba_state.k2k_edges[nei.id-1]->inv_pose;
#else
				my_rba_state.k2k_edges[nei.id].inv_pose.inverseComposeFrom(my_rba_state.k2k_edges[nei.id-1].inv_pose,m_initial_kf_pose);
#endif
			}
			new_k2k_edge_ids.push_back(nei);

			//	:: insert this new kf id into the list of kfs for the current localmap
			m_localmap_kf_IDs[currentLocalmapBaseId].insert(new_kf_id);

		} // end-if-localmap-base
		else
		{
			//	:: localmap base --> more than one edge can be added

			//	:: go thru all observations and for those already-seen LMs, check the distance between their base KFs and (i_id):
			//	:: make a list of base KFs of my new observations, ordered in descending order by # of shared observations:
			typedef std::multimap<size_t,TKeyFrameID,std::greater<size_t> > my_base_sorted_lst_t;

			base_sorted_lst_t obs_for_each_base_sorted;
			srba::internal::make_ordered_list_base_kfs<traits_t,typename rba_engine_t::rba_problem_state_t>(obs, my_rba_state, obs_for_each_base_sorted);
				
			//	:: make vote list for each central KF:
			map<TKeyFrameID,size_t>  obs_for_each_area;
			for( base_sorted_lst_t::const_iterator it = obs_for_each_base_sorted.begin(); it != obs_for_each_base_sorted.end(); ++it )
			{
				const size_t      num_obs_this_base = it->first;
				const TKeyFrameID base_id = it->second;

				const TKeyFrameID thisLocalmapCenter = getLocalmapCenterID(base_id);
				obs_for_each_area[thisLocalmapCenter] += num_obs_this_base;
			}
				
			//	:: sort by votes:
			my_base_sorted_lst_t   obs_for_each_area_sorted;
			for( map<TKeyFrameID,size_t>::const_iterator it = obs_for_each_area.begin(); it != obs_for_each_area.end(); ++it )
			{
				obs_for_each_area_sorted.insert( make_pair(it->second,it->first) );
			}

			//	:: go thru candidate areas:
			for( base_sorted_lst_t::const_iterator it = obs_for_each_area_sorted.begin(); it != obs_for_each_area_sorted.end(); ++it )
			{
				const size_t      num_obs_this_base = it->first;
				const TKeyFrameID central_kf_id = it->second;

				// Create edges to all these central KFs if they're too far:

				// Find the distance between "central_kf_id" <=> "new_kf_id"
				const TKeyFrameID from_id = new_kf_id;
				const TKeyFrameID to_id   = central_kf_id;

				rba_problem_state_t::TSpanningTree::next_edge_maps_t::const_iterator it_from = my_rba_state.spanning_tree.sym.next_edge.find(from_id);

				topo_dist_t  found_distance = numeric_limits<topo_dist_t>::max();

				if (it_from != my_rba_state.spanning_tree.sym.next_edge.end())
				{
					const map<TKeyFrameID,TSpanTreeEntry> & from_Ds = it_from->second;
					map<TKeyFrameID,TSpanTreeEntry>::const_iterator it_to_dist = from_Ds.find(to_id);

					if (it_to_dist != from_Ds.end())
						found_distance = it_to_dist->second.distance;
				}
				else
				{
					// The new KF doesn't still have any edge created to it, that's why we didn't found any spanning tree for it.
					// Since this means that the KF is aisolated from the rest of the world, leave the topological distance to infinity.
				}

				if( found_distance >= parameters.srba.max_optimize_depth )
				{
					if( num_obs_this_base >= MINIMUM_OBS_TO_LOOP_CLOSURE )
					{
						// The KF is TOO FAR: We will need to create an additional edge:
						TNewEdgeInfo nei;

						nei.id = this->create_kf2kf_edge(new_kf_id, TPairKeyFrameID( central_kf_id, new_kf_id ), obs);
						nei.has_approx_init_val = false; // Will need to estimate this one

						new_k2k_edge_ids.push_back(nei);
							
						//	:: set the local map center as my localmap center
						if( new_kf_id < m_kf_localmap_center_ID.size() )
							m_kf_localmap_center_ID[new_kf_id].insert(central_kf_id);
						else
						{
							set<TKeyFrameID> aux; aux.insert(central_kf_id);
							m_kf_localmap_center_ID.push_back( aux );
					}	

					//	:: update localmaps kf ids
					m_localmap_kf_IDs[central_kf_id].insert(new_kf_id);
					my_rba_state.k2k_edges[nei.id].inv_pose = nei.id == 0 ? pose_t() : my_rba_state.k2k_edges[nei.id-1].inv_pose;
						
					}
					else
					{
						//if( this->m_verbose_level >= 1 ) cout << "[edge_creation_policy] Skipped extra edge " << central_kf_id <<"->"<<new_kf_id << " with #obs: "<< num_obs_this_base << " for too few shared obs!" << endl;
					}
				} // end-distance
			} // end-for
			// Recheck: if even with the last attempt we don't have any edge, it's bad:
			ASSERTMSG_(new_k2k_edge_ids.size()>=1, mrpt::format("Error for new KF#%u: no suitable linking KF found with a minimum of %u common observation: the node becomes isolated of the graph!", static_cast<unsigned int>(new_kf_id),static_cast<unsigned int>(MINIMUM_OBS_TO_LOOP_CLOSURE) ))
			m_localmap_kf_IDs[new_kf_id] = set<TKeyFrameID>();
		}

	} // end-edge_creation_policy

};