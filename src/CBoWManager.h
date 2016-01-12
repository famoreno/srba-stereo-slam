#pragma once

// bag of words
#include "DBoW2.h"		// defines Surf64Vocabulary and Surf64Database
#include "DUtils.h"
#include "DUtilsCV.h"	// defines macros CVXX
#include "DVision.h"

// stereo slam keyframe class
#include <mrpt/utils/CConfigFile.h>
#include "CStereoSLAMKF.h"

// namespaces
using namespace DBoW2;
using namespace DUtils;
using namespace DVision;
using namespace mrpt;

/*********************************************
CLASS: Bag of words manager
**********************************************/
class CBoWManager
{
// MEMBERS -------------------------------------
private:
	BriefVocabulary m_voc;	// vocabulary
	BriefDatabase	m_db;	// database

// METHODS -------------------------------------
public: 
	// default constructor
	CBoWManager() {}

	/* loadVocabularyFromConfigFile : Loads the prebuilt vocabulary from a inifile
		[I] config -- Ini file from where to load 'vocabulary' filename
	*/
	inline void loadVocabularyFromConfigFile( const utils::CConfigFile & config, const string & section, const string & param )
	{
		// set vocabulary and create db
		const string VOC_FILENAME = config.read_string(section,param,"",true);
		ASSERT_FILE_EXISTS_( VOC_FILENAME )
		m_voc.load( VOC_FILENAME );
		m_db.setVocabulary( m_voc, true, 5 );
	} // end-loadVocabularyFromConfigFile

	/* insertIntoDB : Inserts a KF into the DB
		[I] kf -- KF that contains in 'm_descriptors_left' the ORB descriptors to insert into the DB
	*/
	inline void insertIntoDB( const CStereoSLAMKF & kf )
	{
		vector<BRIEF::bitset> out;
		m_change_structure_binary( kf.m_descriptors_left, out );
		m_db.add( out );
	} // end -- insertKFIntoDB

	/* queryDB : Queries the DB for the most similar KFs to the input one
		[I] kf			-- KF that contains in 'm_descriptors_left' the ORB descriptors to insert into the DB
		[O] ret			-- Structure with the results of the query
		[i] num_results	-- Number of wanted results (def: 1)
	*/
	inline void queryDB( const CStereoSLAMKF & kf, QueryResults & ret, unsigned int num_results = 1 )
	{ 
		vector<BRIEF::bitset> out;
		m_change_structure_binary( kf.m_descriptors_left, out );
		m_db.query( out, ret, num_results ); 
	} // end -- queryKFInDB

private:
	/* insertIntoDB : Adapts a ORB descriptor matrix to a vector of binary descriptors for using it with BoW
		[I] plain		-- OpenCV Mat with the matrix of ORB descriptors to be converted
		[O] out			-- Vector of bitset to 
	*/
	void m_change_structure_binary( const Mat & plain, vector<BRIEF::bitset> & out )
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
			} // end-for
		} // end-for
	} // end-changeStructureORB
	
}; // end BoWManager