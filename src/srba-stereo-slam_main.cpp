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
#include "CSRBAStereoSLAMEstimator.h" // <-- this includes all the rest of needed
//#include <iterator>

TGeneralOptions	general_options; // global variable

// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------
int main(int argc, char **argv)
{
	try
	{
	    if( argc < 2 )
	    {
	        cout << "Use: srba-stereo-slam configFile" << endl;
	        return -1;
	    }
		
		std::string INI_FILENAME(argv[1]);
		ASSERT_FILE_EXISTS_( INI_FILENAME )
		CConfigFile config( INI_FILENAME );

	    // get general parameters
		general_options.loadFromConfigFile( config );							// general app options

		// create srba estimator
		CSRBAStereoSLAMEstimator srba_stereo_estimator;
		
		// and initialize it
		srba_stereo_estimator.initialize( config );
	
		if( general_options.from_step != 0 && general_options.to_step != 0 && 
			general_options.to_step < general_options.from_step )
			THROW_EXCEPTION( "Parameter 'toStep' is lower than 'fromStep'" );

	    general_options.dumpToConsole();
		srba_stereo_estimator.srba_options.dumpToConsole();
		srba_stereo_estimator.performStereoSLAM(); // MAIN ENTRY
	    cout << "SRBA Stereo SLAM process done!" << endl;

		// srba_stereo_estimator.saveOutputToFile();

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
