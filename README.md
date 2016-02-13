# SRBA-Stereo-SLAM
**Note:** *Preliminary version*

Library that performs stereo visual SLAM within a Sparser Relative Bundle Adjustment (SRBA) framework. 
More info about SRBA can be found [here](http://www.mrpt.org/srba)  

## Building from sources

### Prerequisites

* CMake (>=2.4.5)
* OpenCV (>=2.4.8, >=3.0.0 not supported due to DBoW2 not supporting it) 
* [DBoW2](https://github.com/dorian3d/DBoW2) (1.0.2) (Check dependencies for this library)
* [MRPT](https://github.com/MRPT/mrpt) (>=1.3.0)
* [SRBA](https://github.com/MRPT/srba) (Header-only library, must be downloaded from GitHub for now)
* [stereo-vo](https://github.com/famoreno/stereo-vo)

### Compiling

This software can be compiled in Windows and GNU/Linux with `gcc` or `clang`. It should also work on OSX but it is untested.

1. Install all prerequisites above. Many of those can be installed in Ubuntu as follows:

* Only for versions older than Ubuntu Wily (15.10):

        sudo add-apt-repository ppa:joseluisblancoc/mrpt
        sudo apt-get update
    
* After that, for all Debian / Ubuntu versions:
    
        sudo apt-get install build-essential cmake libmrpt-dev libopencv-dev

* Download and build [DBoW2] (http://webdiis.unizar.es/~dorian/index.php?p=32) library following instruction therein. This library needs [boost] (http://www.boost.org) library to run (tested with 1.55).
 
* Clone [stereo-vo](https://github.com/famoreno/stereo-vo) and build following instructions therein. Optionally, run `make test`

* Clone the header-only library [SRBA](https://github.com/MRPT/srba). Configure and generate the project with `cmake` so `SRBAConfig.cmake` is generated and, optionally, run `make test` to ensure everything works ok. 

2. Create an empty build directory, invoke `cmake` and build as usual.

        mkdir build
        cd build
        cmake .. -DSRBA_DIR=[PATH_TO_SRBA_BUILD_DIR]



