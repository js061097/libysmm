#!/bin/bash


cd ..
export LD_LIBRARY_PATH=/opt/intel-gpu/gmmlib/lib64/:/opt/intel-gpu/igc/lib64/:/opt/intel-gpu/neo/lib64/:$LD_LIBRARY_PATH
mkdir build && cd build
cmake ./../
make
cd ./../examples
export LD_LIBRARY_PATH=./../build/:$LD_LIBRARY_PATH 
