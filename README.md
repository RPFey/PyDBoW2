# PyDBoW2

This a python binding for the DBoW2 implementation in ORB-SLAM2. I use their original implementation so that you can easily benchmark their loop detection results. 

## Installation

### Requirements

1. Run `git submodule init && git submodule update` to clone the pybind11 and DLib into current repo.

2. This repo needs OpenCV. Set `export OpenCV_DIR=/path/to/opencv/cmakeconfig`. This path could be the `build` folder of the opencv.

3. Build DLib repo. 

```bash
cd thirdparty/DLib
mkdir build && cd build && cmake .. && make
```

4. run `python setup.py build_ext --inplace`. You can find examples in `test.py`.

## TODO

1. I notice that the ORB Feature in ORB-SLAM2 is different from the OpenCV implementation.

2. Add init.py

## From ORB-SLAM2

You should have received this DBoW2 version along with ORB-SLAM2 (https://github.com/raulmur/ORB_SLAM2).
See the original DBoW2 library at: https://github.com/dorian3d/DBoW2
All files included in this version are BSD, see LICENSE.txt

We also use Random.h, Random.cpp, Timestamp.pp and Timestamp.h from DLib/DUtils.
See the original DLib library at: https://github.com/dorian3d/DLib
All files included in this version are BSD, see LICENSE.txt
