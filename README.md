## 1.Ubuntu Essential Packages
- pip - sudo apt install python3-pip
- virtual environment - sudo apt install virtualenv
- tmux - sudo apt install tmux
- python link -sudo apt install python-is-python3

## 2. Deploy on Camera
- Create a **virtual environment** and install the above packages via pip using the requirements.txt file

## 3.Packages to install in the virtual environment
pip - pip install pyrealsense2
pip install numpy
pip install opencv-python


## 4. Run the follwing scripts and capture images
- CaptureImagesForCalibration.py - 20 images
- CaptureImagesForDistortion.py - 1 image
- CaptureImagesForPose.py - 1 image
- CaptureImagesForMerge.py - 1 image

## 5.
- Run the camera.py file to compute and save the intrinsic and extrinsic parameters.
 
