#!/usr/bin/env python


#import pyrealsense2 as rs
#import numpy as np
#import cv2
import os, shutil,json
#import socket
from camera import Camera


f = open('calibration.json')
data = json.load(f)
ImagesForCalibration = data['calibrationInfo']
f.close()

f = open('pose.json')
data = json.load(f)
ImageForPose = data['poseInfo']
f.close()

f = open('distortion.json')
data = json.load(f)
ImageForDistortion = data['distortionInfo']
f.close()




# ImageForPose = ''
rows = 6
columns = 9
scale = 30
# imageWithDistortionDir = 'distortion'
# imageName = 'with_distortion_Color.png'
# PoseDir = 'modeloPoseImgs'



myCamera = Camera(ImagesForCalibration,ImageForPose)
cameraInfo = myCamera.getCameraInfo()
print(cameraInfo)
myImages = myCamera.loadImagesForCalib()
ret_val, camMat, distortion,rotVec,transVec  =myCamera.calibrateCamera(myImages,rows,columns,scale)
mean_error = myCamera.computeReprojError(rotVec,transVec,camMat,distortion)
myCamera.intrinsicToJSON(camMat,mean_error,cameraInfo[0],cameraInfo[1], 'intrinsic')
newCamMatrix = myCamera.undistortImg(camMat,distortion, ImageForDistortion, 'Distortion_0.png')
myCamera.intrinsicToJSON(newCamMatrix,mean_error,cameraInfo[0],cameraInfo[1], 'intrinsic_no_distortion')
# # # Get the rotation and translation vector
# # # transvec is the position (in mm in our case) of the world origin in camera co-ords.
[rotVec,transVec] = myCamera.getPose(newCamMatrix,distortion,scale,rows,columns,ImageForPose,'Pose_0.png')        
rotMatrix = myCamera.getRotationMatrix(rotVec)
myCamera.extrinsicToJSON(transVec,rotMatrix,cameraInfo[0],cameraInfo[1])
