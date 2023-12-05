#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2
import os, shutil,json
import socket
from camera import Camera

'''
Author: Nitin Mahadeo
Email: nitin@robotonrails.com
Company : Robot on Rails

Version 1.0 - 08/31/23 - Description: Capture images to be used for calibration - Single camera
Version 1.1 - 11/16/23 - Added code to create a folder to store images with the given computer hostname 
                        + function for each to help with modularity

Reference: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py

'''

def getDistortionFolderName():
    # We append the computer name to the Distortion folder
    hostName = socket.gethostname()
    distImageDir = hostName + 'DistortionImgs'
    return distImageDir


def distortionDirCheck(distImageDir):

    if os.path.isdir(distImageDir):
        print(distImageDir + ' Directory exists! - Deleting Folder and its contents')
        shutil.rmtree(distImageDir)

    os.mkdir(distImageDir)
    print("Creating a new folder to store images for Distortion " + distImageDir +" \n")


def saveToJSON(fname,fnameInfo,myString):
    myConfig = {
        fnameInfo  : myString,
        }

    with open(fname +'.json', 'w', encoding='utf-8') as f:
            json.dump(myConfig, f, ensure_ascii=False, indent=4)
            print(myString +" info written to file.\n")

def captureImgs(distImageDir):

    counter = 0

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    cameraInfo = Camera()

    print(cameraInfo.getCameraInfo() )

    found_rgb = False

    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
        
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # Depth and  Color Stream
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:

            # Wait for frames
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            cv2.imshow('RealSense Live Capture',color_image)

            key = cv2.waitKey(1)

            if key == ord('q'): # quit
                print("Quitting program for image capture.\n")
                break
            elif key == ord('c'): # capture
                cv2.imwrite(distImageDir + "/Distortion_" + str(counter) + ".png", color_image)
                counter = counter + 1
                print("Writing color image " + str(counter) + " to dir:" + distImageDir)

    finally:

        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    
    distortionImageDir = getDistortionFolderName()
    saveToJSON('distortion','distortionInfo',distortionImageDir)
    distortionDirCheck(distortionImageDir)
    captureImgs(distortionImageDir)
