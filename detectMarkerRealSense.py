#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2
import os, shutil,json
import socket
from camera import Camera
from cv2 import aruco

'''
Author: Nitin Mahadeo
Email: nitin@robotonrails.com
Company : Robot on Rails

'''

# dictionary to specify type of the marker
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# detect the marker
param_markers = aruco.DetectorParameters()

def getMarkersFolderName():
    # We append the computer name to the Merge folder
    hostName = socket.gethostname()
    MergeImageDir = hostName + '_ArucoMarkers'
    return MergeImageDir, hostName 


def markerDirCheck(MarkerImgDir):

    if os.path.isdir(MarkerImgDir):
        print(MarkerImgDir + ' Directory exists! - Deleting Folder and its contents')
        shutil.rmtree(MarkerImgDir)

    os.mkdir(MarkerImgDir)
    print("Creating a new folder to store images for Markers " + MarkerImgDir +" \n")


def saveToJSON(fname,myString,marker_IDs, marker_corners,rmat,tvecs):
    
    marker_corners_list = []
    for i in range(len(marker_corners)):
        marker_corners_list.append(marker_corners[i][0].tolist())


    myConfig = {
        #fnameInfo  : myString,
        "marker IDs" : marker_IDs.tolist(),
        "marker corners" : marker_corners_list,
        "rotation matrix" : rmat.tolist(),
        "translation vector" : tvecs.tolist()
        }

    with open(fname +'_'+ myString+'.json', 'w', encoding='utf-8') as f:
            json.dump(myConfig, f, ensure_ascii=False, indent=4)
            print(myString +" info written to file.\n")

def detectMarkers(color_image):
    
    gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )

    return marker_corners, marker_IDs



def getPose(marker_corners, MARKER_SIZE, cam_mat, dist_coef):
        
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(marker_corners, 
                                                                   MARKER_SIZE, cam_mat, 
                                                                   dist_coef)
        return rvecs, tvecs 


def drawMarkers(color_image, marker_corners, marker_IDs):
                # getting conrners of markers
            if marker_corners:
                for ids, corners in zip(marker_IDs, marker_corners):
                    cv2.polylines(
                        color_image, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
                    )
                    
                    corners = corners.reshape(4, 2)
                    corners = corners.astype(int)
                    top_right = corners[0].ravel()
                    cv2.putText(color_image,f"id: {ids[0]}",top_right,
                                cv2.FONT_HERSHEY_PLAIN,
                                1.3,(200, 100, 0),2,cv2.LINE_AA,)



def captureImgs(markerDir, hostName):

    counter = 0

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    cameraInfo = Camera()

    print("Camera Info", cameraInfo.getCameraInfo())

    found_rgb = False

    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
        
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # Depth and  Color Stream
    #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
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

            marker_corners, marker_IDs = detectMarkers(color_image)

            drawMarkers(color_image, marker_corners, marker_IDs)            

            if marker_corners:
                   rvecs, tvecs = getPose(marker_corners, MARKER_SIZE, cam_mat, dist_coef)

                   #rvecs in degrees
                   rvecs = np.degrees(rvecs)
                   print("rvecs in degrees: \n", rvecs) 


                    # rvecs to rotation matrix
                   rmat = cv2.Rodrigues(rvecs[0])[0]

                   print("rmat: \n", rmat)
                   print("rvecs: \n", rvecs)
                   print("tvecs: \n", tvecs)

                   point = cv2.drawFrameAxes(color_image, cam_mat, dist_coef, rvecs[0], tvecs[0], 20, 4)
                   
            
            cv2.imshow('RealSense Live Capture',color_image)

            key = cv2.waitKey(0)

            if key == ord('q'): # quit
                print("Quitting program for image capture.\n")
                break
            elif key == ord('c'): # capture
                cv2.imwrite(markerDir + "/ArucoMarker_" + hostName + str(counter) + ".png", color_image)
                counter = counter + 1
                print("Writing color image " + str(counter) + " to dir:" + markerDir)
                saveToJSON('pose',arucoMarkerDir,marker_IDs, marker_corners,rmat,tvecs)

            elif key == ord('r'): # reset
                counter = 0
                print("Resetting counter to 0.")

    finally:

        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    
    with open('intrinsic_no_distortion.json') as f:
        camera_parameters = json.load(f)
    print(camera_parameters)

    cam_mat = np.array([[camera_parameters["focal length X"], 0, camera_parameters["optical center X"]],  
                       [0, camera_parameters["focal length Y"], camera_parameters["optical center Y"]], 
                       [0, 0, 1]])

    MARKER_SIZE = 105  #mm

    with open('distortion_info.json') as f:
         distCoeffs = json.load(f)
    print(distCoeffs)

    dist_coef = np.array([[distCoeffs["Dist 0"], 
                  distCoeffs["Dist 1"], 
                  distCoeffs["Dist 2"], 
                  distCoeffs["Dist 3"]]])

    arucoMarkerDir, hostName = getMarkersFolderName()
    markerDirCheck(arucoMarkerDir)  
    captureImgs(arucoMarkerDir, hostName)
    
