#!/usr/bin/env python
import pyrealsense2 as rs
import glob
import cv2
import numpy as np
import datetime
import json

'''
Author: Nitin Mahadeo @ Robot on Rails

Version 1.0  -- 11/17/2023  
- Get info of the camera connected to the machine

Version 2.0  -- 11/21/2023   
- Get the calibration matrix and pose - KRT
- Save the KRT to json files
'''

class Camera:

    def __init__(self,imagesForCalibration=None,imagesForPose=None):
        self.device_info = None
        self.serial_number = None
        self.imagesForCalibration = imagesForCalibration
        self.ImageForPose = imagesForPose
        self.points3D = []
        self.points2D = []
        

    def getCameraInfo(self):
        ctx = rs.context()
        
        if len(ctx.devices) == 1:
            for d in ctx.devices:
                self.device_info = d.get_info(rs.camera_info.name)
                self.serial_number = d.get_info(rs.camera_info.serial_number)
                return [self.device_info,self.serial_number]
            
        elif len(ctx.devices) == 0:
                print("No Intel Device connected")
                return False
        
        elif len(ctx.devices) > 1:
                print("More that one Realsense camera found!")
                return False
    
    # Location of images to be used for calibration
    def loadImagesForCalib(self):
        #print(self.imagesForCalibration +'/*.png')
        images = glob.glob(self.imagesForCalibration +'/*.png')
        print(images)
        print("# of images available for calibration: ", len(images))
        return images
    

    def calibrateCamera(self, images, rows, columns, scale):

        # Define the checkerboard 
        CHECKERBOARD = (rows,columns)
        CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

        point3D = np.zeros((1,CHECKERBOARD[0] * CHECKERBOARD[1],3),np.float32)
        point3D[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * scale

        counter = 0
        for currImg in images:

            img =cv2.imread(currImg)
            grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
            # Find the corners in the image
            ret_val, corners = cv2.findChessboardCorners(grayImg,CHECKERBOARD, 
                                                        cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                        cv2.CALIB_CB_FAST_CHECK + 
                                                        cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if ret_val:
                counter  = counter + 1
                self.points3D.append(point3D)
                # Refined cord for a given 2D point
                cornersRefined = cv2.cornerSubPix(grayImg,corners,(11,11),(-1,-1),CRITERIA)
                self.points2D.append(cornersRefined)
                imageCorners = cv2.drawChessboardCorners(img,CHECKERBOARD,cornersRefined,ret_val)


            cv2.imshow('Image with corners',imageCorners)
            cv2.waitKey(100)

        cv2.destroyAllWindows()

        print("# of images used for calibration: ", counter)
        ret_val, camMat, distortion,rotVec,transVec = cv2.calibrateCamera(self.points3D,self.points2D,grayImg.shape[::-1],None,None)
        return [ret_val, camMat, distortion,rotVec,transVec]
    
    # Calculate the reprojection error
    def computeReprojError(self,rotVec,transVec,camMat,distortion):
        mean_error = 0
        for i in range(len(self.points3D)):
            imgpoints2, _ = cv2.projectPoints(self.points3D[i], rotVec[i], transVec[i], camMat, distortion)
            error = cv2.norm(self.points2D[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "Total Reprojection Error: {}".format(mean_error/len(self.points3D)) )  
        return mean_error



    def intrinsicToJSON(self,camMat,mean_error,device_info,serial_number,fname):  
        intrinsic = {
        "focal length X"  : camMat[0][0],
        "focal length Y"  : camMat[1][1],
        "optical center X": camMat[0][2],
        "optical center Y" : camMat[1][2],
        "Acquisition Time":datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"),
        "Reprojection Error:":mean_error/len(self.points3D),
        "Camera info:" :device_info,
        "Camera Serial Number:" :serial_number,
        }

        # Write data to file
        with open(fname +'.json', 'w', encoding='utf-8') as f:
            json.dump(intrinsic, f, ensure_ascii=False, indent=4)
        print("Intrinsic info written to file.\n")


    def  undistortImg(self,camMat,distortion, imageWithDistortionDir, imageName):    
        image_with_distort = cv2.imread(imageWithDistortionDir+'/'+imageName)
        h,w = image_with_distort.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camMat,distortion,(w,h),0,(w,h))
        x,y,w,h = roi
        undistorted_img = cv2.undistort(image_with_distort,camMat,distortion,None,newCameraMatrix)
        undistorted_img = undistorted_img[y:y+h,x:x+w]
        cv2.imwrite(imageWithDistortionDir+'/no_distortion_'+imageName,undistorted_img)
        print("Image without distortion written to file\n")
    
        cv2.imshow("Image without distortion", undistorted_img)
        cv2.waitKey(1000)
        return newCameraMatrix
    

    def getPose(self,newCameraMatrix,distortion,scale,rows,cols,imgDirPose,imgName):
        # Pose estimation
        img = cv2.imread(imgDirPose+'/'+imgName)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((rows*cols,3), np.float32)
        objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2) * scale
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

        ret, corners = cv2.findChessboardCorners(gray, (rows,cols),None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, newCameraMatrix, distortion)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, newCameraMatrix, distortion)
            img = self.draw(img,corners2,imgpts)

            tvecs_str ='tvec:' + str(tvecs[0]) + ' ' + str(tvecs[1]) + ' ' + str(tvecs[2])
            self.putText(img,tvecs_str)
            cv2.imshow('img',img)
            cv2.imwrite(imgDirPose +'/poseEstimated_'+imgName,img)
            print("Estimated pose written to file.")
            cv2.waitKey(1000)
            
        return [rvecs,tvecs]



    def draw(self,img, corners, imgpts):
    
        #print('Image Points',imgpts)
        #print('Coners',corners)

        corner = tuple(corners[0].ravel())
        tmp = imgpts[0].ravel()
        tmp2 = imgpts[1].ravel()
        tmp3 = imgpts[2].ravel()

        img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(tmp[0]),int(tmp[1])), (255,0,0), 5)
        img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(tmp2[0]),int(tmp2[1])), (0,255,0), 5)
        img = cv2.line(img, (int(corner[0]),int(corner[1])), (int(tmp3[0]),int(tmp3[1])), (0,0,255), 5)

        return img


    def putText(self,image,myStr):
        org = (50, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 0)  
        # Line thickness of 2 px
        thickness = 1
        # Using cv2.putText() method
        image = cv2.putText(image, myStr, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

        
        image = cv2.putText(image, 'X', (10,20), font, 
                    fontScale, (255, 0, 0), 1, cv2.LINE_AA)
        image = cv2.putText(image, 'Y', (10,50), font, 
                    fontScale, (0, 255, 0), 1, cv2.LINE_AA)
        image = cv2.putText(image, 'Z', (10,80), font, 
                    fontScale, (0, 0, 255), 1, cv2.LINE_AA)

    def getRotationMatrix(self,rotVec):
        np_rodrigues = np.asarray(rotVec[:,:],np.float64)
        print('Rodrigues \n',np_rodrigues)
        rot_matrix = cv2.Rodrigues(np_rodrigues)[0]
        print('Rot Matrix \n',rot_matrix)
        return rot_matrix
    
    # Save extrinsic parameters to JSON file
    def extrinsicToJSON(self,transVec,rotMatrix,device_info,serial_number):
        extrinsic = {
        "Translation Vector X"  : transVec[0][0],
        "Translation Vector Y"  : transVec[1][0],
        "Translation Vector Z"  : transVec[2][0],
        "Rotation Matrix [0,0]"  : rotMatrix[0,0],
        "Rotation Matrix [0,1]"  : rotMatrix[0,1],
        "Rotation Matrix [0,2]"  : rotMatrix[0,2],
        "Rotation Matrix [1,0]"  : rotMatrix[1,0],
        "Rotation Matrix [1,1]"  : rotMatrix[1,1],
        "Rotation Matrix [1,2]"  : rotMatrix[1,2],
        "Rotation Matrix [2,0]"  : rotMatrix[2,0],
        "Rotation Matrix [2,1]"  : rotMatrix[2,1],
        "Rotation Matrix [2,2]"  : rotMatrix[2,2],
        "Acquisition Time":datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"),
        "Camera info:" :device_info,
        "Camera Serial Number:" :serial_number,
        }

        # Write data to file
        with open('extrinsic_info.json', 'w', encoding='utf-8') as f:
            json.dump(extrinsic, f, ensure_ascii=False, indent=4)
        print("Extrinsic info written to file.")


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






