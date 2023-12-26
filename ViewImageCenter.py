#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2
from camera import Camera


def captureImgs():

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

            # Draw a circle at the center of the image
            cv2.circle(color_image, (640,360), 5, (0,128,0), -1)
            
            cv2.imshow('RealSense Live Capture',color_image)

            key = cv2.waitKey(1)

            if key == ord('q'): # quit
                print("Quitting program for image capture.\n")
                break

    finally:

        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    
    captureImgs()
