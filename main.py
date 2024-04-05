#!/usr/bin/env python3
"""
KOMENTARJI / UPRASANJA
- kk resize img in depth da bodo enake oblike
- a mormo naredet pointcloud
- tsti slide ko pise da dobimo neko ROS kodo ... tste kode ni
- 6.poglavje, geometrijske preslikave - Afina
    - ma profesor kodo za to da določas znacilke.
    - lahk to vajo doma nardimo pa mamo v tstem termino neka drugega

"""

import logging
import argparse
import glob
import cv2
import numpy as np
import glib as gl
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
# from tensorflow.keras.models import Sequential
#import open3d as o3d
img_mouseX, img_mouseY, dpt_mouseX, dpt_mouseY = 0, 0, 0, 0

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Your script description")
    # Add arguments here
    parser.add_argument("filename", help="The name of the file to process")
    # parser.add_argument(...)
    return parser.parse_args()

def get_frames(filename):
    """
    Function to load all the images in folder 
    """
    images = [cv2.imread(file) for file in glob.glob(filename + "/png/*.png")]
    depth = [cv2.imread(file) for file in glob.glob(filename + "/depth/*.png")]

    return images, depth

def display_frames(img_origin, dpth_origin, img, dpth):
    # Convert grayscale images to 3-channel images
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(dpth.shape) == 2:
        dpth = cv2.cvtColor(dpth, cv2.COLOR_GRAY2BGR)

    # Concatenate the images horizontally
    combined_origin = np.hstack((img_origin, dpth_origin))
    combined_end = np.hstack((img, dpth))
    # concatenate the images vertically
    combined = np.vstack((combined_origin, combined_end))
    cv2.imshow('Images', combined)

def cursor_coordinates_image(event,x,y,flags,param):
    global img_mouseX,img_mouseY
    if event == cv2.EVENT_MOUSEMOVE:
        img_mouseX,img_mouseY = x,y
def cursor_coordinates_depth(event,x,y,flags,param):
    global dpt_mouseX,dpt_mouseY
    if event == cv2.EVENT_MOUSEMOVE:
        dpt_mouseX,dpt_mouseY = x,y



def main():
    """
    Main function where the logic of the script is written.
    """
    args = parse_args()
    print(f"Filename: {args.filename}")
    fps_delay = int(1000 / 25)
    # Get the images
    images, depth = get_frames(args.filename)
    print("png:", images[0].shape, "depth: ",depth[0].shape)

    #Global variables
    mouse_positions = []

    ''' CLASS ZoneChecker(12, 100) testing'
    zone1_checker = gl.ZoneChecker(12, 100)
    zone2_checker = gl.ZoneChecker(12, 200)
    zone3_checker = gl.ZoneChecker(12, 400)


    IN WHILE LOOP:
    img, obstacle, distance = zone1_checker.check(dpth)
    print("Obsticele? - ", obstacle, "at distance: ", distance)
    '''
    ''' CLASS Align_images_with_mouse_clicks(3) testing
    #align = gl.Align_images_with_mouse_clicks(3)
    #align.select_points(images[0])
    #resize the images to the same size as the depth image
    images = [cv2.resize(img, (depth[0].shape[1], depth[0].shape[0])) for img in images]
    #inverse of depth image
    depth[10] = cv2.bitwise_not(depth[10])
    cv2.imshow('depth1', depth[10])
    cv2.imshow('color1', images[10])
    img_aligned = align.align_and_resize_images(images[10], depth[10])
    cv2.imshow('Aligned', img_aligned)
    '''
    
    # Set the mouse callback function for the window
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', cursor_coordinates_image)
    cv2.namedWindow('Depth')
    cv2.setMouseCallback('Depth', cursor_coordinates_depth)

    while True:
        for img_origin, dpth_origin in zip(images, depth):
             #display text of pixel value at the mouseX, mouseY
            cv2.putText(img, str(img[img_mouseY, img_mouseX]), (img_mouseX, img_mouseY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(dpth, str(dpth[dpt_mouseY, dpt_mouseX]), (dpt_mouseX, dpt_mouseY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #print("MouseX: ", mouseX, "MouseY: ", mouseY, "Value: ", img[mouseY, mouseX], "\033[F")

            
            """IMAGE PREPARATION ------------------------------------------------------------------"""
            # Convert the images to grayscale
            img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
            dpth = cv2.cvtColor(dpth_origin, cv2.COLOR_BGR2GRAY)
            # do gauss filter
            img = cv2.GaussianBlur(img, (15, 15), 15)
            dpth = cv2.GaussianBlur(dpth, (15, 15), 15)
      
            """IMAGE PROCESSING ------------------------------------------------------------------"""












            """DISPLAYING IMAGES ------------------------------------------------------------------"""
            # Display the images
            cv2.imshow('Image', img)
            cv2.imshow('Depth', dpth)
            # display_frames(img_origin, dpth_origin, img, dpth)
            key = cv2.waitKey(fps_delay)
            if key == ord('q'):  # Exit loop if 'q' key is pressed
                break
        else:
            continue
        break

    cv2.destroyAllWindows()


    


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run the main function
    main()