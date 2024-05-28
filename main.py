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
import time
import numpy as np
import glib as gl
from PIL import Image
import os
from ultralytics import YOLO
import torch

dir = 'output/'
img_mouseX, img_mouseY, dpt_mouseX, dpt_mouseY = 0, 0, 0, 0

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Your script description")
    # Add arguments here
    parser.add_argument("filename", help="The name of the file to process")
    parser.add_argument('--detect-humans', action='store_true', help='Enable human detection')
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

def show_loop_frequency(time):
    """
    Function to show the loop frequency
    """
    if not hasattr(show_loop_frequency, "old_time"):
        show_loop_frequency.old_time = time  # it doesn't exist yet, so initialize it
    else:   
        loop_frequency = 1 / (time - show_loop_frequency.old_time)
        show_loop_frequency.old_time = time
        print("Loop frequency: ", loop_frequency)

def convert_fov(image, image_goal):

    # # Calculate the new size. This assumes that the aspect ratio of the FOV corresponds to the aspect ratio of the image.
    # new_size = (int(img.shape[1] * (69/87)), int(img.shape[0] * (42/58)))

    # # Resize the image
    # img_resized = cv2.resize(img, new_size, interpolation = cv2.INTER_AREA)

    new_width = 69
    new_height = 42

    # Calculate cropping dimensions
    original_width, original_height = image.shape[1], image.shape[0]
    target_aspect_ratio = new_width / new_height
    original_aspect_ratio = original_width / original_height

    if original_aspect_ratio > target_aspect_ratio:
        # Crop horizontally
        crop_width = int(original_height * target_aspect_ratio)
        crop_height = original_height
    else:
        # Crop vertically
        crop_width = original_width
        crop_height = int(original_width / target_aspect_ratio)

    # Calculate cropping coordinates
    left = (original_width - crop_width) // 2
    top = (original_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    # Crop the image
    img_cropped = image[top:bottom, left:right]

    # Resize the cropped image to the same size as the second picture
    

    # Save the new image
    return img_cropped



def main():
    """
    Main function where the logic of the script is written.
    """
    args = parse_args()
    print(f"Filename: {args.filename}")
    fps_delay = 1#int(1000 / 25)
    # Get the images
    images, depth = get_frames(args.filename)
    #invert depth
    
    if len(depth) > 0:
        print("png:", images[0].shape, "depth: ",depth[0].shape)
        depth = [cv2.bitwise_not(img) for img in depth]
    else:
        print("depth list is empty")
        depth = images
        #QuickFix
    

    
    #Global variables
    mouse_positions = []
    img_no = 90
    
    # Check if a compatible GPU is available, otherwise use CPU
    device = torch.device("cuda")
    # Load a model
    model_path_ovir = 'model/best.pt'
    model_path_hum = 'model/best_human_detect.pt'
    model_ovir = YOLO(model_path_ovir)  # load a custom model
    model_hum = YOLO(model_path_hum)  # load a custom model
    # Move the model to the GPU
    model_hum.to(device)
    model_ovir.to(device)
    threshold = 0.5

    # Set door detectors, 
    door_straight_ahead = gl.DoorDetector(100,5)

    ''' CLASS ZoneChecker(12, 100) testing'
    zone1_checker = gl.ZoneChecker(12, 100)
    zone2_checker = gl.ZoneChecker(12, 200)
    zone3_checker = gl.ZoneChecker(12, 400)


    IN WHILE LOOP:
    img, obstacle, distance = zone1_checker.check(dpth)
    print("Obsticele? - ", obstacle, "at distance: ", distance)
    '''
    
    ''' CLASS Align_images_with_mouse_clicks(3) testing'''
    align = gl.Align_images_with_mouse_clicks(8)
    #cv2.imshow('depth1', depth[img_no])
    #cv2.imshow('color1', images[img_no])
    #get oMat2D_t fro file if it exists
    if os.path.exists("aligned/oMat2D_t.npy"):
        oMat2D_t = np.load("aligned/oMat2D_t.npy")
    else:
        img_aligned, oMat2D_t = align.align_and_resize_images(images[img_no], depth[img_no])
        cv2.imshow('Aligned', img_aligned)
        print("oMat2D_t: ", oMat2D_t)
        np.save("aligned/oMat2D_t.npy", oMat2D_t) 
        
    #save and aline images
    for i, img in enumerate(depth):
        depth[i] = align.warp_affine(images[i], depth[i], oMat2D_t)
        cv2.imwrite(f'aligned/depth{i}.png', depth[i])

    # Set the mouse callback function for the window
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', cursor_coordinates_image)
    cv2.namedWindow('Depth')
    cv2.setMouseCallback('Depth', cursor_coordinates_depth)
    

    while True:
        for img_origin, dpth_origin in zip(images, depth):
            
            
            """IMAGE PREPARATION ------------------------------------------------------------------"""
            #   Convert the images to grayscale
            img = img_origin#cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
            dpth = cv2.cvtColor(dpth_origin, cv2.COLOR_BGR2GRAY)
            #   do gauss filter
            #img = cv2.GaussianBlur(img, (15, 15), 15)
            #dpth = cv2.GaussianBlur(dpth, (15, 15), 15)
            #   display text of pixel value at the mouseX, mouseY
            cv2.putText(img, str(img[img_mouseY, img_mouseX]), (img_mouseX, img_mouseY), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)
            cv2.putText(dpth, str(dpth[dpt_mouseY, dpt_mouseX]), (dpt_mouseX, dpt_mouseY), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)
            #print("MouseX: ", mouseX, "MouseY: ", mouseY, "Value: ", img[mouseY, mouseX], "\033[F")
            """Testiranje funkcije za transformacijo FOV testing
            original_width, original_height = dpth.shape[1], dpth.shape[0]
            crop_width, crop_height = img.shape[1], img.shape[0]
            # Calculate cropping coordinates
            left = (original_width - crop_width) // 2
            top = (original_height - crop_height) // 2
            right = left + crop_width
            bottom = top + crop_height
            # Crop the image
            dpth = dpth[top:bottom, left:right]
            dpth = convert_fov(dpth, img)
            dpth_origin = convert_fov(dpth_origin, img_origin)"""
            




            """DETECTION BOX ------------------------------------------------------------------
            - detekcija samo v območju kamor se bo premaknil robot. Območje se lahko prilagodi glede na  planirano trajektorijo. """
            # define the region of interest 480x640
            img_cut = img#[0:480, 100:540]


            """OBJECT / HUMAN DETECTION------------------------------------------------------------------"""
            frame = img_cut
            frame = gl.fromat_for_model(frame)
            # Move the tensor to the GPU
            frame = frame.to(device)
            if args.detect_humans:
                results_hum = model_hum(frame)[0]
                results_hum = results_hum.cpu()
            else:
                results_hum = None  # Or some other default value
                
            results_ovir = model_ovir(frame)[0]
            results_ovir = results_ovir.cpu()

            results_boxes = results_ovir.boxes.data.tolist()
            results_names = results_ovir.names

            if args.detect_humans:
                results_boxes = torch.cat((results_ovir.boxes.data, results_hum.boxes.data), dim=0).tolist()
                results_names = {**results_ovir.names, **results_hum.names}

            frame = frame.cpu()
            frame = gl.format_for_cv(frame)
            i = 0

            for result in results_boxes:
                x1, y1, x2, y2, score, class_id = result
                if score > threshold:
                    """OBJECT DISTANCE -----------------------------------------------------------------"""
                    #calculate center of rectangle from x1, y1, x2, y2
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    #put text distance to the object nex to the name
                    distance = gl.get_distance(dpth, x, y)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, str(distance), (int(x1), int(y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, results_names[int(class_id)].upper(), (int(x1), int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    #ToDo: ce enablas human detection ti vse ovire preimenuje v HUMAN??? vupsie
                    """WARNING pogoji-----------------------------------------------------------------"""
                    if distance < 12:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                        cv2.putText(frame, str(distance), (int(x1), int(y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, results_names[int(class_id)].upper(), (int(x1), int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    #ToDo: ce enablas human detection ti vse ovire preimenuje v HUMAN??? vupsie
                    elif distance < 4:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

            """NOTAROAD DETECTION------------------------------------------------------------------"""
            #TODO: rad bi dobo tk segmentirano sliko ko je v PDF Izziv RV 2023-2024.pdf str13 desna slika
            #frame = img_cut
            """dodaj kodo"""
            #segmented image

            
            # """HUMAN DETECTION------------------------------------------------------------------"""
            # # Detect people in the image
            # boxes, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
            # # Draw bounding boxes around detected people
            # for (x, y, w, h) in boxes:
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


            """DOOR DETECTION------------------------------------------------------------------"""
            #TODO: detekcija dela sam na dvigalu, na dviznih vratah pa ne
            #if user preses c key, check for obstacles
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # Check for obstacles in the zone
                img, obstacle, distance = door_straight_ahead.check(dpth)
                print("Obsticele? - ", obstacle, "at distance: ", distance)
                # Draw the zone on the image
                cv2.rectangle(frame, (frame.shape[1] // 2 - door_straight_ahead.z, 0), (frame.shape[1] // 2 + door_straight_ahead.z, frame.shape[0]), (0, 255, 0), 2)
                # Draw the distance to the obstacle
                cv2.putText(frame, f"Distance to obstacle: {distance}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                if obstacle:
                    cv2.putText(frame, "Obstacle detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "No obstacles detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)





            """DISPLAYING IMAGES ------------------------------------------------------------------"""
            # Display the images
            cv2.imshow('Image', frame)
            cv2.imshow('Depth', dpth)
            #save the images to folder
            if not os.path.exists(dir):
                os.makedirs(dir)
            cv2.imwrite(f'output/img{i}.png', frame)
            i+=1
            #dispay image size
            print(img.shape)
            #display_frames(img_origin, dpth_origin, img, dpth)
            key = cv2.waitKey(fps_delay)
            show_loop_frequency(time.time())
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