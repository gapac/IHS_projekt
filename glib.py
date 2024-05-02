import numpy as np
import cv2
import AffineTransFunctions as at
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
class ZoneChecker:
    def __init__(self, zone, tresh):
        self.z = zone
        self.t = tresh

    def check(self, dpth):
        # Create my ZoneChecker object
        # Crop depth image to 100 vertical pixels in the center
        start_index = dpth.shape[1] // 2 - self.t
        end_index = dpth.shape[1] // 2 + self.t
        zone = dpth[:, start_index:end_index] 
        print("z1 cropped: ",zone[0].shape, "\033[F")

        #if any pixels in the zone are less than 12, then there is an obstacle
        obstacle = np.any(zone < 12)
        #distance to the obstacle
        distance = np.min(zone)

        return zone, obstacle, distance
    


class Align_images_with_mouse_clicks:
    def __init__(self, num_points=3):
        #create as may variables as the number of points
        self.num_points = num_points


    # Define a callback function for mouse events
    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Store the point
            param.append((x, y))


    def select_points(self, image):
        # Create a list to store the selected points
        selected_points = []
        # Set the mouse callback function for the window
        # Display the images
        cv2.imshow('Image', image)
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.select_point, selected_points)
        #pause
        cv2.waitKey(10)
        while len(selected_points) < self.num_points:
            # ... (rest of your code)
            #put red dots on selected_points
            for point in selected_points:
                cv2.circle(image, point, 5, (0, 0, 255), -1)
            # Print the selected points
            if selected_points:
                print("Selected point: ", selected_points[-1], "\033[F")
            # Display the image and wait for a key press
            cv2.imshow('Image', image)
            cv2.waitKey(1)
        # Store the selected points
        return selected_points
        

    def warp_affine(self, img1, img2, oMat2D_t):
        # Assuming oMat2D_t is a 2D transformation matrix
        M = oMat2D_t[:2]  # We only need the top two rows for cv2.warpAffine
        # Apply the transformation
        img2_aligned = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]))
        return img2_aligned

    def align_and_resize_images(self, img1, img2):
        points_img = np.array(self.select_points(img1), dtype=np.float32)
        points_dpt = np.array(self.select_points(img2), dtype=np.float32)
        iPtsRef = points_img
        iPtsMov = points_dpt
        #save the points to a file
        np.save("aligned/points_img.npy", points_img)  
        np.save("aligned/points_dpt.npy", points_dpt) 
        # doloci matriko preslikave
        oMat2D_t, oErr = at.alignICP(iPtsRef, iPtsMov, iEps=1e-6, iMaxIter=50, plotProgress=True)
        #warpAffine
        img2_aligned = self.warp_affine(img1, img2, oMat2D_t)
        '''old# Find affine transformation matrix
        #M = cv2.getAffineTransform(points_img, points_dpt)

        # Apply affine transformation to img2
        #img2_aligned = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
    
        # # Find affine transformation matrix
        # M = cv2.getAffineTransform(points_img, points_dpt)

        # # Apply affine transformation to img2
        # img2_aligned = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]))
       

        # # Find homography matrix
        # H, _ = cv2.findHomography(points_img, points_dpt)

        # # Warp img2 to img1
        # height, width, _ = img1.shape
        # img2_aligned = cv2.warpPerspective(img2, H, (width, height))'''
        return img2_aligned, oMat2D_t

    

class DoorDetector:
    def __init__(self, zone, tresh):
        self.z = zone
        self.t = tresh

    def check(self, dpth):
        # Create my ZoneChecker object
        # Crop depth image to 100 vertical pixels in the center
        start_index = dpth.shape[1] // 2 - self.z
        end_index = dpth.shape[1] // 2 + self.z
        zone = dpth[:, start_index:end_index] 
        #show the zone
        cv2.imshow("zone", zone)
        print("zx cropped: ",zone[0].shape, "\033[F")
        print("zy cropped: ",zone[1].shape, "\033[F")

        distance = get_distance(dpth, zone.shape[0] // 2, zone.shape[1] // 2, kernel_size = zone.shape[0])

        #if any pixels in the zone are less than treshold, then there is an obstacle
        obstacle = distance < self.t
        
        return zone, obstacle, distance
    


def get_distance(depth, x, y, kernel_size=8):
    #fancy avereging...
    x = int(x)
    y = int(y)
    offset = kernel_size // 2

    # Extract a 8x8 region around the pixel
    region = depth[y - offset:y + offset, x - offset:x + offset]
    
    # Apply the box filter to the region
    filtered_region = cv2.boxFilter(region, -1, (8, 8))

    #draw the region
    cv2.rectangle(depth, (x - offset, y - offset), (x + offset, y + offset), (255, 0, 0), 1)

    #get the distance
    distance = filtered_region[offset,offset]
    print("Distance: ", distance, "\033[F")
    #remap values to meters?
    distance = np.interp(distance, (0, 255), (100, 0))

    return distance



def fromat_for_model(frame):
    # Normalize the tensor to the range 0.0-1.0
    frame = frame / 255.0
    # Convert the numpy array to a PyTorch tensor
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        # Add a batch dimension
    frame = frame.unsqueeze(0)
    # Resize to dimensions divisible by 32
    frame = F.interpolate(frame, size=(480, 640))
    return frame

def format_for_cv(frame):
     # Remove the batch dimension
    frame = frame.squeeze(0)
    # Convert the tensor to cv2 image
    frame = frame.permute(1, 2, 0).numpy()
    frame = frame * 255.0
    # Convert the frame to uint8
    frame = np.array(frame, dtype=np.uint8)
    # Convert the frame to BGR format
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame