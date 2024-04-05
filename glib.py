import numpy as np
import cv2
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


    def align_and_resize_images(self, img1, img2):
        points_img = np.array(self.select_points(img1), dtype=np.float32)
        points_dpt = np.array(self.select_points(img2), dtype=np.float32)


        # Find affine transformation matrix
        M = cv2.getAffineTransform(points_img, points_dpt)

        # Apply affine transformation to img2
        img2_aligned = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
    
        # # Find affine transformation matrix
        # M = cv2.getAffineTransform(points_img, points_dpt)

        # # Apply affine transformation to img2
        # img2_aligned = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]))
       

        # # Find homography matrix
        # H, _ = cv2.findHomography(points_img, points_dpt)

        # # Warp img2 to img1
        # height, width, _ = img1.shape
        # img2_aligned = cv2.warpPerspective(img2, H, (width, height))

        return img2_aligned

    

