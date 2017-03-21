import pickle
from perspective import transform
from edge import edges
from camera import undistort
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from window import Window
from line import Line
import calcs

class Tracker:
    def __init__(self, frame, nwindows=9):
        calibration = pickle.load(open('calibration.p','rb'))
        self.mtx = calibration['mtx']
        self.dist = calibration['dist']

        self.nwindows = nwindows
        self.margin = 100
        self.minpix = 50
        self.count = 0
        self.file = open('log.txt', 'w')

        (self.h, self.w, _) = frame.shape
        self.left_fit = Line(self.h, self.w, self.margin)
        self.right_fit = Line(self.h, self.w, self.margin)

        self.initialize_lanes(frame)

    def preprocess(self, frame):
        undist = undistort(frame, self.mtx, self.dist)
        edge = edges(undist)
        binary_warped, _ = transform(edge)
        return binary_warped

    def birds_eye_view(self, frame):
        undist = undistort(frame, self.mtx, self.dist)
        img, _ = transform(undist)
        return img

    def initialize_lanes(self, frame):
        undist = undistort(frame, self.mtx, self.dist)
        edge = edges(undist)
        binary_warped, self.MI = transform(edge)
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/self.nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for i in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            l_window = Window(self.h - (i+1)* window_height, self.h - i*window_height, leftx_current, self.margin, self.minpix)
            r_window = Window(self.h - (i+1)* window_height, self.h - i*window_height, rightx_current, self.margin, self.minpix)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = l_window.inside(nonzero)
            good_right_inds = r_window.inside(nonzero)

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzero[1][good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzero[1][good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzero[1][left_lane_inds]
        lefty = nonzero[0][left_lane_inds]
        rightx = nonzero[1][right_lane_inds]
        righty = nonzero[0][right_lane_inds]

        dist = self.add_curves(leftx, lefty, rightx, righty)
        if dist != 0:
            print('distances', dist)

        # Fit a second order polynomial to each
        self.left_fit.add(leftx, lefty)
        self.right_fit.add(rightx, righty)

    def add_curves(self, leftx, lefty, rightx, righty):
        if len(leftx) == 0:
            return False
        if len(rightx) == 0:
            return False
        fitx = np.polyfit(lefty, leftx, 2)
        fity = np.polyfit(righty, rightx, 2)

        #method = cv2.CV_CONTOURS_MATCH_I1
        method = 1
        dist = cv2.matchShapes(fitx, fity, method, 0)

        return dist

    def save_frame(self, frame):
        #print('saving frame', self.count)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite("debug/frame"+str(self.count).zfill(4)+".png", img)

    def save_edges(self, frame):
        #print('saving edges ', self.count)
        img = undistort(frame, self.mtx, self.dist)
        img = edges(img, True)
        img, _ = transform(img)
        img = img * 255
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("debug/grad"+str(self.count).zfill(4)+".png", img)

    def process(self, frame, debug=True):
        self.count += 1

        self.save_frame(frame)
        self.save_edges(frame)

        binary_warped = self.preprocess(frame)
        nonzero = binary_warped.nonzero()

        left_lane_inds = self.left_fit.inside(nonzero)
        right_lane_inds = self.right_fit.inside(nonzero)

        leftx = nonzero[1][left_lane_inds]
        lefty = nonzero[0][left_lane_inds]

        rightx = nonzero[1][right_lane_inds]
        righty = nonzero[0][right_lane_inds]

        # Fit a second order polynomial to each
        if self.bad_curve(leftx, lefty, rightx, righty):
            self.initialize_lanes(frame)
        else:
            self.left_fit.add(leftx, lefty)
            self.right_fit.add(rightx, righty)


        frame[:250, :, :] = frame[:250, :, :] * .4

        # display overhead edge detection
        binary = self.draw_edges(binary_warped)

        # write curve values for frame
        self.file.write(str(self.left_fit.current_curve()))
        self.file.write(str(self.right_fit.current_curve()))
        self.file.write("\n")
        self.file.flush()

        cv2.imwrite("debug/edges"+str(self.count).zfill(4)+".png", binary)
        binary = cv2.resize(binary, (0, 0), fx=0.3, fy=0.3)
        frame[20:20 + binary.shape[0], 20:20 + binary.shape[1], :] = binary

        # display overhead lanes
        top = self.draw_overhead_lane(self.birds_eye_view(frame))
        img = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)
        cv2.imwrite("debug/top"+str(self.count).zfill(4)+".png", img)
        top = cv2.resize(top, (0, 0), fx=0.3, fy=0.3)
        frame[20:20 + top.shape[0], 20 + 20 + top.shape[1]:20 + 20 + top.shape[1] * 2, :] = top

        text_x = 20 * 3 + top.shape[1] * 2
        self.draw_text(frame, "radius left: {} right: {}".format(self.left_fit.curvature(), self.right_fit.curvature()),text_x,80)

        left_position = self.left_fit.position()
        right_position = self.right_fit.position()
        self.draw_text(frame, "position left: {:.1f}m right: {:.1f}m".format( left_position, right_position),text_x,140)

        lane_mid = (self.left_fit.bottomx() + self.right_fit.bottomx()) / 2
        car_mid = (self.w / 2)
        offset = (car_mid - lane_mid) * 3.7 / 700 * 0.77
        self.draw_text(frame, "lane offset: {:.1f}m".format(offset), text_x, 200)


        return self.draw_lane(frame)

    def draw_edges(self, binary):
        out_img = (np.dstack((binary, binary, binary))*255).astype(np.uint8)

        # Generate x and y values for plotting
        left_fitx, y = self.left_fit.current()
        right_fitx, y = self.right_fit.current()

        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return result

    def draw_lane(self, frame):
        # Create an image to draw the lines on
        overlay = np.zeros_like(frame).astype(np.uint8)


        left_points = self.left_fit.points()
        right_points = self.right_fit.points()
        points = np.vstack((left_points, np.flipud(right_points)))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        overlay = cv2.warpPerspective(overlay, self.MI, (frame.shape[1], frame.shape[0]))
        # Combine the result with the original image
        image = cv2.addWeighted(frame, 1, overlay, 0.3, 0)
        return image

    def draw_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    def draw_overhead_lane(self, frame):
        # Create an image to draw the lines on
        overlay = np.zeros_like(frame).astype(np.uint8)

        left_points = self.left_fit.points()
        right_points = self.right_fit.points()
        points = np.vstack((left_points, np.flipud(right_points)))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        #overlay = cv2.warpPerspective(overlay, self.MI, (frame.shape[1], frame.shape[0]))
        # Combine the result with the original image
        image = cv2.addWeighted(frame, 1, overlay, 0.3, 0)
        return image

    def lane_width(self, left, right):
        return right - left


    # return true if the radius of curvature is widely different between lanes
    # or the lane width is too wide or too thin, when this happens we want to
    # perform the window search algorithm
    def bad_curve(self, leftx, lefty, rightx, righty):
        print(len(leftx), len(rightx))
        if len(leftx) == 0 or len(rightx) == 0:
            return True
        points_left = calcs.points(self.h, leftx, lefty)
        points_right = calcs.points(self.h, rightx, righty)

        curve_left = calcs.curvature(points_left)
        curve_right = calcs.curvature(points_right)
        #print('curve', curve_left, curve_right)

        position_left = calcs.position(points_left)
        position_right = calcs.position(points_right)
        #print('position', position_left, position_right)
        #print('lane width', self.lane_width(position_left, position_right))

        curve_diff = abs(curve_left - curve_right)
        pos_diff = abs(position_left - position_right)

        #print('curve', curve_diff, 'position', pos_diff)

        # If the curvature of the lane lines doesn't make sense perform window
        # algorithm again
        if abs(curve_diff) > 100:
            return True

        # if lane lines are too close together perform window algorithm again
        if pos_diff > 4.0 or pos_diff < 3.4:
            return True

        return False

"""
img = mpimg.imread('test_images/test1.jpg')
tracker = Tracker(img)
image = tracker.process(img)
plt.imshow(image)
plt.show()
"""

video_name = "challenge_video"
video_output_name = video_name + '_annotated.mp4'
video = VideoFileClip(video_name + ".mp4")
tracker = Tracker(video.get_frame(0))
video_output = video.fl_image(tracker.process)
video_output.write_videofile(video_output_name, audio=False)
