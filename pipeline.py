import pickle
from perspective import transform
from edge import edges
from camera import undistort
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

class Window:
    def __init__(self, y1, y2, x, margin=100, minpix=50):
        self.margin = margin
        self.minpix = minpix
        self.y1, self.y2, self.x = y1, y2, x

    # This method returns the indices of the nonzero arrays that are inside the window
    def inside(self, nonzero):
        window_inds = (
            (nonzero[0] >= self.y1) & (nonzero[0] < self.y2) &
            (nonzero[1] >= self.x - self.margin) & (nonzero[1] < self.x + self.margin)
        ).nonzero()[0]
        return window_inds

class Line:
    def __init__(self, h, w, margin):
        self.fits = deque([], 5)
        self.margin = margin
        self.h = h
        self.w = w

        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def inside(self, nonzero):
        #fit = self.fits[0]
        fit = np.average(self.fits, axis=0)
        inds = (
            (nonzero[1] > (fit[0]*(nonzero[0]**2) + fit[1]*nonzero[0] + fit[2] - self.margin)) &
            (nonzero[1] < (fit[0]*(nonzero[0]**2) + fit[1]*nonzero[0] + fit[2] + self.margin))
        )
        return inds

    def curvature(self, fit=None):
        ym_per_pix = 27 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        points = self.points(fit)
        y = points[:, 1]
        x = points[:, 0]
        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        return int(((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0]))

    def position(self):
        points = self.points()
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        x = points[np.max(points[:, 1])][0]
        return np.absolute((self.w // 2 - x) * xm_per_pix)

    def add(self, leftx, lefty):
        if len(leftx) > 0:
            fit = np.polyfit(lefty, leftx, 2)
            self.fits.appendleft(fit)

    def points(self, fit=None):
        y = np.linspace(0, self.h - 1, self.h)
        if fit == None:
            fit = np.average(self.fits, axis=0)
        return np.stack((fit[0] * y ** 2 + fit[1] * y + fit[2], y)).astype(np.int).T

    def current(self):
        y = np.linspace(0, self.h - 1, self.h)
        fit = self.fits[0]
        return (fit[0]*y**2 + fit[1]*y + fit[2]), y

    def current_curve(self):
        return self.fits[0]

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

        # Fit a second order polynomial to each
        self.left_fit.add(leftx, lefty)
        self.right_fit.add(rightx, righty)


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
        self.left_fit.add(leftx, lefty)
        self.right_fit.add(rightx, righty)

        if self.bad_curve():
            print('refinding-window')
            self.initialize_lanes(frame)


        frame[:250, :, :] = frame[:250, :, :] * .4


        # display overhead edge detection
        binary = self.draw_edges(binary_warped)
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
        self.draw_text(frame, "position left: {:.1f} right: {:.1f}".format(self.left_fit.position(), self.right_fit.position()),text_x,140)

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

        # plt.imshow(result)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()

        return out_img


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

    def bad_curve(self):
        curv_diff = self.left_fit.curvature() - self.right_fit.curvature()
        pos_diff = self.right_fit.position() - self.left_fit.position()

        # If the curvature of the lane lines doesn't make sense perform window
        # algorithm again
        if abs(curv_diff) > 80:
            return True

        # if lane lines are too close together perform window algorithm again
        if pos_diff > 0.1:
            return True

        return False

"""
img = mpimg.imread('test_images/test1.jpg')
tracker = Tracker(img)
image = tracker.process(img)
plt.imshow(image)
plt.show()
"""

video_name = "project_video"
video_output_name = video_name + '_annotated.mp4'
video = VideoFileClip(video_name + ".mp4")
tracker = Tracker(video.get_frame(0))
video_output = video.fl_image(tracker.process)
video_output.write_videofile(video_output_name, audio=False)
