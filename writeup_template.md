**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted3.png "Undistorted"
[image2]: ./output_images/undistorted2.png "Undistorted"
[image3]: ./test_images/test1.jpg "Original"
[image4]: ./output_images/undistorted1.jpg "Undistorted"
[image5]: ./output_images/edges.png "Edges Example"
[image6]: ./output_images/perspective1.png "Warp Example"
[image7]: ./output_images/perspective2.png "Warp Example"
[image8]: ./output_images/perspective3.png "Warp Example"
[image9]: ./output_images/lane.png "Lane"
[image10]: ./output_images/lane2.png "Lane"
[image11]: ./output_images/lane3.png "Lane"
[image12]: ./output_images/bad1.png "Lane"
[image13]: ./output_images/bad2.png "Lane"
[image14]: ./output_images/bad3.png "Lane"
[video1]: ./project_video_annotated.mp4 "Video"

---
## Camera Calibration

The code for this step is contained in the file camera-calibration.py. We save the distortion matrix and the calibration matrix into a pickle file called calibration.p. This pickle file is then loaded in our pipeline.

The calibration procedure is pretty simple and took most of the code from the lectures.

**Assumptions:**

Chessboard is flat, hence the z=0 for all the 3D points. The 3D points have coordinates (0,0,0), (0,1,0), ..., (9,6,0).

**Procedure:**

We create an list of points (0,0,0), (0,1,0), ..., (9,6,0), these will be the same for all the calibration images because we are using the same chessboard. The code to generate these points is:

```python
nx, ny = 9, 6
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
```
These points will be reused for each image. We then call the function `cv2.findChessboardCorners` on each calibration image, which gives us 2D points. We create a list of 3D points `objpoints` by simply copying the created points `objp` and a list of 2D points `imgpoints` by appending the result from `cv2.findChessboardCorners`.

We then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to a chessboard image and a road test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]
![alt text][image2]

You can clearly see the effects by looking at the right edge of the image near the back of the white car and a little up on the trees.

## Edge Detection

The code for this part of the pipeline is found in the file edge.py.

I used a combination of gradient and color thresholds on the S channel of an HLS image.For the gradients I used a sobel kernel of size 3 with the following thresholds:

| Type | Min | Max |
|:-------------:|:-------------:|:-----:|
| X | 50 | 200 |
| Y | 50 | 200 |
| Magnitude | 115 | 100 |
| Direction | 0.7 | 1.3 |

A final gradient binary was created, where a pixel is on if both gradient in x direction and gradient in y direction are on, or if gradient magnitude is on and gradient direction is on.

I also added a threshold on the S channel between 90 and 255, and the R-channel from the RGB space with a threshold between 200 and 255. The following is a visualization of this, with the red being the R-channel binary, green being the S-channel binary, and blue being the gradient logic (gradx && grady || grad_mag && grad_dir)

![alt text][image5]

This is what seemed to give the clearest lane lines. In my previous submission I was relying mainly on gradients, a suggestion that was give to me was to rely more on colr thresholds, which seemed to improve things quite a bit.

## Perspective Transform

The code that calculates my perspective matrix is in the file **perspective-transform.py**. I manually found the source and destination points. The example writeup provided a good starting point.

The 4 points for my source and destination that I ended up with are:

```
src = np.float32(
    [[100, h],
    [(w / 2) - 90, h / 2 + 110],
    [(w / 2 + 90), h / 2 + 110],
    [w-100, h]])

dst = np.float32(
    [[100, h],
    [100, 0],
    [w - 100, 0],
    [w - 100, h]])
```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 100, 720      | 100, 720        |
| 550, 470      | 100, 0      |
| 730, 470     | 1180, 0      |
| 1180, 720      | 1180, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]
![alt text][image7]

Below is a curved test image.

![alt text][image8]

#### Curve fitting

`pipeline.py` contains the code to track the lane. The pipeline functions as follows:

- **initialize_lanes** This finds the lanes using the sliding window method.

  We first need to find all the edges and look at it from a bird's eye view:
  - Undistort image
  - Convert to a binary image using gradients and color thresholds
  - Perform a perspective transform on the image

  We then look at a histogram of the bottom half of the image. The maximum points in the histogram are assumed to be the beginning of the lane lines. My implementation used 9 windows. The algorithm worked as follows:
  - Center a right window on the max of the histogram on the left side of the image
  - Find all the nonzero points inside the window
  - Take the mean x value of those points
  - Center the next window on the x-mean of the previous window
  - Perform for the remaining windows.
  - This will give us a set of points inside the windows that we can use to perform a curve fitting.

  I used a second degree polynomial to fit the points

Once I had a polynomial I didn't need to find the windows again in the next frame, instead I used the curve with the same margin size as the windows to find points to fit in the second frame. This simplified finding the lane points.

When the algorithm had problems finding the lane lines, in some cases the right lane would move to the left and find the same points as the right lane. In order to prevent this from happening, I added sanity checks to the lane lines.

**Sanity Checks**

If one lane line curves to the left and the other one curves to the right, or one curves much more than the other curve there is probably an error. Also if the lane lines are too close to each other, one lane line probably switched from the left to the right or vice versa.

I also assumed that the bottom of the lane lines won't change too much between frames. If the x position of the bottom of the lane lines changes by more than 15 pixels then that probably means that we might have shadows or weird effects. When this happens, the fit doesn't get added to the list of lane lines. I implemented this suggestion after looking at some suggstions from other students in the class.

**Curve Averaging**

To smooth imperfections in the images from frame to frame, I average the polynomal fit over the past 5 frames. This means that a strange frame will not make us completly start searching windows from scratch.

In file pipeline.py I created a Tracker class that tracks the lane lines by first. There is a method called *initialize_lanes* that uses the sliding window method to track the lane lines. There is a second method called *process* which uses the already found
The code for curve fitting can be found in the file pipeline.py on lines 128 to 185.

The function *bad_curve* determines if the curves are bad and we should start the window algorithm again. I also have a `Line` class and a `Window` class. The `Line` class stores the last 5 polynomal fits and can average them. The method `inside` finds the points inside the region defined by the polynomal fit, the margin lets us know how far away from the curve to search for points.

The `Window` class also has an `inside` method to find points inside the window.

Below is an image showing the identified lane.

![alt text][image9]

#### 5. Radius of Curvature & position of car

I did this in the `Line` class between lines 63-78 in file pipeline.py. I performed the same procedure that was outlined in the *Measuring Curvature* section. We first need to convert the curve points to world space and then fit another line. These new coefficients will be used in the formula for calculating curvature.

#### 6. Lane Identification

When changing the perspective of my image I save an invese matrix to undo the warped perspective. Using the polynomal fit I generated points for the left and right lanes. These points were then used to draw a filled polynomal. I then performed the inverse perspective transform and added it to the original image. The code to add the lane to the image is found in file `pipeline.py` line 292-306.

Below are a few examples of my algorithm identifying the lane.

![alt text][image9]
![alt text][image10]
![alt text][image11]

---

### Pipeline (video)


Here's a [link to my video result](./project_video_annotated.mp4)

---

### Discussion

The biggest problem that I had initially was that the right lane would jump from the right to the left and become the same line. From that point on the right lane would not be found.

This problem was solved by doing a sanity check and performing the lane finding algorithm again.

Overall when there are no cars in the lane performs well. I did try the algorithm on the challenge videos and found some issues that we need to come up with in order for the algorithm to be better.

The possible challenges are:

**Glare**

Glare on the windshield generates too many gradient points. Here is an example screenshot:

![alt text][image13]

One way to deal with this is could be to add a glare filter to the lense of the camera. I don't know much about lenses but I think that this might be possible. Maybe also detect this type of glare and simply drop any fits when there is too much glare at the bottom. As a driver if I would see this, I would just continue the direction I'm going and wait for the glare to go away.

**Sharp Turns**

The hard challenge video had really hard turns. We can see that right lane is not visible with out birds-eye perspective because it is so far to the right. The left lane was all the way into the space of the right lane. I also noticed that the second degree polynomial was not a good fit in some cases.

![alt text][image12]

One possible thing to do would be to fit a higher order degree polynomal as well in order to detect if there are sharp upcoming turns in the road.

**Multi-colored Roads**

Another problem is when a road is paved with darker asphalt in one area and not as dark asphalt in another area as shown in the image below:

![alt text][image14]

The image could probably be improved if we tune our parameters of distance between lines, however, if the pavement was closer to the left lane it would still not work. One possible fix could be to perform the binary gradient on colors that lane lines couuld be.
