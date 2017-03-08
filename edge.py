import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    abs_sobel = np.abs(sobel)
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return grad_binary

def mag_threshold(image, sobel_kernel=3, thresh=(0, 255)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    mag = np.sqrt(sobelx * sobelx + sobely * sobely)
    mag = np.uint8(255.0 * mag / np.max(mag))

    # Apply threshold
    mag_binary = np.zeros_like(mag)
    mag_binary[(mag >= thresh[0]) & (mag <= thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(dir)
    dir_binary[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    return dir_binary

def color_threshold(image, thresh=(0, 255)):
    binary = np.zeros_like(image)
    binary[(image > thresh[0]) & (image <= thresh[1])] = 1
    return binary

def edges(image, visualize=False):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)
    hchannel = hls[:,:,0]
    #plt.imshow(hchannel, cmap='gray')
    #plt.show()
    lchannel = hls[:,:,1]
    #plt.imshow(lchannel, cmap='gray')
    #plt.show()
    schannel = hls[:,:,2]
    #plt.imshow(schannel, cmap='gray')
    #plt.show()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 200))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 200))
    magnitude = mag_threshold(image, sobel_kernel=3, thresh=(20, 200))
    direction = dir_threshold(image, sobel_kernel=3, thresh=(0.7, 1.3))

    grad = np.zeros_like(image)
    grad[((gradx == 1) & (grady == 1)) | ((magnitude == 1) & (direction == 1))] = 1

    color = color_threshold(schannel, thresh=(150, 255))

    binary = np.zeros_like(grad)
    binary[(color == 1) | (grad == 1)] = 1

    if visualize:
        return np.dstack((np.zeros_like(image), grad, color))
    else:
        binary = np.zeros_like(grad)
        binary[(grad == 1) | (color == 1)] = 1
        return binary

if __name__ == '__main__':
    img = mpimg.imread('debug/frame0001.png')
    #img = mpimg.imread('test_images/test1.jpg')
    print(img.dtype)
    foo = edges(img, True)
    plt.imshow(foo)
    #img = cv2.cvtColor(foo, cv2.COLOR_RGB2BGR)
    #cv2.imwrite( 'output_images/edges.png')
    plt.show()
