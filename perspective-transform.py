import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle


fname = 'test2.jpg'
img = mpimg.imread('test_images/'+fname)
#img = mpimg.imread('debug/frame0001.png')
#print(img.shape)
(h, w, _) = img.shape


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

print(src, dst)

M = cv2.getPerspectiveTransform(src, dst)

# save the perspective matrix
pickle.dump(M, open('pmatrix.p','wb'))


# Run Code on test images
def drawLines(img, points, color=(255, 0, 0), thickness=8):
    num = len(points)
    for i in range(num):
        p1 = (points[i][0], points[i][1])
        p2 = (points[(i+1) % num][0], points[(i+1) % num][1])
        cv2.line(img,p1,p2,color,thickness)

f, (ax1, ax2) = plt.subplots(1, 2)

# draw original image with lines
original = img.copy()
drawLines(original, src)
ax1.imshow(original)
ax1.set_title('original')

# draw warped image with lines
warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
drawLines(warped, dst)
ax2.imshow(warped)
ax2.set_title('warped')

plt.show()
