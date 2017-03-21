import numpy as np

# returns a curve that fits the set of given points
def points(h, x, y):
    fit = np.polyfit(y, x, 2)
    y = np.linspace(0, h - 1, h)
    return np.stack((fit[0] * y ** 2 + fit[1] * y + fit[2], y)).astype(np.int).T

# returns radius of curvature in meters for points
def curvature(points):
  ym_per_pix = 27 / 720  # meters per pixel in y dimension
  xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

  y = points[:, 1]
  x = points[:, 0]
  fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
  return int(((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0]))

# returns distance from left in meters for point at bottom
def position(points):
  xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
  x = points[np.max(points[:, 1])][0]
  x = x * xm_per_pix * 0.77
  return x
