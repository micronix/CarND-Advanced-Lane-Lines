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
