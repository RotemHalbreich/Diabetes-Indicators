"""
A class for line classification: [y=mx+b]
Attributes: m (the slope), b (the y-intercept), a boolean called "up" that says if
the line classifies all the points above it as positive and below it as negative.
"""


class ClassifyLine:

    def __init__(self, x1, y1, x2, y2, up=True):  # the default classification is 1 -> above the line, -1 -> below it
        self.up = up
        if x1 != x2:
            self.m = (y2 - y1) / (x2 - x1)  # calculate m (slope)
            self.b = y1 - self.m * x1  # find b
            self.indicator = 0  # if line isn't parallel to y axis
        else:
            self.indicator = 1  # if line is parallel to y axis
            self.x = x1

    def classify(self, x, y):
        if self.indicator == 0:
            result = self.m * x + self.b  # compute the result of line given x
            if result >= y:  # the point (x,y) is below the line
                return 1 if self.up else -1
            else:  # the point (x,y) is above the line
                return -1 if self.up else 1
        else:
            if x > self.x:
                return 1 if self.up else -1
            else:
                return -1 if self.up else 1