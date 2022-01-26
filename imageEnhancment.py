"""
Read the input
Convert to gray
Threshold and invert as a mask
Optionally apply morphology to clean up any extraneous spots
Anti-alias the edges
Convert a copy of the input to BGRA and insert the mask as the alpha channel
Save the results
"""

import os
import cv2
import numpy as np

class imageEnhancment():
    def __int__(self):
        self.img = []
        self.gray = []
        self.mask = []
        self.result = []
        self.kernel = np.ones((3, 3), np.uint8)
        print(self.kernel)

    def deleteBackGround(self, im):
        self.img = im

        # convert to graky
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshold input image as mask
        self.mask = cv2.threshold(self.gray, 250, 255, cv2.THRESH_BINARY)[1]

        # negate mask
        self.mask = 255 - self.mask
        kernel = np.ones((3, 3), np.uint8)

        # apply morphology to remove isolated extraneous noise
        # use borderconstant of black since foreground touches the edges
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)

        # anti-alias the mask -- blur then stretch
        # blur alpha channel
        self.mask = cv2.GaussianBlur(self.mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)

        # linear stretch so that 127.5 goes to 0, but 255 stays 255
        self.mask = (2 * (self.mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)

        # put mask into alpha channel
        self.result = self.img.copy()
        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2BGRA)
        self.result[:, :, 3] = self.mask
        return self.result


    def display(self):
        cv2.imshow("INPUT", self.img)
        cv2.imshow("GRAY", self.gray)
        cv2.imshow("MASK", self.mask)
        cv2.imshow("RESULT", self.result)

    def saveResult(self, path):
        cv2.imwrite(path, self.result)

    def QPixmapToArray(self, pixmap):
        ## Get the size of the current pixmap
        size = pixmap.size()
        h = size.width()
        w = size.height()

        ## Get the QImage Item and convert it to a byte string
        qimg = pixmap.toImage()
        byte_str = qimg.bits().tobytes()

        ## Using the np.frombuffer function to convert the byte string into an np array
        img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w, h, 4))

        return img

if __name__ == "__main__":
    t = imageEnhancment()
    # load image
    img = cv2.imread('1.png')
    img = cv2.resize(img, (300, 300))

    t.deleteBackGround(img)
    t.display()
    t.saveResult(os.getcwd()+"/33.png")
    cv2.waitKey(0)
    cv2.destroyAllWindows()