
import cv2
import numpy as np

class CogDetector:
    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0

        main_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(main_contour, returnPoints=False)

        if hull is None or len(hull) < 3:
            return 0

        defects = cv2.convexityDefects(main_contour, hull)
        return len(defects) if defects is not None else 0
