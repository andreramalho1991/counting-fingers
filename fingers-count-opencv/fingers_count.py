import cv2
import numpy as np
from sklearn.metrics import pairwise

class HandGestureDetector:
    def __init__(self):
        self.background = None
        self.accumulated_weight = 0.5
        self.roi_top = 20
        self.roi_bottom = 300
        self.roi_right = 300
        self.roi_left = 600
        self.num_frames = 0

    def calc_accum_avg(self, frame):
        if self.background is None:
            self.background = frame.copy().astype("float")
            return None
        cv2.accumulateWeighted(frame, self.background, self.accumulated_weight)

    def segment(self, frame, threshold=25):
        diff = cv2.absdiff(self.background.astype("uint8"), frame)
        _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        hand_segment = max(contours, key=cv2.contourArea)
        return thresholded, hand_segment

    def count_fingers(self, thresholded, hand_segment):
        conv_hull = cv2.convexHull(hand_segment)
        top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
        bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
        left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
        right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
        cX = (left[0] + right[0]) // 2
        cY = (top[1] + bottom[1]) // 2
        distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
        max_distance = distance.max()
        radius = int(0.7 * max_distance)
        circumference = (2 * np.pi * radius)
        circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
        cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
        circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
        contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count = 0
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
            limit_points = ((circumference * 0.25) > cnt.shape[0])
            if out_of_wrist and limit_points:
                count += 1
        return count

    def run(self):
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            roi = frame[self.roi_top:self.roi_bottom, self.roi_right:self.roi_left]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            if self.num_frames < 60:
                self.calc_accum_avg(gray)
                if self.num_frames <= 59:
                    cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.imshow("Finger Count", frame_copy)
            else:
                hand = self.segment(gray)
                if hand is not None:
                    thresholded, hand_segment = hand
                    cv2.drawContours(frame_copy, [hand_segment + (self.roi_right, self.roi_top)], -1, (255, 0, 0), 1)
                    fingers = self.count_fingers(thresholded, hand_segment)
                    cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.imshow("Thesholded", thresholded)
            cv2.rectangle(frame_copy, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom), (0,0,255), 5)
            self.num_frames += 1
            cv2.imshow("Finger Count", frame_copy)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cam.release()
        cv2.destroyAllWindows()

