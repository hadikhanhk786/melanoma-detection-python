import os
import json

import cv2
import numpy as np

import features
import color_contour
import active_contour


class Lesion:
    def __init__(self, file_path):
        self.file_path = file_path
        self.base_file, _ = os.path.splitext(file_path)
        self.image = cv2.imread(file_path)
        self.segmented_img = None
        self.hsv_image = None
        self.contour_binary = None
        self.contour_image = None
        self.contour_mask = None
        self.contour_mask_warp = None
        self.color_contour = None
        self.asymmetry_vertical = None
        self.asymmetry_horizontal = None
        self.results = None
        self.value_threshold = 150
        self.hsv_colors = {
            'Blue Gray': [np.array([15, 0, 0]),
                          np.array([179, 255, self.value_threshold]),
                          (0, 153, 0)],  # Green
            'White': [np.array([0, 0, 145]),
                      np.array([15, 80, self.value_threshold]),
                      (255, 255, 0)],  # Cyan
            'Light Brown': [np.array([0, 80, self.value_threshold + 3]),
                            np.array([15, 255, 255]), (0, 255, 255)],  # Yellow
            'Dark Brown': [np.array([0, 80, 0]),
                           np.array([15, 255, self.value_threshold - 3]),
                           (0, 0, 204)],  # Red
            'Black': [np.array([0, 0, 0]), np.array([15, 140, 90]),
                      (0, 0, 0)],  # Black
        }
        self.borders = 2
        self.isImageValid = False
        self.contour = None
        self.max_area_pos = None
        self.contour_area = None
        self.feature_set = []

    def check_image(self):
        try:
            if self.image is None:
                self.isImageValid = False
                return
            if self.image.shape[2] != 3:
                self.isImageValid = False
                return
            self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            self.isImageValid = True
            return True
        except:
            self.isImageValid = False
            return

    def get_mask(self, iterations=9999):
        if self.isImageValid:
            img2 = cv2.copyMakeBorder(self.image, self.borders,
                                      self.borders, self.borders,
                                      self.borders,
                                      cv2.BORDER_CONSTANT,
                                      value=[255, 255, 255])
            self.contour_mask = np.zeros(img2.shape[:2], dtype=np.uint8)
            self.contour_binary = active_contour.run(img2, iterations)
            im_mask, mask_contours, hierarchy = \
                cv2.findContours(self.contour_binary, cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_NONE)
            cnt = len(mask_contours)
            if cnt > 0:
                area = np.zeros(cnt)
                for i in np.arange(cnt):
                    area[i] = cv2.contourArea(mask_contours[i])
                max_area_pos = np.argpartition(area, -1)[-1:][0]
                cv2.drawContours(self.contour_mask, mask_contours,
                                 max_area_pos,
                                 (255, 255, 255),
                                 -1)
            if cnt <= 0:
                return
            self.contour_mask = self.contour_mask[2:-2, 2:-2]
            im_mask, mask_contours, hierarchy = \
                cv2.findContours(self.contour_mask, cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_NONE)
            cnt = len(mask_contours)
            if cnt > 0:
                area = np.zeros(cnt)
                for i in np.arange(cnt):
                    area[i] = cv2.contourArea(mask_contours[i])
                self.max_area_pos = np.argpartition(area, -1)[-1:][0]
                self.contour = mask_contours[self.max_area_pos]
                self.contour_image = np.copy(self.image)
                cv2.drawContours(self.contour_image, mask_contours,
                                 self.max_area_pos,
                                 (255, 0, 0),
                                 2)
                return True
            if cnt <= 0:
                return

    def segment(self):
        ret = self.get_mask()
        if ret is not None:
            self.contour_area = cv2.contourArea(
                self.contour)
            self.segmented_img = cv2.bitwise_and(self.image, self.image,
                                                 mask=self.contour_mask)

    def extract_features(self):
            returnVars = features.extract(
                self.contour_mask, self.contour, self.base_file)
            if len(returnVars) == 0:
                self.feature_set = returnVars
            else:
                self.feature_set = returnVars[0]
                self.asymmetry_horizontal = returnVars[1]
                self.asymmetry_vertical = returnVars[2]

    def get_color_contours(self):
        tolerance = 30
        self.value_threshold = np.uint8(cv2.mean(self.hsv_image)[2]) \
                               - tolerance
        hsv = cv2.cvtColor(self.segmented_img, cv2.COLOR_BGR2HSV)
        no_of_colors = []
        self.color_contour = np.copy(self.image)
        for color in self.hsv_colors:
            #            print color
            cnt = color_contour.extract(self.segmented_img, hsv,
                                        self.hsv_colors[color],
                                        self.contour_area)
            centroid = []
            dist = []
            color_attr = {}
            if len(cnt) > 0:
                for contour in cnt:
                    moments = cv2.moments(contour)
                    if moments['m00'] == 0:
                        print color
                        continue
                    color_ctrd = [int(moments['m10'] / moments['m00']),
                                  int(moments['m01'] / moments['m00'])]
                    color_dist = ((color_ctrd[0] -
                                   self.feature_set['centroid'][
                        0]) ** 2 + (color_ctrd[1] -
                                    self.feature_set['centroid'][
                            1]) ** 2) ** 0.5
                    if color_dist > 0.6 * np.min(
                                    np.array(self.feature_set['diam'][
                                                          -2:]) / 2) \
                            and color == 'White':
                        # White cannot be at the center
                        continue
                    dist.append(color_dist)
                    centroid.append(color_ctrd)
            if len(dist) != 0 and len(centroid) != 0:
                cv2.drawContours(self.color_contour, cnt, -1,
                                 self.hsv_colors[color][2],
                                 2)
                color_attr['color'] = color
                color_attr['centroids'] = centroid
                color_attr['dist'] = np.int0(dist).tolist()
                no_of_colors.append(color_attr)
        self.feature_set['image'] = self.file_path
        self.feature_set['colors_attr'] = no_of_colors
        self.feature_set['no_of_colors'] = len(no_of_colors)

    def save_images(self):
        cv2.imwrite(self.base_file + '.PNG', self.contour_image)
        cv2.imwrite(self.base_file + '_active_contour.PNG',
                                self.contour_binary)
        cv2.imwrite(self.base_file + '_colors.PNG',
                                self.color_contour)
        cv2.imwrite(self.base_file + '_mask.PNG',
                                self.contour_mask)
        cv2.imwrite(self.base_file + '_segmented.PNG',
                                self.segmented_img)
        cv2.imwrite(self.base_file + '_horizontal.PNG',
                                self.asymmetry_horizontal)
        cv2.imwrite(self.base_file + '_vertical.PNG',
                                self.asymmetry_vertical)

    def save_result(self):
        target = open(self.base_file+".json", 'w')
        target.write(json.dumps(self.feature_set,
                                sort_keys=True, indent=2) + '\n')
        target.close()

    def extract_info(self,save=False):
        if self.check_image():
            self.segment()
            self.extract_features()
            self.get_color_contours()
            if save and len(self.feature_set) != 0:
                self.save_images()
                self.save_result()
        else:
            print "Invalid image"
