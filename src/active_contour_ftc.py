# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:15:06 2017

@author: ukalwa
"""
import cmath
import numpy as np
import cv2


def get_pixel_points_ellipse_new(img_width, img_height, init_width,
                                 init_height):
    centre_x = int(img_width / 2)
    centre_y = int(img_height / 2)
    axis_x = int(init_width * img_width / 2)
    axis_y = int(init_height * img_height / 2)
    zero_image = np.zeros([img_height, img_width], dtype=np.uint8)
    cv2.ellipse(zero_image,
                (centre_x, centre_y), (axis_x, axis_y),
                0.0, 0.0, 360.0, 255, -1)
    pixel_points = cv2.findNonZero(zero_image)
    interior_points = np.fliplr(np.reshape(pixel_points,
                                           [len(pixel_points), 2]))
    zero_image = cv2.bitwise_not(zero_image)
    pixel_points = cv2.findNonZero(zero_image)
    exterior_points = np.fliplr(np.reshape(pixel_points,
                                           [len(pixel_points), 2]))

    return interior_points, exterior_points


def get_pixel_points_ellipse(img_width, img_height, init_width, init_height,
                             offset, is_filled):
    centre_x = int(img_width / 2)
    centre_y = int(img_height / 2)
    axis_x = int(init_width * img_width / 2)
    axis_y = int(init_height * img_height / 2)
    zero_image = np.zeros([img_height, img_width], dtype=np.uint8)
    if is_filled:
        cv2.ellipse(zero_image,
                    (centre_x, centre_y), (axis_x + offset, axis_y + offset),
                    0.0, 0.0, 360.0, 255, -1)
    else:
        cv2.ellipse(zero_image,
                    (centre_x, centre_y), (axis_x + offset, axis_y + offset),
                    0.0, 0.0, 360.0, 255, 1)
    pixel_points = cv2.findNonZero(zero_image)
    interior_points = np.fliplr(np.reshape(pixel_points,
                                           [len(pixel_points), 2]))
    return interior_points


def get_pixel_points_rectangle(img_width, img_height, init_width, init_height,
                               offset, is_filled):
    pt1_x = int(img_width * (1 - init_width) / 2)
    pt1_y = int(img_height * (1 - init_height) / 2)
    pt2_x = int(img_width * (1 + init_width) / 2)
    pt2_y = int(img_height * (1 + init_height) / 2)
    zero_image = np.zeros([img_height, img_width], dtype=np.uint8)
    if is_filled:
        cv2.rectangle(zero_image, (pt1_x + offset, pt1_y + offset),
                      (pt2_x - offset, pt2_y - offset), 255, -1)
    else:
        cv2.rectangle(zero_image, (pt1_x + offset, pt1_y + offset),
                      (pt2_x - offset, pt2_y - offset), 255, -1)
    pixel_points = cv2.findNonZero(zero_image)
    interior_points = np.fliplr(np.reshape(pixel_points,
                                           [len(pixel_points), 2]))
    return interior_points


def calculate_yuv(r, g, b):
    y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
    u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128
    v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128
    return y, u, v


def signum_function(i):
    if i < 0:
        return -1
    else:
        return 1


def find_common_array(arr1, arr2):
    dims = np.maximum(arr2.max(0), arr1.max(0)) + 1
    out = arr1[~np.in1d(np.ravel_multi_index(arr1.T, dims),
                        np.ravel_multi_index(arr2.T, dims))]
    return out


class ActiveContourFTC:
    """find boundary by active contour based self algorithm"""

    def __init__(self, img, has_ellipse1,
                 init_width1, init_height1, center_x1, center_y1,
                 has_smoothing_cycle1, kernel_length1, sigma1, has_cleaning1,
                 na1, ns1, lambda_out1, lambda_in1, alpha1, beta1, gamma1):

        self.hasEllipse = has_ellipse1
        self.init_width = init_width1
        self.init_height = init_height1
        self.center_x = center_x1
        self.center_y = center_y1
        self.hasSmoothingCycle = has_smoothing_cycle1
        self.kernel_length = kernel_length1
        self.sigma = sigma1
        self.hasCleaning = has_cleaning1
        self.Na_max = na1
        self.Ns_max = ns1
        self.lambda_out = lambda_out1
        self.lambda_in = lambda_in1
        self.alpha = alpha1
        self.beta = beta1
        self.gamma = gamma1
        self.L_out = np.zeros([0, 0])
        self.L_in = np.zeros([0, 0])
        self.exterior_points = np.zeros([0, 0])
        self.interior_points = np.zeros([0, 0])

        self.Na = 0  # in the cycle 1
        self.Ns = 0  # in the cycle 2
        self.Ns_last = 0  # in the last cycle 2
        self.hasCycle1 = True

        # 3 variables of 3 stopping conditions
        self.hasListsChanges = True
        self.hasOscillation = False
        self.iterations = 0

        self.hasAlgoStopping = False  # to start the algorithm

        self.previous_lists_length = 999999
        self.lists_length = 0
        self.counter = 0

        if self.kernel_length % 2 == 0:
            self.kernel_length -= 1
        if self.kernel_length < 1:
            self.kernel_length = 1
        if self.sigma < 0.000000001:
            self.sigma = 0.000000001

        self.kernel_size = self.kernel_length ** 2
        self.kernel_radius = (self.kernel_length - 1) / 2
        self.gaussian_kernel = \
            np.zeros([self.kernel_length, self.kernel_length])

        self.hasOutwardEvolution = False
        self.hasInwardEvolution = False

        if img is not None:
            self.img_width = img.shape[1]
            self.img_height = img.shape[0]
            self.zero_img = np.zeros(img.shape[:2], dtype=np.uint8)
            self.img = img
            self.phi = np.zeros(img.shape[:2], dtype=int)
            self.redundant_matrix_l_in = np.zeros(self.phi.shape, dtype=bool)
            self.redundant_matrix_l_out = np.ones(self.phi.shape, dtype=bool)
            self.x, self.y = range(self.img_width), range(self.img_height)
            self.xv, self.yv = np.meshgrid(self.x, self.y)
            self.c_in_R = 0
            self.c_out_R = 0

            self.c_in_G = 0
            self.c_out_G = 0

            self.c_in_B = 0
            self.c_out_B = 0

            self.c_in_Y = 0
            self.c_out_Y = 0

            self.c_in_U = 0
            self.c_out_U = 0

            self.c_in_V = 0
            self.c_out_V = 0

            self.n_in = 0
            self.n_out = 0

            self.sum_out = np.array([0, 0, 0])
            self.sum_in = np.array([0, 0, 0])

        else:
            print "Invalid image"

    def initialize_params(self):
        self.make_gaussian_kernel()
        self.run_active_contour()
        self.sum_in = np.sum(self.img[self.phi <= 0], axis=0)
        self.n_in = len(self.img[self.phi <= 0])

        self.sum_out = np.sum(self.img[self.phi > 0], axis=0)
        self.n_out = len(self.img[self.phi > 0])

        self.calculate_means()

    def calculate_means(self):

        if self.n_out == 0:
            self.c_out_B, self.c_out_G, self.c_out_R = np.uint8([0, 0, 0])
        else:
            (self.c_out_B, self.c_out_G, self.c_out_R) = \
                np.int32(self.sum_out / self.n_out)
        if self.n_in == 0:
            self.c_in_B, self.c_in_G, self.c_in_R = np.uint8([0, 0, 0])
        else:
            (self.c_in_B, self.c_in_G, self.c_in_R) = \
                np.int32(self.sum_in / self.n_in)

        self.c_in_Y, self.c_in_U, self.c_in_V = \
            calculate_yuv(self.c_in_R, self.c_in_G, self.c_in_B)
        self.c_out_Y, self.c_out_U, self.c_out_V = \
            calculate_yuv(self.c_out_R, self.c_out_G, self.c_out_B)

    def compute_external_speed_Fd(self, i, j):
        res = np.int32(self.img[i, j, :])
        b, g, r = res[:, 0], res[:, 1], res[:, 2]
        y, u, v = calculate_yuv(r, g, b)
        return self.lambda_out * (self.alpha * (y - self.c_out_Y) ** 2
                                  + self.beta * (u - self.c_out_U) ** 2
                                  + self.gamma * (v - self.c_out_V) ** 2) \
               - self.lambda_in * (self.alpha * (y - self.c_in_Y) ** 2
                                   + self.beta * (u - self.c_in_U) ** 2
                                   + self.gamma * (v - self.c_in_V) ** 2)

    def do_specific_in_step(self, i, j):
        res = self.img[i, j, :]
        if len(i) == 1:
            b, g, r = res[0], res[1], res[2]
            self.sum_out -= np.array([b, g, r])
            self.n_out -= 1
            self.sum_in += np.array([b, g, r])
            self.n_in += 1
        else:
            b, g, r = res[:, 0], res[:, 1], res[:, 2]
            self.sum_out -= np.sum([b, g, r], axis=1)
            self.n_out -= len(i)
            self.sum_in += np.sum([b, g, r], axis=1)
            self.n_in += len(i)

    def do_specific_out_step(self, i, j):
        res = self.img[i, j, :]
        if len(i) == 1:
            b, g, r = res[0], res[1], res[2]
            self.sum_out += np.array([b, g, r])
            self.n_out += 1
            self.sum_in -= np.array([b, g, r])
            self.n_in -= 1
        else:
            b, g, r = res[:, 0], res[:, 1], res[:, 2]
            self.sum_out += np.sum([b, g, r], axis=1)
            self.n_out += len(i)
            self.sum_in -= np.sum([b, g, r], axis=1)
            self.n_in -= len(i)

    def run_active_contour(self):
        if self.hasEllipse:
            int_points, out_points = \
                get_pixel_points_ellipse_new(self.img_width,
                                             self.img_height, self.init_width,
                                             self.init_height)

            self.phi[int_points[:, 0], int_points[:, 1]] = -1
            self.phi[out_points[:, 0], out_points[:, 1]] = 1

            self.update_outer_points(out_points)
            self.update_inner_points(int_points)
        else:
            # for i in range(self.img_height):
            #     for j in range(self.img_width):
            #         if ((1 - self.init_height) * self.img_height / 2.0
            #                 + self.center_x * self.img_height) \
            #                 < i < (self.img_height - (
            #                     1 - self.init_height) * self.img_height / 2.0
            #                            + self.center_x
            #                 * self.img_height) and \
            #                                 ((
            #                                           1 - self.init_width) *
            #                                      self.img_width / 2.0
            #                                      + self.center_y *
            #                                         self.img_width) < j < (
            #                                 self.img_width
            #                                 - (
            #                                         1 - self.init_width) *
            #                                 self.img_width / 2.0
            #                             + self.center_y * self.img_width):
            #             self.phi[i, j] = -1
            #         # self.L_in.append([i, j])
            #         else:
            #             self.phi[i, j] = 1
            #             #                        self.L_out.append([i, j])
            #
            # self.L_out = np.argwhere(self.phi == 1)
            # self.L_in = np.argwhere(self.phi == -1)

            # pt1_x = int(self.img_width * (1 - self.init_width) / 2)
            # pt1_y = int(self.img_height * (1 - self.init_height) / 2)
            # pt2_x = int(self.img_width * (1 + self.init_width) / 2)
            # pt2_y = int(self.img_height * (1 + self.init_height) / 2)

            # Interior points
            interior_points = \
                get_pixel_points_rectangle(self.img_width,
                                           self.img_height, self.init_width,
                                           self.init_height, 1, is_filled=True)
            # zero_image_temp = self.zero_img.copy()
            # cv2.rectangle(zero_image_temp, (pt1_x + 1, pt1_y + 1),
            #               (pt2_x - 1, pt2_y - 1), 255, -1)
            # pixel_points = cv2.findNonZero(zero_image_temp)
            # interior_points = np.fliplr(np.reshape(pixel_points,
            #                                        [len(pixel_points), 2]))
            self.phi[interior_points[:, 0], interior_points[:, 1]] = -3

            # L_in points
            self.L_in = \
                get_pixel_points_rectangle(self.img_width,
                                           self.img_height, self.init_width,
                                           self.init_height, 0,
                                           is_filled=False)
            # zero_image_temp = self.zero_img.copy()
            # cv2.rectangle(zero_image_temp,
            #               (pt1_x, pt1_y), (pt2_x, pt2_y), 255, 1)
            # pixel_points = cv2.findNonZero(zero_image_temp)
            # self.L_in = np.fliplr(np.reshape(pixel_points,
            #                                  [len(pixel_points), 2]))
            self.phi[self.L_in[:, 0], self.L_in[:, 1]] = -1

            # L_out points
            self.L_out = \
                get_pixel_points_rectangle(self.img_width,
                                           self.img_height, self.init_width,
                                           self.init_height, -1,
                                           is_filled=False)
            # zero_image_temp = self.zero_img.copy()
            # cv2.rectangle(zero_image_temp,
            #               (pt1_x - 1, pt1_y - 1), (pt2_x + 1, pt2_y + 1),
            # 255,
            #               1)
            # pixel_points = cv2.findNonZero(zero_image_temp)
            # self.L_out = np.fliplr(np.reshape(pixel_points,
            #                                   [len(pixel_points), 2]))
            self.phi[self.L_out[:, 0], self.L_out[:, 1]] = 1

            # Exterior points
            exterior_points = \
                get_pixel_points_rectangle(self.img_width,
                                           self.img_height, self.init_width,
                                           self.init_height, -2,
                                           is_filled=True)
            # zero_image_temp = self.zero_img.copy()
            # cv2.rectangle(zero_image_temp, (pt1_x - 2, pt1_y - 2),
            #               (pt2_x + 2, pt2_y + 2), 255, -1)
            # pixel_points = cv2.findNonZero(zero_image_temp)
            # exterior_points = np.fliplr(np.reshape(pixel_points,
            #                                        [len(pixel_points), 2]))
            self.phi[exterior_points[:, 0], exterior_points[:, 1]] = 3

    def update_outer_points(self, out_points):
        self.redundant_matrix_l_out[1:-1, 1:-1] = np.logical_and(
            np.logical_and(self.phi[self.yv[0:-2, 1:-1],
                                    self.xv[1:-1, 1:-1]] > 0,
                           self.phi[self.yv[1:-1, 1:-1],
                                    self.xv[1:-1, 0:-2]] > 0),
            np.logical_and(self.phi[self.yv[2:, 1:-1],
                                    self.xv[1:-1, 1:-1]] > 0,
                           self.phi[self.yv[1:-1, 1:-1],
                                    self.xv[1:-1, 2:]] > 0))

        res_lout = self.redundant_matrix_l_out[
            out_points[:, 0], out_points[:, 1]]
        self.exterior_points = out_points[res_lout == 1]
        self.L_out = out_points[res_lout == 0]

        self.phi[self.L_out[:, 0], self.L_out[:, 1]] = 1
        self.phi[self.exterior_points[:, 0],
                 self.exterior_points[:, 1]] = 3

    def update_inner_points(self, int_points):
        self.redundant_matrix_l_in[1:-1, 1:-1] = np.logical_and(
            np.logical_and(self.phi[self.yv[0:-2, 1:-1],
                                    self.xv[1:-1, 1:-1]] <= 0,
                           self.phi[self.yv[1:-1, 1:-1],
                                    self.xv[1:-1, 0:-2]] <= 0),
            np.logical_and(self.phi[self.yv[2:, 1:-1],
                                    self.xv[1:-1, 1:-1]] <= 0,
                           self.phi[self.yv[1:-1, 1:-1],
                                    self.xv[1:-1, 2:]] <= 0))

        res_lin = self.redundant_matrix_l_in[
            int_points[:, 0], int_points[:, 1]]
        self.interior_points = int_points[res_lin == 1]
        self.L_in = int_points[res_lin == 0]

        self.phi[self.L_in[:, 0], self.L_in[:, 1]] = -1
        self.phi[self.interior_points[:, 0],
                 self.interior_points[:, 1]] = -3

    def eliminate_redundant_in_Lout(self):
        for point in self.L_out:
            if self.find_is_redundant_point_of_Lout(point):
                self.phi[point[0], point[1]] = 3
        self.L_out = find_common_array(self.L_out, np.argwhere(self.phi == 3))

    def eliminate_redundant_in_Lin(self):
        for point in self.L_in:
            if self.find_is_redundant_point_of_Lin(point):
                self.phi[point[0], point[1]] = -3
                #                self.L_in.remove(point)
        self.L_in = find_common_array(self.L_in, np.argwhere(self.phi == -3))

    def max_iterations(self):
        return 5 * min(self.img_height, self.img_width)

    def calculate_stopping_conditions1(self):
        if not self.hasInwardEvolution and not self.hasOutwardEvolution:
            self.hasListsChanges = False
        self.iterations += 1

        if not self.hasListsChanges \
                or self.iterations >= self.max_iterations():
            self.hasCycle1 = False
            if self.hasSmoothingCycle:
                self.hasAlgoStopping = True

    def calculate_stopping_conditions2(self):
        if abs(self.previous_lists_length - self.lists_length) / \
                max(self.lists_length, self.previous_lists_length) < 0.01:
            self.counter += 1
        else:
            if self.counter != 0:
                self.counter = 0
        if self.counter == 3:
            self.hasOscillation = True
        if self.hasOscillation or self.iterations >= self.max_iterations() \
                or not self.hasCycle1:
            self.hasAlgoStopping = True
        self.previous_lists_length = self.lists_length
        self.lists_length = 0

    def find_is_redundant_point_of_Lout(self, point):
        x, y = point

        if y - 1 >= 0:
            if self.phi[x, y - 1] <= 0:
                return False
        if x - 1 >= 0:
            if self.phi[x - 1, y] <= 0:
                return False
        if x + 1 < self.img_height:
            if self.phi[x + 1, y] <= 0:
                return False
        if y + 1 < self.img_width:
            if self.phi[x, y + 1] <= 0:
                return False
        return True

    def find_is_redundant_point_of_Lin(self, point):
        x, y = point

        if y - 1 >= 0:
            if self.phi[x, y - 1] >= 0:
                return False
        if x - 1 >= 0:
            if self.phi[x - 1, y] >= 0:
                return False
        if x + 1 < self.img_height:
            if self.phi[x + 1, y] >= 0:
                return False
        if y + 1 < self.img_width:
            if self.phi[x, y + 1] >= 0:
                return False
        return True

    def compute_internal_speed_Fint(self, x, y):
        f_int = 0
        # x, y = point
        # if not in the border of the image, no neighbor test
        if self.kernel_radius - 1 < x < self.img_height - self.kernel_radius \
                and self.kernel_radius - 1 < y < self.img_width - \
                        self.kernel_radius:
            for dy in range(-self.kernel_radius, self.kernel_radius + 1):
                for dx in range(-self.kernel_radius, self.kernel_radius + 1):
                    f_int += self.gaussian_kernel[self.kernel_radius + dx,
                                                  self.kernel_radius + dy] * \
                             signum_function(self.phi[x + dx, y + dy])
        else:
            for dy in range(-self.kernel_radius, self.kernel_radius + 1):
                for dx in range(-self.kernel_radius, self.kernel_radius + 1):
                    if 0 <= x + dx < self.img_height \
                            and 0 <= y + dy < self.img_width:
                        f_int += self.gaussian_kernel[self.kernel_radius + dx,
                                                      self.kernel_radius
                                                      + dy] \
                                 * signum_function(
                            self.phi[x + dx, y + dy])
                    else:
                        f_int += self.gaussian_kernel[self.kernel_radius + dx,
                                                      self.kernel_radius
                                                      + dy] \
                                 * signum_function(
                            self.phi[x, y])
        return f_int

    def make_gaussian_kernel(self):
        for i in range(self.kernel_length):
            for j in range(self.kernel_length):
                self.gaussian_kernel[i, j] = \
                    round(abs(0.5 + 100000.0 *
                              cmath.exp(-((i - self.kernel_radius) ** 2
                                          + (j - self.kernel_radius) ** 2) /
                                        (2 * self.sigma ** 2))), 2)

    def do_specific_beginning_step(self):
        self.calculate_means()

    def do_one_iteration_in_cycle1(self):
        self.do_specific_beginning_step()

        self.hasOutwardEvolution = False
        # for point in self.L_out:
        #     if self.compute_external_speed_Fd(point[0], point[1]) > 0:
        #         self.do_specific_in_step(point[0], point[1])
        #         self.switch_in(point)
        #         if not self.hasOutwardEvolution:
        #             self.hasOutwardEvolution = True

        speeds = self.compute_external_speed_Fd(self.L_out[:, 0],
                                                self.L_out[:, 1])
        if speeds.any() > 0:
            in_points = self.L_out[speeds > 0]
            # Update variables to calculate Cin and Cout means
            self.do_specific_in_step(in_points[:, 0], in_points[:, 1])
            self.switch_in_vector(in_points)  # outward local movement
            if not self.hasOutwardEvolution:
                self.hasOutwardEvolution = True

        if self.hasOutwardEvolution and self.hasCleaning:
            # self.eliminate_redundant_in_Lin()
            self.update_inner_points(np.vstack((self.interior_points,
                                                self.L_in)))

        self.hasInwardEvolution = False
        # for point in self.L_in:
        #     if self.compute_external_speed_Fd(point[0], point[1]) < 0:
        #         self.do_specific_out_step(point[0], point[1])
        #         self.switch_out(point)
        #         if not self.hasInwardEvolution:
        #             self.hasInwardEvolution = True
        speeds = self.compute_external_speed_Fd(self.L_in[:, 0],
                                                self.L_in[:, 1])
        if speeds.any() < 0:
            out_points = self.L_in[speeds < 0]
            self.do_specific_out_step(out_points[:, 0], out_points[:, 1])
            self.switch_out_vector(out_points)
            if not self.hasInwardEvolution:
                self.hasInwardEvolution = True
        if self.hasInwardEvolution and self.hasCleaning:
            # self.eliminate_redundant_in_Lout()
            self.update_outer_points(np.vstack((self.exterior_points,
                                                self.L_out)))
        self.calculate_stopping_conditions1()

    def do_one_iteration_in_cycle2(self):
        if self.Ns == self.Ns_max - 1:
            self.lists_length += 1
        for point in self.L_out:
            if self.compute_internal_speed_Fint(point[0], point[1]) < 0:
                self.do_specific_in_step(point[0], point[1])
        # speeds = self.compute_internal_speed_Fint(self.L_out[:, 0],
        #                                         self.L_out[:, 1])
        # if speeds.any() < 0:
        #     in_points = self.L_out[speeds < 0]
        #     # Update variables to calculate Cin and Cout means
        #     self.do_specific_in_step(in_points[:, 0], in_points[:, 1])
        # if self.hasCleaning:
        #     self.update_outer_points(np.vstack((self.exterior_points,
        #                                         self.L_out)))

        for point in self.L_in:
            if self.Ns == self.Ns_max - 1:
                self.lists_length += 1
            if self.compute_internal_speed_Fint(point[0], point[1]) > 0:
                self.do_specific_out_step(point[0], point[1])
        if self.hasCleaning:
            self.eliminate_redundant_in_Lout()

        # speeds = self.compute_internal_speed_Fint(self.L_in[:, 0],
        #                                         self.L_in[:, 1])
        # if speeds.any() > 0:
        #     out_points = self.L_in[speeds > 0]
        #     self.do_specific_out_step(out_points[:, 0], out_points[:, 1])
        # if self.hasCleaning:
        #     # self.eliminate_redundant_in_Lout()
        #     self.update_outer_points(np.vstack((self.exterior_points,
        #                                         self.L_out)))
        self.iterations += 1

    def evolve_one_iteration(self):
        if not self.hasAlgoStopping:
            if self.hasCycle1:
                if self.Na < self.Na_max - 1:
                    self.do_one_iteration_in_cycle1()
                    self.Na += 1
                    return
                if self.Na == self.Na_max - 1:
                    self.do_one_iteration_in_cycle1()
                    if self.hasSmoothingCycle:
                        self.Na += 1
                    else:
                        self.Na = 0
                    return
            if self.hasSmoothingCycle:
                if self.Ns < self.Ns_max - 1:
                    self.do_one_iteration_in_cycle2()
                    self.Ns += 1
                    return
                if self.Ns == self.Ns_max - 1:
                    self.do_one_iteration_in_cycle2()
                    self.calculate_stopping_conditions2()

                    self.Na = 0
                    self.Ns = 0
                    return
        return

    def evolve_n_iterations(self, n):
        n_total = self.iterations + n
        while not self.hasAlgoStopping and self.iterations < n_total:
            while self.Na < self.Na_max and self.hasCycle1 \
                    and self.iterations < n_total:
                self.do_one_iteration_in_cycle1()
                self.Na += 1
            self.Na = 0

            if self.hasSmoothingCycle:
                while self.Ns < self.Ns_max and self.iterations < n_total:
                    self.do_one_iteration_in_cycle2()
                    self.Ns += 1
                self.Ns = 0
                self.calculate_stopping_conditions2()

    # def switch_in(self, point):
    #     x, y = point
    #     self.L_out.remove(point)
    #     # self.L_in.append(point)
    #     np.vstack((self.L_in, point))
    #     self.phi[x, y] = -1
    #     if y - 1 >= 0:
    #         self.add_Rout_neighbor_to_Lout(x, y - 1)
    #     if x - 1 >= 0:
    #         self.add_Rout_neighbor_to_Lout(x - 1, y)
    #     if x + 1 < self.img_height:
    #         self.add_Rout_neighbor_to_Lout(x + 1, y)
    #     if y + 1 < self.img_width:
    #         self.add_Rout_neighbor_to_Lout(x, y + 1)

    def switch_in_vector(self, points):
        x, y = points[:, 0], points[:, 1]
        self.L_out = find_common_array(self.L_out, points)
        self.L_in = np.vstack((self.L_in, points))
        self.phi[x, y] = -1
        self.update_outer_points(np.vstack((self.L_out,
                                            self.exterior_points)))

    def switch_out_vector(self, points):
        x, y = points[:, 0], points[:, 1]
        self.L_in = find_common_array(self.L_in, points)
        self.L_out = np.vstack((self.L_out, points))
        self.phi[x, y] = 1
        self.update_inner_points(np.vstack((self.L_in,
                                            self.interior_points)))

        # def add_Rout_neighbor_to_Lout(self, x, y):
        #     if self.phi[x, y] == 3:
        #         self.phi[x, y] = 1
        #         self.L_out.append([x, y])

        # def switch_out(self, point):
        #     x, y = point
        #     self.L_in.remove(point)
        #     self.L_out.append(point)
        #     self.phi[x, y] = 1
        #     if y - 1 >= 0:
        #         self.add_Rout_neighbor_to_Lin(x, y - 1)
        #     if x - 1 >= 0:
        #         self.add_Rout_neighbor_to_Lin(x - 1, y)
        #     if x + 1 < self.img_height:
        #         self.add_Rout_neighbor_to_Lin(x + 1, y)
        #     if y + 1 < self.img_width:
        #         self.add_Rout_neighbor_to_Lin(x, y + 1)

        # def add_Rout_neighbor_to_Lin(self, x, y):
        #     if self.phi[x, y] == -3:
        #         self.phi[x, y] = -1
        #         self.L_in.append([x, y])
