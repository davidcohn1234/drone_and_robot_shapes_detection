import time

import cv2
import numpy as np
import imutils
from enum import Enum
from pytesseract import pytesseract
import nltk
from imutils import object_detection
import math
#from fuzzywuzzy import process
import random as rng


class ShapeDetector:
    class ImageDateType(Enum):
        NO_TYPE = 0
        QR_COLORS_ONLY = 1
        QR_SHAPES_ONLY = 2
        SHAPES_AND_COLORS = 3
        WORDS_SHAPES_NAMES = 4
        WORDS_COLORS_NAMES = 5

    def __init__(self):
        self.color_name_to_rgb = {'name': 'brown', 'rgb_color': (19, 69, 139)}
        self.colors_name_to_rgb = [
            {'name': 'brown', 'rgb_color': (19, 69, 139)},
            {'name': 'yellow', 'rgb_color': (0, 255, 255)},
            {'name': 'orange', 'rgb_color': (0, 165, 255)},
            {'name': 'red', 'rgb_color': (0, 0, 255)},
            {'name': 'purple', 'rgb_color': (153, 51, 102)},
            {'name': 'green', 'rgb_color': (0, 255, 0)},
            {'name': 'blue', 'rgb_color': (180, 10, 0)},
            {'name': 'pink', 'rgb_color': (255, 0, 255)}]

        self.brown_color_info_tello_without_stick = {"name": 'brown', "lower": (0, 98, 81), "upper": (11, 190, 135),
                                                     "rgb_color": (19, 69, 139)} # back:0 98 81 11  190 135,rest: 0 95 71 13 194 126
        self.yellow_color_info_tello_without_stick = {"name": 'yellow', "lower": (19,139, 128), "upper": (37, 255, 214),
                                                      "rgb_color": (0, 255, 255)} #back:19 144  159 040 255 214,rest: 19 139 128 37 255 192
        self.orange_color_info_tello_without_stick = {"name": 'orange', "lower": (8, 221, 132), "upper": (29, 255, 202),
                                                      "rgb_color": (0, 165, 255)} # back:8 221 132 29 255 202,rest:8 221 132 20 255 182
        self.red_color_info_tello_without_stick = {"name": 'red', "lower": (0, 210, 106), "upper": (4, 255, 182),
                                                   "rgb_color": (0, 0, 255)} #back: 0 192 106 004 255 182,rest: 0 210 106 4 255 167
        self.purple_color_info_tello_without_stick = {"name": 'purple', "lower": (119, 62, 52),
                                                      "upper": (140, 189, 131), "rgb_color": (200, 0, 119)} # back:119 62 52 140 189 131,rest: 119 62 52 140 189 116
        self.green_color_info_tello_without_stick = {"name": 'green', "lower": (52, 46, 55), "upper": (90, 233, 137),
                                                     "rgb_color": (0, 255, 0)} #back:52, 46, 55, 90, 233, 137,rest:52 46 55 90 233 115
        self.blue_color_info_tello_without_stick = {"name": 'blue', "lower": (90, 95, 84), "upper": (117, 218, 177),
                                                    "rgb_color": (180, 10, 0)} #back:90, 95, 84, 117, 218, 177,rest:90 95 84 117 218
        self.pink_color_info_tello_without_stick = {"name": 'pink', "lower": (164, 211, 103),
                                                    "upper": (184, 255, 194), "rgb_color": (255, 0, 255)} # back:155 211 103 184 255 194,rest:164 211 103 184255 174
        self.unknown_color_info = {"name": 'unknown', "lower": (0, 0, 0), "upper": (0, 0, 0), "rgb_color": (0, 0, 0)}

        self.colors_ranges_info_tello_without_stick = [self.brown_color_info_tello_without_stick,
                                                       self.yellow_color_info_tello_without_stick,
                                                       self.orange_color_info_tello_without_stick,
                                                       self.red_color_info_tello_without_stick,
                                                       self.purple_color_info_tello_without_stick,
                                                       self.green_color_info_tello_without_stick,
                                                       self.blue_color_info_tello_without_stick,
                                                       self.pink_color_info_tello_without_stick]

        self.max_screen_width = 750
        self.min_screen_width = 200

        self.max_screen_height = 450
        self.min_screen_height = 300

        # self.max_square_size = 90
        # self.min_square_size = 50

        self.max_square_size = 110
        self.min_square_size = 30
        self.max_width_height_ratio = 1.2

        self.shapes_types = ['circle', 'octagon', 'pentagon', 'rectangle', 'square', 'rhombus', 'star', 'triangle']
        self.image_data = {
            'image_data_type': self.ImageDateType.NO_TYPE,
            'shapes_data': None,
            'screen_data': {'screen_color_data': {'name': None, 'rgb_color': None},
                            'screen_box': None,
                            'screen_box_dict': None,
                            'small_box_on_empty_part_of_screen': None,
                            'small_box_on_empty_part_of_screen_dict': None}
        }
        self.template_gray_images = self.create_template_gray_images(self.shapes_types)
        self.detect = cv2.QRCodeDetector()

    def combine_shapes_boxes_into_a_single_list(self, shapes_boxes):
        boxes = []
        for single_shape_box in shapes_boxes:
            boxes.append(single_shape_box)
        return boxes

    def draw_boxes_and_write_their_sizes_on_image(self, rgb_image, boxes, box_color, box_thickness):

        # x, y, w, h = cv2.boundingRect(single_contour)
        # if self.max_screen_width > w > self.min_screen_width and \
        #         self.max_screen_height > h > self.min_screen_height:
        #     x1 = x
        #     x2 = x1 + w
        #     y1 = y
        #     y2 = y1 + h
        #     boxes.append((x1, y1, x2, y2))

        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            width = x2 - x1
            height = y2 - y1
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.3
            text_color = (0, 0, 0)
            text_thickness = 1

            local_max_screen_width = 300
            local_min_screen_width = 100

            local_max_screen_height = 450
            local_min_screen_height = 100

            local_max_square_size = 85
            local_min_square_size = 40

            # if self.max_screen_width > width > self.min_screen_width and \
            #         self.max_screen_height > height > self.min_screen_height:
            # if local_max_square_size > width > local_min_square_size and \
            #         local_max_square_size > height > local_min_square_size:
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), box_color, box_thickness)
            cv2.putText(rgb_image, f'({width},{height})', (x1, y1), font, font_scale, text_color, text_thickness,
                        cv2.LINE_AA)
        return rgb_image

    def draw_boxes_on_image(self, rgb_image, boxes, box_color, box_thickness):
        for box in boxes:
            if len(box) == 0:
                return rgb_image
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), box_color, box_thickness)

            width = x2 - x1
            height = y2 - y1
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.4
            text_color = (0, 0, 0)
            text_thickness = 1
            cv2.putText(rgb_image, f'({width},{height})', (x1, y1+10), font, font_scale, text_color,
                        text_thickness, cv2.LINE_AA)
        return rgb_image

    def draw_shapes_boxes_on_image(self, rgb_image, shapes_boxes):
        shape_box_color = (255, 0, 0)
        shape_box_thickness = 4
        boxes = self.combine_shapes_boxes_into_a_single_list(shapes_boxes)
        rgb_image = self.draw_boxes_on_image(rgb_image, boxes, shape_box_color, shape_box_thickness)
        return rgb_image

    def draw_small_box_on_empty_part_of_image(self, rgb_image, small_box_on_empty_part_of_screen):
        if small_box_on_empty_part_of_screen is None:
            return rgb_image
        empty_screen_box_color = (255, 0, 255)
        empty_screen_box_thickness = 4
        boxes = [small_box_on_empty_part_of_screen]
        rgb_image = self.draw_boxes_on_image(rgb_image, boxes, empty_screen_box_color, empty_screen_box_thickness)
        return rgb_image

    def draw_screen_box_on_image(self, rgb_image, screen_box):
        if screen_box is None or len(screen_box) == 0:
            return rgb_image
        screen_color = (0, 0, 255)
        boxes = [screen_box]
        screen_box_thickness = 4
        rgb_image = self.draw_boxes_on_image(rgb_image, boxes, screen_color, screen_box_thickness)
        return rgb_image

    def create_template_gray_images(self, shapes_types):
        template_images = {}
        for shape_index, shape_name in enumerate(shapes_types):
            template_full_path = '../template_shapes/' + shape_name + '.jpg'
            template_rgb_image = cv2.imread(template_full_path)
            template_gray_image = cv2.cvtColor(template_rgb_image, cv2.COLOR_BGR2GRAY)
            # template_gray_image_without_border = self.get_image_without_border(template_gray_image)
            template_images[shape_name] = template_gray_image
        return template_images

    def get_left_most_vertex_of_left_most_shape(self, shapes_data):
        num_of_shapes_in_screen = len(shapes_data)
        left_x_of_shapes = np.zeros((num_of_shapes_in_screen))
        for shape_index, single_shape_data in enumerate(shapes_data):
            left_x_of_current_shape = single_shape_data['shape_box'][0]
            left_x_of_shapes[shape_index] = left_x_of_current_shape
        min_index = np.argmin(left_x_of_shapes)
        left_most_shape_data = shapes_data[min_index]
        left_most_shape_x = left_most_shape_data['shape_box'][0]
        bottom_most_shape_y = left_most_shape_data['shape_box'][3]
        left_most_shape_name = left_most_shape_data['shape_name']
        return left_most_shape_name, left_most_shape_x, bottom_most_shape_y

    def get_partial_box(self, box, ratio):
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        width = x2 - x1
        height = y2 - y1

        start_x = int(x1 + ratio * width)
        start_y = int(y1 + ratio * height)

        end_x = int(x1 + (1 - ratio) * width)
        end_y = int(y1 + (1 - ratio) * height)

        partial_box = np.array((start_x, start_y, end_x, end_y))
        return partial_box

    def get_small_box_on_empty_part_of_screen(self, shapes_data, screen_box):
        if len(shapes_data) == 0:
            # screen is empty or all the shapes are not recognized
            # we'll just hope that the screen is empty and return a box in the middle of the screen
            empty_screen_ratio = 0.25
            partial_box_of_empty_screen = self.get_partial_box(screen_box, empty_screen_ratio)
            return partial_box_of_empty_screen

        screen_left_x = int(screen_box[0])
        screen_top_y = int(screen_box[1])
        screen_right_x = int(screen_box[2])
        screen_bottom_y = int(screen_box[3])
        (left_most_shape_name, left_most_x_of_left_most_shape,
         bottom_most_y_of_left_most_shape) = self.get_left_most_vertex_of_left_most_shape(shapes_data)
        width_start_screen_to_left_most_shape = left_most_x_of_left_most_shape - screen_left_x
        height_top_screen_to_left_most_shape = bottom_most_y_of_left_most_shape - screen_top_y

        ratio_x = 0.25
        ratio_y = 0.4
        start_x = int(screen_left_x + ratio_x * width_start_screen_to_left_most_shape)
        start_y = int(screen_top_y + ratio_y * height_top_screen_to_left_most_shape)

        end_x = int(screen_left_x + (1 - ratio_x) * width_start_screen_to_left_most_shape)
        end_y = int(screen_top_y + (1 - ratio_y) * height_top_screen_to_left_most_shape)

        # small_portion_of_empty_screen = rgb_image[start_y: end_y, start_x: end_x, :]
        small_portion_box_of_empty_screen = np.array((start_x, start_y, end_x, end_y))
        return small_portion_box_of_empty_screen

    def get_screen_color_data(self, rgb_image, small_box_on_empty_part_of_screen):
        screen_color_data = {'name': None, 'rgb_color': None}
        rgb_sub_image = self.get_sub_image_by_box(rgb_image, small_box_on_empty_part_of_screen)
        if rgb_sub_image is None:
            david = 5
        if len(rgb_sub_image) == 0:
            david = 7

        mean_pixel = np.mean(rgb_sub_image, axis=(0, 1))
        if np.any(np.isnan(mean_pixel)):
            david = 9
        threshold_red = 65
        threshold_green = 50
        threshold_blue = 65
        if mean_pixel[0] < threshold_blue and mean_pixel[1] > threshold_green and mean_pixel[2] < threshold_red:
            screen_color_data['name'] = 'green'
            screen_color_data['rgb_color'] = (0, 255, 0)
        elif mean_pixel[0] < threshold_blue and mean_pixel[1] < threshold_green and mean_pixel[2] > threshold_red:
            screen_color_data['name'] = 'red'
            screen_color_data['rgb_color'] = (0, 0, 255)
        elif mean_pixel[0] > threshold_blue and mean_pixel[1] > threshold_green and mean_pixel[2] > threshold_red:
            screen_color_data['name'] = 'white'
            screen_color_data['rgb_color'] = (255, 255, 255)
        else:
            screen_color_data['name'] = 'black'
            screen_color_data['rgb_color'] = (0, 0, 0)
        return screen_color_data

    def get_color_data_in_box(self, rgb_image, box):
        color_data = {'name': None, 'rgb_color': None}
        rgb_sub_image = self.get_sub_image_by_box(rgb_image, box)
        # cv2.imshow('rgb_sub_image', rgb_sub_image)
        # cv2.waitKey(1)
        color_code = cv2.COLOR_BGR2HSV
        colors_ranges_info = self.colors_ranges_info_tello_without_stick
        num_of_colors = len(colors_ranges_info)
        colors_num_of_pixels = np.zeros([num_of_colors])
        for index, single_color_range_info in enumerate(colors_ranges_info):
            color_name = single_color_range_info["name"]
            lower = single_color_range_info["lower"]
            upper = single_color_range_info["upper"]
            single_color_num_of_pixels = self.count_pixels_in_color_range(lower, upper, rgb_sub_image, color_name,
                                                                          color_code)
            colors_num_of_pixels[index] = single_color_num_of_pixels
        max_index = colors_num_of_pixels.argmax()
        shape_color_data = colors_ranges_info[max_index]
        if np.sum(colors_num_of_pixels) == 0:
            return color_data
        color_data['name'] = shape_color_data['name']
        color_data['rgb_color'] = shape_color_data['rgb_color']
        return color_data

    def convert_box_to_dict(self, box):
        if len(box) == 0:
            return {}
        dict_box = {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
        return dict_box

    def get_color_rgb_by_its_name(self, color_name):
        for val in self.colors_name_to_rgb:
            if val['name'] == color_name:
                return val['rgb_color']
        return None



    def get_colors_only_frame_data(self, rgb_image, colors_names, screen_data, image_data_type):
        self.image_data['screen_data'] = screen_data
        self.image_data['image_data_type'] = image_data_type
        shapes_data = []
        for color_name in colors_names:
            color_rgb = self.get_color_rgb_by_its_name(color_name)
            shape_color_data = {'name': color_name, 'rgb_color': color_rgb}
            single_shape_data = {'shape_name': None,
                                 'shape_color_data': shape_color_data,
                                 'shape_box': None,
                                 'shape_box_dict': None,
                                 'small_shape_box_for_color_detection': None}
            shapes_data.append(single_shape_data)
        self.image_data['shapes_data'] = shapes_data
        return self.image_data

    def get_shapes_only_frame_data(self, rgb_image, shapes_names, screen_data, image_data_type):
        self.image_data['screen_data'] = screen_data
        self.image_data['image_data_type'] = image_data_type
        shapes_data = []
        for shape_name in shapes_names:
            shape_color_data = {'name': None, 'rgb_color': None}
            single_shape_data = {'shape_name': shape_name,
                                 'shape_color_data': shape_color_data,
                                 'shape_box': None,
                                 'shape_box_dict': None,
                                 'small_shape_box_for_color_detection': None}
            shapes_data.append(single_shape_data)
        self.image_data['shapes_data'] = shapes_data
        return self.image_data

    def filter_words_by_list(self, list_text_in_image, possible_words):
        num_of_allowed_mistaken_letters = 0
        num_of_possible_words = len(possible_words)
        filtered_list_text_in_image = []
        for text_in_image in list_text_in_image:
            text_in_image = text_in_image.lower()
            list_dists = np.zeros(num_of_possible_words)
            for index, possible_word in enumerate(possible_words):
                current_dist = nltk.edit_distance(text_in_image, possible_word)
                list_dists[index] = current_dist
            min_dist = np.min(list_dists)
            min_index = np.argmin(list_dists)
            min_possible_word = possible_words[min_index]
            if min_dist <= num_of_allowed_mistaken_letters:
                print(min_dist)
                filtered_list_text_in_image.append(text_in_image)
        return filtered_list_text_in_image

    def read_words(self, rgb_image):
        rgb_image_with_words_data = rgb_image.copy()
        # opening an image from the source path
        num_of_allowed_mistaken_letters = 1
        screen_point_for_tello = None
        words_type = None
        screen_data = dict()
        screen_data['screen_box'] = None
        screen_data['screen_color_data'] = None
        screen_data['small_box_on_empty_part_of_screen'] = None
        screen_data['screen_point_for_tello'] = None
        screen_data['center_of_screen_box'] = None



        # image = cv2.imread(filename)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("grey.png", gray)

        binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        #binary_image_white_background = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2)
        #binary_image = cv2.bitwise_not(binary_image_white_background)
        # cv2.imwrite("binary_image.png", binary_image)
        text = pytesseract.image_to_string(binary_image, lang='eng', config='--psm 6')
        text = text.lower()
        list_text_in_image = text.split()

        data = pytesseract.image_to_data(binary_image, output_type='dict')
        num_of_boxes = len(data['level'])
        x_vals = []
        y_vals = []
        for i in range(num_of_boxes):
            #confidence = data['conf'][i] > 95
            text_in_screen = data['text'][i]
            text_in_screen_no_spaces = text_in_screen.replace(" ", "")
            text_in_screen_no_spaces_length = len(text_in_screen_no_spaces)
            confidence = text_in_screen_no_spaces_length >= 3 #since the shortest word is red (for colors) and star (for shapes)
            if confidence:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                x_vals.append(x)
                y_vals.append(y)
                cv2.rectangle(rgb_image_with_words_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if len(x_vals) > 0:
            np_x_vals = np.array(x_vals)
            np_y_vals = np.array(y_vals)
            x_min_words = np.min(np_x_vals)
            x_max_words = np.max(np_x_vals)
            y_min_words = np.min(np_y_vals)
            y_max_words = np.max(np_y_vals)

            contours = self.get_image_contours(rgb_image)
            screen_box = self.get_screen_box_screen_content(contours, x_min_words, y_min_words, x_max_words,
                                                            y_max_words)
            screen_center = self.get_screen_center(screen_box)


            delta_y = min(y_min_words, 40)
            start_x = x_min_words
            start_y = int(y_min_words - 0.5 * delta_y)
            end_x = x_max_words
            end_y = int(y_min_words - 0.25 * delta_y)
            small_box_on_empty_part_of_screen = np.array((start_x, start_y, end_x, end_y))
            screen_color_data = self.get_screen_color_data(rgb_image, small_box_on_empty_part_of_screen)
            x_screen_point_for_tello = int(0.5 * (start_x + end_x))
            y_screen_point_for_tello = int(0.5 * (start_y + end_y))
            screen_point_for_tello = np.array((x_screen_point_for_tello, y_screen_point_for_tello))


            screen_data['screen_box'] = screen_box
            screen_data['screen_color_data'] = screen_color_data
            screen_data['small_box_on_empty_part_of_screen'] = small_box_on_empty_part_of_screen
            screen_data['screen_point_for_tello'] = screen_point_for_tello
            screen_data['center_of_screen_box'] = screen_center

        # rgb_image = self.draw_circle_on_image(rgb_image, screen_point_for_tello, circle_color=(255, 0, 255), radius=10)
        # cv2.imshow('rgb_image', rgb_image)
        # cv2.waitKey(0)

        list_of_colors = ['blue', 'yellow', 'purple', 'brown', 'orange', 'green', 'pink', 'red', 'circle']
        list_of_shapes = ['triangle', 'star', 'octagon', 'square', 'rhombus', 'rectangle', 'pentagon']
        list_of_possible_words = list_of_colors + list_of_shapes

        filtered_list_text_in_image = []

        for text_in_image in list_text_in_image:
            name = []
            grade = []
            for possible_word in list_of_possible_words:
                dist_words = nltk.edit_distance(text_in_image, possible_word)
                if dist_words <= num_of_allowed_mistaken_letters:
                    name.append(possible_word)
                    grade.append(dist_words)
            if grade:
                result = np.where(grade == np.amin(grade))
                # print(name[int(result[0])])
                filtered_list_text_in_image.append(name[int(result[0])])
        if len(filtered_list_text_in_image) > 0:
            for name in filtered_list_text_in_image:
                if name in list_of_colors:
                    words_type = 4
                elif name in list_of_shapes:
                    words_type = 5
                else:
                    words_type = None
        return filtered_list_text_in_image, words_type, screen_data, rgb_image_with_words_data
        # print(filtered_list_text_in_image)

    #     filtered_list_text_in_image1 = []
    #     for text_in_image in list_text_in_image:
    #         Ratio = process.extract(text_in_image.lower(), possible_words)
    #         # print(Ratio)
    #         # You can also select the string with the highest matching percentage
    #         highest = process.extractOne(text_in_image.lower(), possible_words)
    #         # print(highest)
    #         if highest[1] > 80:
    #             filtered_list_text_in_image1.append(highest[0])
    #     # print(filtered_list_text_in_image1)
    #     value = []
    #     for text_in_image in filtered_list_text_in_image:
    #         Ratio = process.extract(text_in_image.lower(), filtered_list_text_in_image1)
    #         # print(Ratio)
    #         # You can also select the string with the highest matching percentage
    #         highest = process.extractOne(text_in_image.lower(), filtered_list_text_in_image1)
    #         # print(highest)
    #         if highest[1] > 95:
    #             value.append(highest[0])
    #
    # except:
    #     value = []
    # if len(value) > 0:
    #     for name in value:
    #         if name in list_of_colors:
    #             words_type = 4
    #         elif name in list_of_shapes:
    #             words_type = 5
    #         else:
    #             words_type = None


    def get_image_data_from_frame(self, rgb_image):
        qr_data, qr_points, qr_data_type, screen_data = self.get_image_qr_data(rgb_image)
        if qr_data_type == 'qr_colors_only':
            self.image_data = self.get_colors_only_frame_data(rgb_image, qr_data, screen_data, self.ImageDateType.QR_COLORS_ONLY)
        elif qr_data_type == 'qr_shapes_only':
            self.image_data = self.get_shapes_only_frame_data(rgb_image, qr_data, screen_data, self.ImageDateType.QR_SHAPES_ONLY)
        else:
            words, words_type, screen_data, rgb_image_with_words_data = self.read_words(rgb_image)
            if words_type == 5:
                self.image_data = self.get_shapes_only_frame_data(rgb_image, words, screen_data, self.ImageDateType.WORDS_SHAPES_NAMES)
            elif words_type == 4:
                self.image_data = self.get_colors_only_frame_data(rgb_image, words, screen_data, self.ImageDateType.WORDS_COLORS_NAMES)
            else:
                self.image_data = self.get_shapes_and_colors_data_from_frame(rgb_image)
                num_of_detected_shapes_in_image = len(self.image_data['shapes_data'])
                if num_of_detected_shapes_in_image > 0:
                    self.image_data['image_data_type'] = self.ImageDateType.SHAPES_AND_COLORS
                else:
                    self.image_data['image_data_type'] = self.ImageDateType.NO_TYPE
                dd = self.image_data
        return self.image_data

    def get_screen_center(self, screen_box):
        if screen_box is None:
            return None
        x1 = screen_box[0]
        y1 = screen_box[1]
        x2 = screen_box[2]
        y2 = screen_box[3]
        x_screen_center = int(0.5 * (x1 + x2))
        y_screen_center = int(0.5 * (y1 + y2))
        screen_center = np.array((x_screen_center, y_screen_center))
        return screen_center

    def get_screen_box_screen_content(self, contours, x_min_shapes, y_min_shapes, x_max_shapes, y_max_shapes):
        screen_candidates = []
        for single_contour in contours:
            x, y, w, h = cv2.boundingRect(single_contour)
            # if self.max_square_size > w > self.min_square_size and \
            #         self.max_square_size > h > self.min_square_size:
            x1 = x
            x2 = x1 + w
            y1 = y
            y2 = y1 + h
            ratio_w_h = w / h
            if x1 < x_min_shapes and y1 < y_min_shapes and x2 > x_max_shapes and y2 > y_max_shapes and ratio_w_h <=1.8 and ratio_w_h >= 1.4:
                screen_box = (x1, y1, x2, y2)
                screen_area = cv2.contourArea(single_contour)
                single_screen_candidate = dict()
                single_screen_candidate['screen_box'] = screen_box
                single_screen_candidate['screen_area'] = screen_area
                screen_candidates.append(single_screen_candidate)
        screen_candidates_areas = np.array(
            [single_screen_candidate['screen_area'] for single_screen_candidate in screen_candidates])
        if len(screen_candidates_areas) == 0:
            return None
        index_min = np.argmin(screen_candidates_areas)
        screen_box = screen_candidates[index_min]['screen_box']
        return screen_box

    def get_screen_data(self, rgb_image, contours, shapes_data):
        screen_data = {'screen_box': None, 'screen_color_data': None, 'small_box_on_empty_part_of_screen': None, 'screen_point_for_tello': None, 'center_of_screen_box': None}
        shapes_boxes = [single_shape_data['shape_box'] for single_shape_data in shapes_data]
        if len(shapes_boxes) == 0:
            return screen_data

        x_min_values = [single_shape_box[0] for single_shape_box in shapes_boxes]
        y_min_values = [single_shape_box[1] for single_shape_box in shapes_boxes]
        x_max_values = [single_shape_box[2] for single_shape_box in shapes_boxes]
        y_max_values = [single_shape_box[3] for single_shape_box in shapes_boxes]

        x_min_shapes = min(x_min_values)
        y_min_shapes = min(y_min_values)
        x_max_shapes = max(x_max_values)
        y_max_shapes = max(y_max_values)

        screen_box = self.get_screen_box_screen_content(contours, x_min_shapes, y_min_shapes, x_max_shapes, y_max_shapes)
        screen_center = self.get_screen_center(screen_box)



        delta_y = min(y_min_shapes, 40)
        start_x = x_min_shapes
        start_y = int(y_min_shapes - 0.5 * delta_y)
        end_x = x_max_shapes
        end_y = int(y_min_shapes - 0.25 * delta_y)
        if end_x < start_x:
            david = 6
        small_box_on_empty_part_of_screen = np.array((start_x, start_y, end_x, end_y))
        screen_color_data = self.get_screen_color_data(rgb_image, small_box_on_empty_part_of_screen)

        x_screen_point_for_tello = int(0.5 * (start_x + end_x))
        y_screen_point_for_tello = int(0.5 * (start_y + end_y))
        screen_point_for_tello = np.array((x_screen_point_for_tello, y_screen_point_for_tello))

        screen_data['screen_box'] = screen_box
        screen_data['screen_color_data'] = screen_color_data
        screen_data['small_box_on_empty_part_of_screen'] = small_box_on_empty_part_of_screen
        screen_data['screen_point_for_tello'] = screen_point_for_tello
        screen_data['center_of_screen_box'] = screen_center

        # rgb_image = self.draw_shapes_boxes_on_image(rgb_image, [small_box_on_empty_part_of_screen])
        # cv2.imshow('rgb_image', rgb_image)
        # cv2.waitKey(0)

        return screen_data

        # original_rgb_image = rgb_image.copy()
        # rgb_image_with_shapes_boxes = self.draw_boxes_and_write_their_sizes_on_image(rgb_image=original_rgb_image,
        #                                                                           boxes=shapes_boxes,
        #                                                                           box_color=(255, 0, 0),
        #                                                                           box_thickness=2)
        #
        # rgb_image_with_screen_boxes = self.draw_boxes_and_write_their_sizes_on_image(rgb_image=original_rgb_image,
        #                                                                           boxes=screen_boxes,
        #                                                                           box_color=(0, 255, 0),
        #                                                                           box_thickness=2)
        # cv2.imshow('rgb_image_with_screen_boxes', rgb_image_with_screen_boxes)
        # cv2.waitKey(0)
        david = 5

    def get_shapes_and_colors_data_from_frame(self, rgb_image):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        contours = self.get_image_contours(rgb_image)
        # screen_box = self.detect_screen_box(contours)
        # rgb_image = write_frame_number_on_image(rgb_image, frame_index + 1, num_of_frames)
        # self.image_data['screen_data']['screen_box'] = screen_box
        # self.image_data['screen_data']['screen_box_dict'] = self.convert_box_to_dict(screen_box)

        shapes_boxes = self.detect_shapes_boxes_in_screen(contours, rgb_image)
        # all_boxes = get_all_contours_boxes(contours)
        shapes_data = self.detect_shapes_data(shapes_boxes, rgb_image, self.template_gray_images)
        screen_data = self.get_screen_data(rgb_image, contours, shapes_data)
        # small_box_on_empty_part_of_screen = self.get_small_box_on_empty_part_of_screen(shapes_data, screen_box)
        # screen_color_data = self.get_screen_color_data(rgb_image, small_box_on_empty_part_of_screen)
        # self.image_data['screen_data']['screen_color_data'] = screen_color_data
        # self.image_data['screen_data']['small_box_on_empty_part_of_screen'] = small_box_on_empty_part_of_screen
        # small_box_on_empty_part_of_screen_dict = self.convert_box_to_dict(small_box_on_empty_part_of_screen)
        # self.image_data['screen_data'][
        #    'small_box_on_empty_part_of_screen_dict'] = small_box_on_empty_part_of_screen_dict
        self.image_data['shapes_data'] = shapes_data
        self.image_data['screen_data'] = screen_data
        return self.image_data

    def check_if_image_contains_one_of_the_shapes(self, rgb_image, shape_box, template_gray_images):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        x_min = int(shape_box[0])
        y_min = int(shape_box[1])
        x_max = int(shape_box[2])
        y_max = int(shape_box[3])

        shape_width = x_max - x_min
        shape_height = y_max - y_min

        max_dim = max(shape_width, shape_height)
        gray_image_with_single_shape = gray_image[y_min:y_max, x_min:x_max]

        num_of_template_shapes = len(template_gray_images)
        max_values = {}
        sum_vals_all_shapes = {}
        template_images_folder_path = 'original_shapes_jpg'
        # ids = find_closest_match(template_images_folder_path, gray_image_with_single_shape)

        non_zero_elements_for_all_templates = np.zeros((num_of_template_shapes))
        first_key = list(template_gray_images.keys())[0]
        first_template_gray_image = template_gray_images[first_key]
        template_height, template_width = first_template_gray_image.shape[:2]
        template_dim = (template_height, template_width)

        resized_gray_image_with_single_shape = cv2.resize(gray_image_with_single_shape, template_dim,
                                                          interpolation=cv2.INTER_AREA)

        resized_gray_image_with_single_shape_without_border = self.get_image_without_border(
            resized_gray_image_with_single_shape)

        (template_thresh, resized_binary_image_with_single_shape_white_background_without_border) = cv2.threshold(
            resized_gray_image_with_single_shape_without_border, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        resized_binary_image_with_single_shape_without_border = cv2.bitwise_not(
            resized_binary_image_with_single_shape_white_background_without_border)

        for template_shape_index, template_image_name in enumerate(template_gray_images):
            template_gray_image = template_gray_images[template_image_name]
            (template_thresh, binary_template_white_background) = cv2.threshold(template_gray_image, 128, 255,
                                                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            binary_template = cv2.bitwise_not(binary_template_white_background)

            binary_template_without_border = self.get_image_without_border(binary_template)

            diff_images_without_border = resized_binary_image_with_single_shape_without_border.astype(
                float) - binary_template_without_border.astype(float)
            # unique_diff_images = np.unique(diff_images_without_border)
            abs_diff_images_without_border = abs(diff_images_without_border).astype(np.uint8)
            # unique_abs_diff_images = np.unique(abs_diff_images_without_border)
            # sum_vals_specific_shape = np.sum(abs_diff_images_without_border)

            # def build_contours_and_plot_them_on_image(self, binary_image):
            #     single_contour_with_max_vertices, single_approx_contour, single_approx_contour_with_reduced_angles_and_sides = self.build_contours(
            #         binary_image)
            #     rgb_image_with_single_contour_with_max_vertices = self.plot_contours_on_image(binary_image,
            #                                                                                   single_contour_with_max_vertices)
            #     rgb_image_with_single_approx_contour = self.plot_contours_on_image(binary_image,
            #                                                                        single_approx_contour)
            #     rgb_image_with_single_approx_contour_with_reduced_angles_and_sides = self.plot_contours_on_image(
            #         binary_image,
            #         single_approx_contour_with_reduced_angles_and_sides)
            #     return rgb_image_with_single_contour_with_max_vertices, rgb_image_with_single_approx_contour, rgb_image_with_single_approx_contour_with_reduced_angles_and_sides

            rgb_with_contour_with_max_vertices, rgb_with_contour_with_single_approx_contour, resized_binary_image_with_single_shape_without_border_with_vertices = self.build_contours_and_plot_them_on_image(
                resized_binary_image_with_single_shape_without_border)
            rgb_template_with_contour_with_max_vertices, rgb_template_with_contour_with_single_approx_contour, binary_template_without_border_with_vertices = self.build_contours_and_plot_them_on_image(
                binary_template_without_border)

            non_zero_elements_for_current_template = np.count_nonzero(abs_diff_images_without_border)
            non_zero_elements_for_all_templates[template_shape_index] = non_zero_elements_for_current_template

            # images = [
            #     (abs_diff_images_without_border, 'abs_diff_images_without_border'),
            #     (resized_binary_image_with_single_shape_without_border_with_vertices, 'resized_binary_image_with_single_shape_without_border_with_vertices'),
            #     (binary_template_without_border_with_vertices, 'binary_template_without_border_with_vertices'),
            #     (rgb_with_contour_with_max_vertices, 'rgb_with_contour_with_max_vertices'),
            #     (rgb_with_contour_with_single_approx_contour, 'rgb_with_contour_with_single_approx_contour')
            # ]
            # combined_rgb_images = self.get_combined_images(images)
            # resized_combined_rgb_images = self.resize_image(rgb_image=combined_rgb_images, scale_percent=500)
            #
            # cv2.imshow('resized_combined_rgb_images', resized_combined_rgb_images)
            # cv2.waitKey(0)

            # cv2.imshow('abs_diff_images_without_border', abs_diff_images_without_border)
            # cv2.imshow('resized_gray_image_with_single_shape_without_border',
            #            resized_binary_image_with_single_shape_without_border_with_vertices)
            # cv2.imshow('template_gray_image_without_border', binary_template_without_border_with_vertices)
            # cv2.waitKey(1)
        min_non_zero_elements = np.min(non_zero_elements_for_all_templates)
        # print('min_non_zero_elements')
        # print(min_non_zero_elements)
        # print()
        if min_non_zero_elements >= 5000:
            return False
        else:
            return True

    def detect_shapes_data(self, shapes_boxes, rgb_image, template_gray_images):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        shapes_data = []
        for shape_index, shape_box in enumerate(shapes_boxes):
            x_min = int(shape_box[0])
            y_min = int(shape_box[1])
            x_max = int(shape_box[2])
            y_max = int(shape_box[3])

            # shape_width = x_max - x_min
            # shape_height = y_max - y_min

            # max_dim = max(shape_width, shape_height)
            gray_image_with_single_shape = gray_image[y_min:y_max, x_min:x_max]
            # rgb_image_with_single_shape = rgb_image[y_min:y_max, x_min:x_max, :]

            # resized_rgb_image_with_single_shape = self.get_image_in_template_dimensions(template_gray_images,
            #                                                                             rgb_image_with_single_shape)

            # resized_rgb_image_with_single_shape_without_border = self.get_image_without_border(
            #     resized_rgb_image_with_single_shape)

            resized_binary_image_with_single_shape_without_border = self.convert_gray_image_to_binary_image_with_same_dims_as_template_image(
                template_gray_images,
                gray_image_with_single_shape)

            # is_shape_legal = self.check_if_image_contains_one_of_the_shapes(rgb_image, shape_box,
            #                                                                 template_gray_images)
            small_shape_box_for_color_detection = self.get_partial_box(shape_box, ratio=0.45)
            # if is_shape_legal == False:
            #     continue
            # shape_name = list(sorted_max_values_for_single_shape.keys())[0]
            shape_contour_with_max_vertices = self.detect_shape_contour(
                resized_binary_image_with_single_shape_without_border)
            shape_name = self.detect_shape_type_by_image(resized_binary_image_with_single_shape_without_border)
            shape_color_data = self.get_color_data_in_box(rgb_image, small_shape_box_for_color_detection)
            # single_shape_data = {'shape_name': shape_name, 'shape_box': shape_box, 'sorted_max_values': sorted_max_values_for_single_shape}
            single_shape_data = {'shape_name': shape_name,
                                 'shape_color_data': shape_color_data,
                                 'shape_box': shape_box,
                                 'shape_box_dict': self.convert_box_to_dict(shape_box),
                                 'small_shape_box_for_color_detection': small_shape_box_for_color_detection,
                                 'resized_binary_image_with_single_shape_without_border': resized_binary_image_with_single_shape_without_border}
            shapes_data.append(single_shape_data)
        return shapes_data

    def filter_boxes_in_ranges(self, boxes, min_width, max_width, min_height, max_height):
        filtered_boxes = []
        for box in boxes:
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])

            width = x_max - x_min
            height = y_max - y_min
            is_shape_dimensions = self.check_if_shape_dimensions(width=width,
                                                                 height=height,
                                                                 min_width=min_width,
                                                                 max_width=max_width,
                                                                 min_height=min_height,
                                                                 max_height=max_height)
            if is_shape_dimensions:
                filtered_boxes.append(box)
        np_filtered_boxes = np.array(filtered_boxes)
        return np_filtered_boxes

    def sort_boxes(self, boxes, threshold):
        if len(boxes) == 0:
            return boxes
        num_of_boxes = boxes.shape[0]
        ascending_indexes = np.arange(1, num_of_boxes + 1, 1, dtype=int)
        extra_column = np.vstack(ascending_indexes)
        boxes_with_indexes = np.hstack((boxes, extra_column))
        # sorted_boxes_by_y_vals = boxes_with_indexes
        y_indexes = np.argsort(boxes_with_indexes[:, 1])
        sorted_boxes_by_y_vals = boxes_with_indexes[y_indexes]
        sorted_boxes_by_y_vals[sorted_boxes_by_y_vals[:, 1].argsort()]
        y_values = sorted_boxes_by_y_vals[:, 1]
        first_y = sorted_boxes_by_y_vals[0, 1]
        dists_from_first_y = y_values - first_y
        row_order_on_screen = (dists_from_first_y > threshold).astype(int) + 1
        index_of_first_box_in_second_row = np.argmax(row_order_on_screen)
        row_order_on_screen = np.vstack(row_order_on_screen)
        boxes_with_rows_order_sorted_by_y_vals = np.hstack((sorted_boxes_by_y_vals, row_order_on_screen))
        boxes_of_first_row_on_screen = boxes_with_rows_order_sorted_by_y_vals[0:index_of_first_box_in_second_row, :]
        boxes_of_second_and_third_row_on_screen = boxes_with_rows_order_sorted_by_y_vals[
                                                  index_of_first_box_in_second_row:, :]
        y_values_in_second_and_third_rows = boxes_of_second_and_third_row_on_screen[:, 1]
        first_y_in_second_row = boxes_of_second_and_third_row_on_screen[0, 1]
        dists_from_first_y_in_second_row = y_values_in_second_and_third_rows - first_y_in_second_row
        row_order_on_screen_on_second_row = (dists_from_first_y_in_second_row > threshold).astype(int) + 2
        max_row_order_on_screen_on_second_row = np.max(row_order_on_screen_on_second_row)
        min_row_order_on_screen_on_second_row = np.min(row_order_on_screen_on_second_row)
        if max_row_order_on_screen_on_second_row == min_row_order_on_screen_on_second_row:
            boxes_of_second_row_on_screen = boxes_of_second_and_third_row_on_screen
        else:
            index_of_first_box_in_third_row = np.argmax(row_order_on_screen_on_second_row)
            row_order_on_screen_on_second_row = np.vstack(row_order_on_screen_on_second_row)
            row_order_on_screen[index_of_first_box_in_second_row:] = row_order_on_screen_on_second_row
            a = boxes_with_rows_order_sorted_by_y_vals[:, 5]
            squeezed_row_order_on_screen = np.squeeze(row_order_on_screen)
            boxes_with_rows_order_sorted_by_y_vals[:, 5] = squeezed_row_order_on_screen
            boxes_of_second_row_on_screen = boxes_of_second_and_third_row_on_screen[0:index_of_first_box_in_third_row]

        x_indexes_row_1 = np.argsort(boxes_of_first_row_on_screen[:, 0])
        boxes_of_first_row_on_screen_ordered_by_x = boxes_of_first_row_on_screen[x_indexes_row_1]
        x_indexes_row_2 = np.argsort(boxes_of_second_row_on_screen[:, 0])
        boxes_of_second_row_on_screen_ordered_by_x = boxes_of_second_row_on_screen[x_indexes_row_2]
        sorted_boxes_with_indexes = np.vstack(
            (boxes_of_first_row_on_screen_ordered_by_x, boxes_of_second_row_on_screen_ordered_by_x))
        sorted_boxes = sorted_boxes_with_indexes[:, :-2]
        return sorted_boxes

    def check_if_shape_dimensions(self, width, height, min_width, max_width, min_height, max_height):
        diff_width_height = abs(width - height)
        if max_width > width > min_width and \
                max_height > height > min_height and \
                diff_width_height < 20:
            return True
        else:
            return False

    def detect_shapes_boxes_in_screen(self, contours, rgb_image):

        boxes = list()
        # local_max_screen_width = 900
        # local_min_screen_width = 500
        # local_max_screen_height = 500
        # local_min_screen_height = 200
        rgb_image_with_data = rgb_image.copy()
        for single_contour in contours:
            x, y, w, h = cv2.boundingRect(single_contour)
            ratio_w_h = w / float(h)

            if self.max_square_size > w > self.min_square_size and \
                    self.max_square_size > h > self.min_square_size and \
                    self.max_width_height_ratio > ratio_w_h > (1 / self.max_width_height_ratio):
                x1 = x
                x2 = x1 + w
                y1 = y
                y2 = y1 + h
                shape_box_candidate = (x1, y1, x2, y2)
                rgb_image_with_data = self.draw_boxes_on_image(rgb_image_with_data, [shape_box_candidate],
                                                                         box_color=(255, 0, 0), box_thickness=2)

                is_shape_legal = self.check_if_image_contains_one_of_the_shapes(rgb_image, shape_box_candidate,
                                                                                self.template_gray_images)
                if is_shape_legal == True:
                    boxes.append(shape_box_candidate)
        boxes = np.array(boxes)
        boxes_non_max_suppression = imutils.object_detection.non_max_suppression(boxes)
        for single_box_non_max_supression in boxes_non_max_suppression:
            is_shape_legal = self.check_if_image_contains_one_of_the_shapes(rgb_image, single_box_non_max_supression, self.template_gray_images)

        sorted_boxes = self.sort_boxes(boxes_non_max_suppression, threshold=40)

        # rgb_image_with_shapes_boxes = rgb_image.copy()
        # rgb_image_non_max_suppression = self.draw_boxes_on_image(rgb_image_with_shapes_boxes, boxes_non_max_suppression,
        #                                                          box_color=(255, 0, 0), box_thickness=2)
        # rgb_image_sorted_boxes = self.draw_boxes_on_image(rgb_image_with_shapes_boxes, sorted_boxes, box_color=(255, 0, 0),
        #                                                   box_thickness=2)
        return sorted_boxes

    def check_if_box_inside_screen(self, shape_box, screen_box):
        x1_screen = screen_box[0]
        y1_screen = screen_box[1]
        x2_screen = screen_box[2]
        y2_screen = screen_box[3]
        screen_width = x2_screen - x1_screen
        screen_height = y2_screen - y1_screen

        x1_shape = shape_box[0]
        y1_shape = shape_box[1]
        x2_shape = shape_box[2]
        y2_shape = shape_box[3]
        shape_width = x2_shape - x1_shape
        shape_height = y2_shape - y1_shape

        if x1_screen < x1_shape < x2_screen and \
                x1_screen < x2_shape < x2_screen and \
                y1_screen < y1_shape < y2_screen and \
                y1_screen < y2_shape < y2_screen and \
                shape_width < screen_width and \
                shape_height < screen_height:
            return True
        else:
            return False

    def calc_angles(self, coords):
        vertices = np.concatenate((coords, coords[:1]), axis=0)
        vectors = np.diff(vertices, axis=0)
        shape_vectors = np.transpose(vectors, (1, 0))
        L2_norms_of_shape_vectors = np.sqrt((shape_vectors * shape_vectors).sum(axis=0))
        zero_indexes = np.where(L2_norms_of_shape_vectors == 0)[0]
        # prevent division by zero
        L2_norms_of_shape_vectors[zero_indexes] = 1
        unit_shape_vectors = shape_vectors / L2_norms_of_shape_vectors
        unit_shape_vectors[:, zero_indexes] = np.zeros((2, zero_indexes.size))
        first_vector = np.zeros((2, 1), unit_shape_vectors.dtype)
        first_vector[:, 0] = unit_shape_vectors[:, 0]
        num_of_vectors = unit_shape_vectors.shape[1]
        second_to_last_vectors = unit_shape_vectors[:, 1:num_of_vectors]
        unit_successor_shape_vectors = np.concatenate((second_to_last_vectors, first_vector), axis=1)
        unit_successor_shape_vectors_negative_directions = -unit_successor_shape_vectors
        cos_angles = np.sum(unit_shape_vectors * unit_successor_shape_vectors_negative_directions, axis=0)
        cos_angles[cos_angles > 1] = 1
        cos_angles[cos_angles < -1] = -1
        angles_radians = np.arccos(cos_angles)
        angles_degrees = np.rad2deg(angles_radians)
        return angles_degrees

    def remove_redundant_points_by_angles(self, points, angles):
        angles_too_big_indexes = angles > 160
        points_indexes_to_remove = np.roll(angles_too_big_indexes, 1)
        points_indexes_to_keep = np.logical_not(points_indexes_to_remove)
        points_after_removal = points[points_indexes_to_keep]
        return points_after_removal

    def remove_redundant_points_by_sides_lengths(self, points, sides_lengths):
        max_side = np.max(sides_lengths)
        ratios = sides_lengths / max_side
        points_indexes_to_remove = ratios < 0.3
        points_indexes_to_keep = np.logical_not(points_indexes_to_remove)
        points_after_removal = points[points_indexes_to_keep]
        return points_after_removal

    def get_sides_of_contour(self, single_approx_contour):
        vertices = np.concatenate((single_approx_contour, single_approx_contour[:1]), axis=0)
        vectors = np.diff(vertices, axis=0)
        sides = np.linalg.norm(vectors, axis=-1)
        # sides1 = np.hypot(*vectors.T) #for 2D only
        return sides

    def reduce_sides_of_contour_by_too_big_angles_and_too_little_side_lengths(self, single_approx_contour):
        single_approx_contour = np.squeeze(single_approx_contour)
        angles = self.calc_angles(single_approx_contour)
        single_approx_contour = self.remove_redundant_points_by_angles(single_approx_contour, angles)
        sides = self.get_sides_of_contour(single_approx_contour)
        single_approx_contour = self.remove_redundant_points_by_sides_lengths(single_approx_contour, sides)
        return single_approx_contour

    def detect_shape_type_by_contour(self, single_contour_with_max_vertices, single_approx_contour,
                                     resized_binary_image_with_single_shape_without_border):
        shape_name = 'unidentified'
        num_of_sides = single_approx_contour.shape[0]
        angles = self.calc_angles(single_approx_contour)
        sides = self.get_sides_of_contour(single_approx_contour)
        # print(f'num_of_sides = {num_of_sides}')
        # if the shape is a triangle, it will have 3 vertices
        if num_of_sides == 3:
            shape_name = 'triangle'
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif num_of_sides == 4:
            sorted_angles = np.sort(angles)
            num_of_angles_to_test = 2
            first_small_angles = sorted_angles[0:num_of_angles_to_test]
            last_big_angles = sorted_angles[-num_of_angles_to_test:]
            mean_small_angles = np.mean(first_small_angles)
            mean_big_angles = np.mean(last_big_angles)
            ratio_angles = mean_big_angles / mean_small_angles
            rhombus_ratio_threshold = 1.15
            if ratio_angles > rhombus_ratio_threshold:
                shape_name = 'rhombus'
            else:
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(single_approx_contour)
                ar = w / float(h)
                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                shape_name = 'square' if ar >= 0.9 and ar <= 1.1 else 'rectangle'
        # if the shape is a pentagon, it will have 5 vertices
        elif num_of_sides == 5:
            shape_name = 'pentagon'
        # otherwise, we assume the shape is a circle
        elif num_of_sides >= 6 and num_of_sides <= 9:
            peri = cv2.arcLength(single_contour_with_max_vertices, True)
            single_approx_contour_for_octagon_and_circle = cv2.approxPolyDP(single_contour_with_max_vertices,
                                                                            0.008 * peri, True)
            resized_binary_image_with_single_shape_without_border_with_vertices = self.plot_contours_on_image(
                resized_binary_image_with_single_shape_without_border,
                single_approx_contour_for_octagon_and_circle)
            # cv2.imshow('resized_binary_image_with_single_shape_without_border_with_vertices', resized_binary_image_with_single_shape_without_border_with_vertices)
            # cv2.waitKey(0)
            num_of_sides_new = single_approx_contour_for_octagon_and_circle.shape[0]
            if num_of_sides_new >= 12:
                shape_name = 'circle'
            else:
                shape_name = 'octagon'
        elif num_of_sides >= 10 and num_of_sides <= 40:
            sorted_angles = np.sort(angles)
            num_of_angles_to_test = 3
            first_small_angles = sorted_angles[0:num_of_angles_to_test]
            last_big_angles = sorted_angles[-num_of_angles_to_test:]
            mean_small_angles = np.mean(first_small_angles)
            mean_big_angles = np.mean(last_big_angles)
            ratio_angles = mean_big_angles / mean_small_angles
            star_ratio_threshold = 1.6
            if ratio_angles > star_ratio_threshold:
                shape_name = 'star'
            else:
                shape_name = 'circle'
        # return the name of the shape
        return shape_name

    def detect_shape_contour(self, binary_image):
        findContoursResults = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(findContoursResults)
        num_of_contours = len(contours)
        max_contour_index = -1
        max_num_rows = -1
        for contour_index, single_contour in enumerate(contours):
            num_rows = single_contour.shape[0]
            if num_rows > max_num_rows:
                max_num_rows = num_rows
                max_contour_index = contour_index

        shape_contour_with_max_vertices = contours[max_contour_index]
        return shape_contour_with_max_vertices

    def get_small_inside_image_for_color_detection(self, resized_rgb_image_with_single_shape_without_border):
        (height, width) = resized_rgb_image_with_single_shape_without_border.shape[:2]
        radiuo_ratio = 0.2
        start_row_index = int((1 - radiuo_ratio) * 0.5 * height)
        end_row_index = int((1 + radiuo_ratio) * 0.5 * height)

        start_column_index = int((1 - radiuo_ratio) * 0.5 * width)
        end_column_index = int((1 + radiuo_ratio) * 0.5 * width)

        ratio_image = resized_rgb_image_with_single_shape_without_border[start_row_index:end_row_index,
                      start_column_index:end_column_index]
        return ratio_image

    # def detect_balloon_data(frame, sub_image):
    #     colorCode = cv2.COLOR_BGR2HSV
    #     num_of_colors = len(colors_ranges_info)
    #     colors_num_of_pixels = np.zeros([num_of_colors])
    #     for index, single_color_range_info in enumerate(colors_ranges_info):
    #         lower = single_color_range_info["lower"]
    #         upper = single_color_range_info["upper"]
    #         single_color_num_of_pixels = count_pixels_in_color_range(lower, upper, frame, colorCode)
    #         colors_num_of_pixels[index] = single_color_num_of_pixels
    #     max_index = colors_num_of_pixels.argmax()
    #     balloon_data = colors_ranges_info[max_index]
    #     if np.sum(colors_num_of_pixels) == 0:
    #         # distances = np.zeros([num_of_colors])
    #         # hsv_sub_image = cv2.cvtColor(sub_image, colorCode)
    #         # average_pixel = np.mean(hsv_sub_image, axis=(0, 1))
    #         # for index, single_color_range_info in enumerate(colors_ranges_info):
    #         #     np_lower = np.array(single_color_range_info["lower"])
    #         #     np_upper = np.array(single_color_range_info["upper"])
    #         #     dist_from_point_to_rectangular_box = calc_dist_from_point_to_rectangular_box(average_pixel, np_lower, np_upper)
    #         #     distances[index] = dist_from_point_to_rectangular_box
    #         # min_index = distances.argmin()
    #         # balloon_data = colors_ranges_info[min_index]
    #         return unknown_color_info
    #     return balloon_data

    def get_sub_image_by_box(self, rgb_image, box):
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        rgb_sub_image = rgb_image[y_min:y_max + 1, x_min:x_max + 1, :]
        return rgb_sub_image

    def count_pixels_in_image_box_in_color_range(self, box, colorLower, colorUpper, rgb_image, color_name,
                                                 code=cv2.COLOR_BGR2HSV):
        rgb_sub_image = self.get_sub_image_by_box(rgb_image, box)
        num_of_pixels = self.count_pixels_in_color_range(colorLower, colorUpper, rgb_sub_image, color_name, code)
        return num_of_pixels

    def count_pixels_in_color_range(self, colorLower, colorUpper, frame, color_name, code=cv2.COLOR_BGR2HSV):
        hsv_frame = cv2.cvtColor(frame, code)

        np_colorLower = np.array(colorLower)
        np_colorUpper = np.array(colorUpper)
        hsv_frame_low = np.array((np.min(hsv_frame[:, :, 0]), np.min(hsv_frame[:, :, 1]), np.min(hsv_frame[:, :, 2])))
        hsv_frame_high = np.array((np.max(hsv_frame[:, :, 0]), np.max(hsv_frame[:, :, 1]), np.max(hsv_frame[:, :, 2])))

        new_colorLower = np.minimum(np_colorLower, hsv_frame_low)
        new_colorUpper = np.maximum(np_colorUpper, hsv_frame_high)

        delta_val = 3
        np_new_colorLower = np.maximum(np_colorLower - delta_val, np.array((0, 0, 0)))
        np_new_colorUpper = np.minimum(np_colorUpper + delta_val, np.array((255, 255, 255)))

        # print(new_colorLower)
        # print(new_colorUpper)
        # print()

        mask = cv2.inRange(hsv_frame, np_new_colorLower, np_new_colorUpper)
        num_of_pixels = cv2.countNonZero(mask)
        return num_of_pixels

    # def get_shape_color(self, image_with_approx_single_color):
    #     colorCode = cv2.COLOR_BGR2HSV
    #     colors_ranges_info = self.colors_ranges_info_tello_without_stick
    #     num_of_colors = len(colors_ranges_info)
    #     colors_num_of_pixels = np.zeros([num_of_colors])
    #     for index, single_color_range_info in enumerate(colors_ranges_info):
    #         lower = single_color_range_info["lower"]
    #         upper = single_color_range_info["upper"]
    #         single_color_num_of_pixels = self.count_pixels_in_color_range(lower, upper, image_with_approx_single_color,
    #                                                                       colorCode)
    #         colors_num_of_pixels[index] = single_color_num_of_pixels
    #     max_index = colors_num_of_pixels.argmax()
    #     shape_color_data = colors_ranges_info[max_index]
    #     if np.sum(colors_num_of_pixels) == 0:
    #         return self.unknown_color_info
    #     return shape_color_data

    def detect_shape_color(self, resized_rgb_image_with_single_shape_without_border,
                           shape_contour_with_max_vertices):
        image_with_approx_single_color = self.get_small_inside_image_for_color_detection(
            resized_rgb_image_with_single_shape_without_border)
        shape_color_data = self.get_shape_color(image_with_approx_single_color)
        return shape_color_data

    def detect_shape_type_by_image(self, resized_binary_image_with_single_shape_without_border):
        rgb_with_contour_with_max_vertices, rgb_with_contour_with_single_approx_contour, resized_binary_image_with_single_shape_without_border_with_vertices = self.build_contours_and_plot_them_on_image(
            resized_binary_image_with_single_shape_without_border)
        single_contour_with_max_vertices, single_approx_contour, single_approx_contour_with_reduced_angles_and_sides = self.build_contours(
            resized_binary_image_with_single_shape_without_border)
        shape_name = self.detect_shape_type_by_contour(single_contour_with_max_vertices,
                                                       single_approx_contour_with_reduced_angles_and_sides,
                                                       resized_binary_image_with_single_shape_without_border)
        # I assume there's only only one contour in contours
        return shape_name

    def plot_contours_on_image(self, binary_image, single_contour):
        squeezed_single_contour = np.squeeze(single_contour)
        (height, width) = binary_image.shape
        img_rgb = np.zeros((height, width, 3), np.uint8)
        img_rgb[:, :, 0] = binary_image
        img_rgb[:, :, 1] = binary_image
        img_rgb[:, :, 2] = binary_image
        if len(single_contour.shape) == 2 and len(squeezed_single_contour.shape) == 1:
            return img_rgb

        middle_image_pixel = (int(0.5 * width), int(0.5 * height))

        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 0.3
        text_color = (255, 0, 255)
        text_thickness = 1

        middle_image_font = cv2.FONT_HERSHEY_COMPLEX
        middle_image_font_scale = 0.6
        middle_image_text_color = (0, 255, 0)
        middle_image_text_thickness = 1

        contourColor = (0, 0, 255)
        cv2.drawContours(img_rgb, [squeezed_single_contour], -1, contourColor, 2)

        num_of_contour_points = squeezed_single_contour.shape[0]
        circle_radius = 3
        circle_color = (0, 255, 0)
        circle_thickness = -1

        cv2.putText(img_rgb,
                    f'{num_of_contour_points} coords', middle_image_pixel, middle_image_font,
                    middle_image_font_scale, middle_image_text_color, middle_image_text_thickness, cv2.LINE_AA)

        for counterPointIndex in range(0, num_of_contour_points):
            current_contour_point = squeezed_single_contour[counterPointIndex, :]
            current_contour_point_tuple = (current_contour_point[0], current_contour_point[1])
            cv2.circle(img_rgb, current_contour_point_tuple, circle_radius, circle_color, circle_thickness)

            cv2.putText(img_rgb,
                        f'{counterPointIndex + 1}', current_contour_point_tuple, font,
                        font_scale, text_color, text_thickness, cv2.LINE_AA)

        # center_circle_radius = 5
        # center_circle_color = (255, 0, 255)
        # cv2.circle(img_rgb, (center_x, center_y), center_circle_radius, center_circle_color, circle_thickness)
        return img_rgb

    def draw_circle_on_image(self, rgb_image, circle_center, circle_color=(0, 255, 0), radius=10):
        circle_thickness = -1
        x = int(circle_center[1])
        y = int(circle_center[0])
        cv2.circle(rgb_image, (y, x), radius, circle_color, circle_thickness)
        return rgb_image

    def build_contours(self, binary_image):
        findContoursResults = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(findContoursResults)
        max_contour_index = -1
        max_num_rows = -1
        for contour_index, single_contour in enumerate(contours):
            num_rows = single_contour.shape[0]
            if num_rows > max_num_rows:
                max_num_rows = num_rows
                max_contour_index = contour_index

        single_contour_with_max_vertices = contours[max_contour_index]
        if single_contour_with_max_vertices.shape[0] <= 2:
            print('error. too few vertices in contour')
            return None, None, None

        peri = cv2.arcLength(single_contour_with_max_vertices, True)
        single_approx_contour = single_contour_with_max_vertices
        single_approx_contour = cv2.approxPolyDP(single_approx_contour, 0.02 * peri, True)
        single_approx_contour_with_reduced_angles_and_sides = self.reduce_sides_of_contour_by_too_big_angles_and_too_little_side_lengths(
            single_approx_contour)
        return single_contour_with_max_vertices, single_approx_contour, single_approx_contour_with_reduced_angles_and_sides

    def build_contours_and_plot_them_on_image(self, binary_image):
        single_contour_with_max_vertices, single_approx_contour, single_approx_contour_with_reduced_angles_and_sides = self.build_contours(
            binary_image)
        rgb_image_with_single_contour_with_max_vertices = self.plot_contours_on_image(binary_image,
                                                                                      single_contour_with_max_vertices)
        rgb_image_with_single_approx_contour = self.plot_contours_on_image(binary_image,
                                                                           single_approx_contour)
        rgb_image_with_single_approx_contour_with_reduced_angles_and_sides = self.plot_contours_on_image(binary_image,
                                                                                                         single_approx_contour_with_reduced_angles_and_sides)
        return rgb_image_with_single_contour_with_max_vertices, rgb_image_with_single_approx_contour, rgb_image_with_single_approx_contour_with_reduced_angles_and_sides

    def check_if_1_channel_image(self, image):
        if len(image.shape) == 2:
            return True
        else:
            return False

    def get_image_without_border(self, image):
        delta_pixels_to_remove_border = 19
        is_1_channel_image = self.check_if_1_channel_image(image)
        height = image.shape[0]
        width = image.shape[1]
        if is_1_channel_image == True:
            image_without_border = image[delta_pixels_to_remove_border:height - delta_pixels_to_remove_border,
                                   delta_pixels_to_remove_border:width - delta_pixels_to_remove_border]
        else:
            image_without_border = image[delta_pixels_to_remove_border:height - delta_pixels_to_remove_border,
                                   delta_pixels_to_remove_border:width - delta_pixels_to_remove_border,
                                   :]
        return image_without_border

    def get_image_in_template_dimensions(self, template_gray_images, image):
        first_template_image_name = list(template_gray_images.keys())[0]
        first_template_gray_image = template_gray_images[first_template_image_name]
        template_height, template_width = first_template_gray_image.shape[:2]
        template_dim = (template_height, template_width)
        resized_image = cv2.resize(image, template_dim, interpolation=cv2.INTER_AREA)
        return resized_image

    def convert_gray_image_to_binary_image_with_same_dims_as_template_image(self, template_gray_images,
                                                                            gray_image_with_single_shape):

        # template_image_name = list(template_gray_images.keys())[0]
        # #for shape_index, template_image_name in enumerate(template_gray_images):
        # template_gray_image = template_gray_images[template_image_name]
        #
        # template_height, template_width = template_gray_image.shape[:2]
        #
        #
        #
        #
        # template_dim = (template_height, template_width)
        # resized_gray_image_with_single_shape = cv2.resize(gray_image_with_single_shape, template_dim, interpolation=cv2.INTER_AREA)

        resized_gray_image_with_single_shape = self.get_image_in_template_dimensions(template_gray_images,
                                                                                     gray_image_with_single_shape)

        resized_gray_image_with_single_shape_without_border = self.get_image_without_border(
            resized_gray_image_with_single_shape)

        # height = resized_gray_image_with_single_shape_without_border.shape[0]
        # width = resized_gray_image_with_single_shape_without_border.shape[1]
        # inside_shape_pixel_val = resized_gray_image_with_single_shape_without_border[int(0.5 * height), int(0.5 * width)]
        # outside_shape_pixel_val = resized_gray_image_with_single_shape_without_border[0, 0]
        # threshold_val = int(0.5*(inside_shape_pixel_val + outside_shape_pixel_val))
        #
        # resized_binary_image_with_single_shape_without_border = resized_gray_image_with_single_shape_without_border
        # if inside_shape_pixel_val > outside_shape_pixel_val:
        #     resized_binary_image_with_single_shape_without_border[resized_binary_image_with_single_shape_without_border > threshold_val] = 255
        #     resized_binary_image_with_single_shape_without_border[
        #         resized_binary_image_with_single_shape_without_border <= threshold_val] = 0
        # else:
        #     resized_binary_image_with_single_shape_without_border[resized_binary_image_with_single_shape_without_border > threshold_val] = 0
        #     resized_binary_image_with_single_shape_without_border[
        #         resized_binary_image_with_single_shape_without_border <= threshold_val] = 255

        (template_thresh, resized_binary_image_with_single_shape_without_border_white_background) = cv2.threshold(
            resized_gray_image_with_single_shape_without_border, 128, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        resized_binary_image_with_single_shape_without_border = cv2.bitwise_not(
            resized_binary_image_with_single_shape_without_border_white_background)

        # cv2.imshow('resized_gray_image_with_single_shape_without_border', resized_gray_image_with_single_shape_without_border)
        # cv2.imshow('resized_binary_image_with_single_shape_without_border', resized_binary_image_with_single_shape_without_border)
        # cv2.waitKey()

        return resized_binary_image_with_single_shape_without_border

    def remove_noise(self, gray_image, kernel_size):
        # kernel_size = 5
        # kernel = np.ones((kernel_size, kernel_size), np.uint8) / (kernel_size * kernel_size)
        # kernel_center = kernel_size // 2
        # #kernel[kernel_center, kernel_center] = 1
        #
        # binary_image_no_noise = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, kernel)

        gray_image_no_noise = cv2.GaussianBlur(gray_image,
                                               (kernel_size, kernel_size),
                                               0)
        return gray_image_no_noise

    def detect_screen_box(self, contours):
        boxes = list()
        for single_contour in contours:
            x, y, w, h = cv2.boundingRect(single_contour)
            if self.max_screen_width > w > self.min_screen_width and \
                    self.max_screen_height > h > self.min_screen_height:
                x1 = x
                x2 = x1 + w
                y1 = y
                y2 = y1 + h
                boxes.append((x1, y1, x2, y2))

        boxes = np.array(boxes)
        boxes = object_detection.non_max_suppression(boxes)
        if len(boxes) == 0:
            return boxes
        screen_box = boxes[0]
        return screen_box

    def expand_1_channel_image_to_3_channels_image(self, image_1_channel):
        rows = image_1_channel.shape[0]
        cols = image_1_channel.shape[1]
        image_3_channels = np.zeros((rows, cols, 3), 'uint8')
        image_3_channels[:, :, 0] = image_1_channel
        image_3_channels[:, :, 1] = image_1_channel
        image_3_channels[:, :, 2] = image_1_channel
        return image_3_channels

    def draw_rgb_lines_on_image(self, lines, binary_image, rgb_color):
        binary_image_with_complete_lines_3_channels = self.expand_1_channel_image_to_3_channels_image(binary_image)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(binary_image_with_complete_lines_3_channels, (x1, y1), (x2, y2), color=rgb_color,
                             thickness=1)
        return binary_image_with_complete_lines_3_channels

    def complete_partial_lines_in_image(self, binary_image):
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 100  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 20  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(binary_image, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        binary_image_with_extra_lines = self.draw_white_lines_on_image(lines, binary_image)
        lines_color = (255, 0, 255)
        rgb_image_with_extra_rgb_lines = self.draw_rgb_lines_on_image(lines, binary_image, lines_color)
        return binary_image_with_extra_lines, rgb_image_with_extra_rgb_lines

    def draw_white_lines_on_image(self, lines, binary_image):
        binary_image_with_extra_lines = binary_image.copy()
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(binary_image_with_extra_lines, (x1, y1), (x2, y2), color=255, thickness=1)
        return binary_image_with_extra_lines

    def resize_image(self, rgb_image, scale_percent):
        if scale_percent == 100:
            return rgb_image
        width = int(rgb_image.shape[1] * scale_percent / 100)
        height = int(rgb_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        interpolation_method = cv2.INTER_AREA
        resized_rgb_image = cv2.resize(rgb_image, dim, interpolation=interpolation_method)
        return resized_rgb_image

    def write_headline_on_image(self, rgb_image, headline):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 0, 255)
        thickness = 1
        org = (20, 20)
        cv2.putText(rgb_image, headline, org, font, fontScale, color, thickness, cv2.LINE_AA)
        return rgb_image

    def write_text_on_image(self, rgb_image, text, position, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.2
        thickness = 2
        cv2.putText(rgb_image, text, position, font, fontScale, color, thickness, cv2.LINE_AA)
        return rgb_image

    def get_combined_images_in_row(self, images):
        seperator_size = 20
        num_of_images = len(images)
        list_images_in_row = []
        for image_index in range(num_of_images):
            current_image = images[image_index][0]
            current_image_name = images[image_index][1]
            if len(current_image.shape) == 2:
                current_image_3_channels = self.expand_1_channel_image_to_3_channels_image(current_image)
            else:
                current_image_3_channels = current_image
            resized_current_image_3_channels = self.resize_image(rgb_image=current_image_3_channels, scale_percent=100)
            resized_current_image_3_channels = self.write_headline_on_image(resized_current_image_3_channels,
                                                                            current_image_name)
            resized_height = resized_current_image_3_channels.shape[0]

            horizontal_separtor_image = np.zeros((resized_height, seperator_size, 3), 'uint8')

            horizontal_separtor_image[:, :, 1] = 255
            list_images_in_row.append(resized_current_image_3_channels)
            if image_index < num_of_images - 1:
                # list_images.append(vertical_separtor_image)
                list_images_in_row.append(horizontal_separtor_image)
        combined_image_in_row = np.hstack(list_images_in_row)
        return combined_image_in_row

    def get_combined_images(self, images):
        num_of_images = len(images)
        sqrt_num_of_images = math.sqrt(num_of_images)
        num_of_images_in_column = int(sqrt_num_of_images)
        num_of_images_in_row = math.ceil(num_of_images / num_of_images_in_column)

        num_of_black_image = num_of_images_in_row * num_of_images_in_column - num_of_images
        height = images[0][0].shape[0]
        width = images[0][0].shape[1]
        for i in range(num_of_black_image):
            black_image = np.zeros((height, width, 3), np.uint8)
            images.append((black_image, 'black_image'))

        seperator_size = 20
        list_images = []
        for image_index in range(num_of_images_in_column):
            start_index = num_of_images_in_row * image_index
            end_index = start_index + num_of_images_in_row
            sub_images = images[start_index:end_index]
            combined_image_in_row = self.get_combined_images_in_row(sub_images)

            resized_width = combined_image_in_row.shape[1]
            vertical_sepertor_image = np.zeros((seperator_size, resized_width, 3), 'uint8')
            vertical_sepertor_image[:, :, 1] = 255
            list_images.append(combined_image_in_row)
            if image_index < num_of_images_in_column - 1:
                list_images.append(vertical_sepertor_image)
        combined_image = np.vstack(list_images)

        return combined_image

    def get_contours_from_binary_image(self, binary_image):
        findContoursResults = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(findContoursResults)
        return contours

    def get_image_contours(self, rgb_image):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        # gray_image_no_noise = self.remove_noise(gray_image=gray_image, kernel_size=5)
        gray_image_no_noise = gray_image
        binary_image_edges_image = cv2.Canny(image=gray_image_no_noise, threshold1=50,
                                             threshold2=0)
        binary_image_with_complete_lines, rgb_image_with_extra_rgb_lines = self.complete_partial_lines_in_image(
            binary_image_edges_image)
        contours = self.get_contours_from_binary_image(binary_image_edges_image)
        image_edges_3_channels = self.expand_1_channel_image_to_3_channels_image(binary_image_edges_image)
        image_with_contours = image_edges_3_channels.copy()

        # for i in range(len(contours)):
        #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        #     cv2.drawContours(image_with_contours, contours, i, color)
        #     single_contour = contours[i]
        #     x, y, w, h = cv2.boundingRect(single_contour)
        #     # if local_max_screen_width > w > local_min_screen_width and \
        #     #         local_max_screen_height > h > local_min_screen_height:
        #     x1 = x
        #     x2 = x1 + w
        #     y1 = y
        #     y2 = y1 + h
        #     box_thickness = 3
        #     cv2.rectangle(image_with_contours, (x1, y1), (x2, y2), color, box_thickness)

        cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 2)
        screen_box = self.detect_screen_box(contours)
        all_boxes = self.get_all_contours_boxes(contours)
        rgb_image_with_all_boxes = rgb_image.copy()
        rgb_image_with_all_boxes = self.draw_boxes_and_write_their_sizes_on_image(rgb_image=rgb_image_with_all_boxes,
                                                                                  boxes=all_boxes,
                                                                                  box_color=(0, 255, 0),
                                                                                  box_thickness=2)

        # images = [
        #     (gray_image, 'gray_image'),
        #     (gray_image_no_noise, 'gray_image_no_noise'),
        #     (binary_image_edges_image, 'binary_image_edges_image'),
        #     (binary_image_with_complete_lines, 'binary_image_with_complete_lines'),
        #     (rgb_image_with_extra_rgb_lines, 'rgb_image_with_extra_rgb_lines'),
        #     (rgb_image, 'rgb_image'),
        #     (image_with_contours, 'image_with_contours'),
        #     (rgb_image_with_all_boxes, 'rgb_image_with_all_boxes')
        # ]
        # combine_images = self.get_combined_images(images)
        #
        # cv2.imshow('combine_images', combine_images)

        # cv2.imshow('gray_image', gray_image)
        # cv2.imshow('gray_image_no_noise', gray_image_no_noise)
        # cv2.imshow('binary_image_with_complete_lines', binary_image_with_complete_lines)
        # cv2.imshow('rgb_image_with_extra_rgb_lines', rgb_image_with_extra_rgb_lines)
        # cv2.imshow('binary_image_edges_image', binary_image_edges_image)
        # cv2.imshow('image_with_contours', image_with_contours)
        # cv2.imshow('rgb_image_with_all_boxes', rgb_image_with_all_boxes)
        # cv2.waitKey(0)

        return contours

    def write_shapes_names_on_image(self, rgb_image, shapes_data):
        for single_shape_data in shapes_data:
            shape_box = single_shape_data['shape_box']
            # sorted_max_values_for_single_shape = single_shape_data['sorted_max_values']
            x_min = int(shape_box[0])
            y_min = int(shape_box[1])
            x_max = int(shape_box[2])
            y_max = int(shape_box[3])
            x_center = int(0.5 * (x_min + x_max))
            y_center = int(0.5 * (y_min + y_max))
            box_center = np.array((x_center, y_center), int)

            # shape_name = list(sorted_max_values_for_single_shape.keys())[0]
            shape_name = single_shape_data['shape_name']
            shape_color_data = single_shape_data['shape_color_data']

            shape_color_name = shape_color_data["name"]
            shape_color = shape_color_data["rgb_color"]

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
            thickness = 2
            cv2.putText(rgb_image, f'{shape_name}', box_center, font,
                        fontScale, shape_color, thickness, cv2.LINE_AA)
            cv2.putText(rgb_image, f'{shape_color_name}', box_center + (0, 30), font,
                        fontScale, shape_color, thickness, cv2.LINE_AA)

        return rgb_image

    def write_shapes_names_on_image_by_their_order(self, rgb_image, shapes_data):
        x = 50
        y = 100
        for single_shape_data in shapes_data:
            shape_name = single_shape_data['shape_name']
            shape_color_data = single_shape_data['shape_color_data']

            shape_color_name = shape_color_data["name"]
            shape_color = shape_color_data["rgb_color"]

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1.0
            thickness = 2
            cv2.putText(rgb_image, f'{shape_name}', (x, y), font,
                        fontScale, shape_color, thickness, cv2.LINE_AA)
            y += 25
            cv2.putText(rgb_image, f'{shape_color_name}', (x, y), font,
                        fontScale, shape_color, thickness, cv2.LINE_AA)

            y += 50

        return rgb_image

    def write_image_type_on_image(self, rgb_image, image_data):
        position = (200, 50)
        if image_data['image_data_type'] == self.ImageDateType.QR_COLORS_ONLY:
            self.write_text_on_image(rgb_image=rgb_image, text="image type: QR colors only", position=position,
                                     color=(255, 0, 0))
        elif image_data['image_data_type'] == self.ImageDateType.QR_SHAPES_ONLY:
            self.write_text_on_image(rgb_image=rgb_image, text="image type: QR shapes only", position=position,
                                     color=(255, 0, 255))
        elif image_data['image_data_type'] == self.ImageDateType.SHAPES_AND_COLORS:
            self.write_text_on_image(rgb_image=rgb_image, text="image type: shapes and colors", position=position, color=(255, 255, 0))
        elif image_data['image_data_type'] == self.ImageDateType.WORDS_SHAPES_NAMES:
            self.write_text_on_image(rgb_image=rgb_image, text="image type: words - shapes names", position=position, color=(0, 255, 255))
        elif image_data['image_data_type'] == self.ImageDateType.WORDS_COLORS_NAMES:
            self.write_text_on_image(rgb_image=rgb_image, text="image type: words - colors names", position=position, color=(0, 0, 255))
        else:
            self.write_text_on_image(rgb_image=rgb_image, text="image type: no type", position=position, color=(0, 0, 0))
        return rgb_image


    def write_screen_color_on_image(self, rgb_image, screen_data):
        screen_color_data = screen_data['screen_color_data']
        small_box_on_empty_part_of_screen = screen_data['small_box_on_empty_part_of_screen']
        if small_box_on_empty_part_of_screen is None:
            return rgb_image
        box_center = self.get_box_center(small_box_on_empty_part_of_screen)
        screen_color_name = screen_color_data['name']
        screen_color = screen_color_data['rgb_color']
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        thickness = 2
        text_to_write = screen_color_name + ' screen'
        cv2.putText(rgb_image, text_to_write, box_center, font,
                    fontScale, screen_color, thickness, cv2.LINE_AA)
        return rgb_image

    def get_box_center(self, box):
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        x_center = int(0.5 * (x_min + x_max))
        y_center = int(0.5 * (y_min + y_max))
        box_center = np.array((x_center, y_center), int)
        return box_center

    def write_frame_number_on_image(self, img_rgb, frame_index):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 0, 255)
        thickness = 2
        org = (20, 50)
        cv2.putText(img_rgb, f'{frame_index}', org, font, font_scale, color, thickness, cv2.LINE_AA)
        return img_rgb

    def get_all_contours_boxes(self, contours):
        boxes = list()
        local_max_screen_width = 900
        local_min_screen_width = 500
        local_max_screen_height = 500
        local_min_screen_height = 200
        for single_contour in contours:
            x, y, w, h = cv2.boundingRect(single_contour)
            # if local_max_screen_width > w > local_min_screen_width and \
            #         local_max_screen_height > h > local_min_screen_height:
            x1 = x
            x2 = x1 + w
            y1 = y
            y2 = y1 + h
            box = (x1, y1, x2, y2)
            boxes.append(box)
        return boxes

    def get_qr_data(self, rgb_image):
        screen_data = None
        if rgb_image.shape[0] == 0:
            print('EMPTY IMAGE!!!!!')
            return [], [], []
        qr_result, qr_points, _ = self.detect.detectAndDecode(rgb_image)

        if qr_points is not None:
            x_qr_top_left = int(qr_points[0][0][0])
            y_qr_top_left = int(qr_points[0][0][1])

            x_qr_top_right = int(qr_points[0][1][0])
            y_qr_top_right = int(qr_points[0][1][1])

            x_qr_bottom_right = int(qr_points[0][2][0])
            y_qr_bottom_right = int(qr_points[0][2][1])

            x_qr_bottom_left = int(qr_points[0][3][0])
            y_qr_bottom_left = int(qr_points[0][3][1])

            contours = self.get_image_contours(rgb_image)
            screen_box = self.get_screen_box_screen_content(contours, x_qr_top_left, y_qr_top_left, x_qr_bottom_right,
                                                            y_qr_bottom_right)
            screen_center = self.get_screen_center(screen_box)

            if x_qr_top_right <= x_qr_top_left:
                # I think it means that it's not really a qr
                return [], [], []

            np_y_vals = np.array([y_qr_top_left, y_qr_top_right])
            y_min_qr = np.min(np_y_vals)

            delta_y = min(y_min_qr, 40)
            start_x = min([x_qr_top_left, x_qr_top_right])
            start_y = int(y_min_qr - 0.5 * delta_y)
            end_x = max([x_qr_top_left, x_qr_top_right])
            end_y = int(y_min_qr - 0.25 * delta_y)
            if end_x < start_x:
                david = 8
            small_box_on_empty_part_of_screen = np.array((start_x, start_y, end_x, end_y))
            screen_color_data = self.get_screen_color_data(rgb_image, small_box_on_empty_part_of_screen)
            x_screen_point_for_tello = int(0.5 * (start_x + end_x))
            y_screen_point_for_tello = int(0.5 * (start_y + end_y))
            screen_point_for_tello = np.array((x_screen_point_for_tello, y_screen_point_for_tello))

            screen_data = dict()
            screen_data['screen_box'] = screen_box
            screen_data['screen_color_data'] = screen_color_data
            screen_data['small_box_on_empty_part_of_screen'] = small_box_on_empty_part_of_screen
            screen_data['screen_point_for_tello'] = screen_point_for_tello
            screen_data['center_of_screen_box'] = screen_center


        if qr_result == '':
            return [], [], []
        last_char = qr_result[-1]
        if last_char == '\n':
            qr_result = qr_result[0:-1]
        qr_data = qr_result.split('\n')
        return qr_data, qr_points, screen_data

    def calc_intersection(self, list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        sets_intersection = set1 & set2
        lists_intersection = list(sets_intersection)
        return lists_intersection

    def detect_qr_data_type(self, qr_list_data):
        if not qr_list_data:
            return 'no_qr_data'
        shapes_list = ['square', 'circle', 'octagon', 'rhombus', 'triangle', 'rectangle', 'pentagon', 'star']
        colors_list = ['brown', 'yellow', 'orange', 'red', 'purple', 'green', 'blue', 'pink']
        shapes_intersection = self.calc_intersection(shapes_list, qr_list_data)
        colors_intersection = self.calc_intersection(colors_list, qr_list_data)
        if len(shapes_intersection) > 0:
            return 'qr_shapes_only'
        elif len(colors_intersection) > 0:
            return 'qr_colors_only'
        else:
            return None  # error. should never reach here

    def get_screen_color_by_qr_points(self, rgb_image, qr_points):

        qr_x_vals = qr_points[:, :, 0]
        qr_y_vals = qr_points[:, :, 1]
        min_qr_x = np.min(qr_x_vals)
        max_qr_x = np.max(qr_x_vals)
        min_qr_y = np.min(qr_y_vals)
        max_qr_y = np.max(qr_y_vals)

        delta_y = min(min_qr_y, 40)
        end_y = int(min_qr_y - 0.5 * delta_y)
        start_y = int(min_qr_y - delta_y)
        start_x = int(min_qr_x + 0.25 * (max_qr_x - min_qr_x))
        end_x = int(min_qr_x + 0.75 * (max_qr_x - min_qr_x))

        small_box_on_empty_part_of_screen = np.array((start_x, start_y, end_x, end_y))
        # rgb_image = self.draw_small_box_on_empty_part_of_image(rgb_image, small_box_on_empty_part_of_screen)
        # cv2.imshow('rgb_image', rgb_image)
        # cv2.waitKey(0)
        screen_color_data = self.get_screen_color_data(rgb_image, small_box_on_empty_part_of_screen)
        screen_data = {'screen_color_data': screen_color_data,
                       'small_box_on_empty_part_of_screen': small_box_on_empty_part_of_screen, 'screen_box': []}
        return screen_data

    def get_image_qr_data(self, rgb_image):
        qr_data_list, qr_points, screen_data = self.get_qr_data(rgb_image)
        qr_data_type = self.detect_qr_data_type(qr_data_list)
        if qr_data_type == 'no_qr_data':
            screen_data = {'screen_color_data': {'name': None, 'rgb_color': None},
                           'small_box_on_empty_part_of_screen': [], 'screen_box': [], 'screen_point_for_tello': None, 'center_of_screen_box': None}
        # else:
        #     screen_data = self.get_screen_color_by_qr_points(rgb_image, qr_points)
        return qr_data_list, qr_points, qr_data_type, screen_data
