import cv2
import numpy as np
import torch
import common_utils

class ShapeDetectorYolo:
    def __init__(self):
        self.brown_color_data = {"name": 'brown', "lower": (0, 10, 33), "upper": (15, 162, 131),
                                                     "rgb_color": (19, 69, 139)}
        self.yellow_color_data = {"name": 'yellow', "lower": (24, 64, 85), "upper": (46, 255, 187),
                                                      "rgb_color": (0, 255, 255)}
        self.orange_color_data = {"name": 'orange', "lower": (11, 161, 119), "upper": (21, 255, 154),
                                                      "rgb_color": (0, 165, 255)}
        self.red_color_data = {"name": 'red', "lower": (0, 91, 84), "upper": (11, 235, 151),
                                                   "rgb_color": (0, 0, 255)}
        self.purple_color_data = {"name": 'purple', "lower": (120, 59, 65),
                                                      "upper": (140, 191, 180), "rgb_color": (200, 0, 119)}
        self.green_color_data = {"name": 'green', "lower": (38, 81, 43), "upper": (99, 255, 93),
                                                     "rgb_color": (0, 255, 0)}
        self.blue_color_data = {"name": 'blue', "lower": (40, 91, 81), "upper": (120, 255, 248),
                                                    "rgb_color": (180, 10, 0)}
        self.pink_color_data = {"name": 'pink', "lower": (141, 61, 120),
                                                       "upper": (181, 195, 206), "rgb_color": (255, 0, 255)}

        self.colors_ranges_data = [self.brown_color_data,
                                   self.yellow_color_data,
                                   self.orange_color_data,
                                   self.red_color_data,
                                   self.purple_color_data,
                                   self.green_color_data,
                                   self.blue_color_data,
                                   self.pink_color_data]
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights_simon_grayscale.pt')
        self.model.conf = 0.8
        self.zip_folder = '.'
        common_utils.download_input_images_from_google_drive(zip_folder=self.zip_folder,
                                                             zip_file_id='1dimHaktpjQSFCEgG3S29L_5tkH82aSN3')
        common_utils.extract_frames_from_videos(self.zip_folder + '/' + 'input_data')


    def detect_shapes(self, rgb_frame, frame_index):
        [frame_height, frame_width, channels] = rgb_frame.shape
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        results = self.model(gray_frame)

        frame_pandas_resutls = results.pandas()
        shapes_dataframes = frame_pandas_resutls.xyxy[0]
        shapes_numpy_locations = shapes_dataframes.to_numpy()
        num_of_shapes = shapes_numpy_locations.shape[0]
        x_ratio = 0.0
        y_ratio = 0.0

        frame_with_shapes_data = rgb_frame.copy()
        for shape_index in range(0, num_of_shapes):
            mask_RGB = np.zeros([frame_height, frame_width, channels], dtype=rgb_frame.dtype)
            current_shape_data = shapes_numpy_locations[shape_index]
            x_min = current_shape_data[0]
            y_min = current_shape_data[1]
            x_max = current_shape_data[2]
            y_max = current_shape_data[3]

            # x_min = int(shape_box[0])
            # y_min = int(shape_box[1])
            # x_max = int(shape_box[2])
            # y_max = int(shape_box[3])

            shape_name = current_shape_data[6]
            start_point = (int(x_min), int(y_min))
            end_point = (int(x_max), int(y_max))
            x_middle = int(0.5 * (x_min + x_max))
            y_middle = int(0.5 * (y_min + y_max))
            shape_width = x_max - x_min
            shape_height = y_max - y_min

            x_start = int(x_min + x_ratio * shape_width)
            x_end = int(x_min + (1 - x_ratio) * shape_width)

            y_start = int(y_min + y_ratio * shape_height)
            y_end = int(y_min + (1 - y_ratio) * shape_height)

            sub_image = rgb_frame[y_start: y_end, x_start: x_end, :]

            mask_RGB[y_start: y_end, x_start: x_end, :] = sub_image

            small_shape_box_for_color_detection = self.get_partial_box(current_shape_data, ratio=0.45)
            shape_color_data = self.get_color_data_in_box(rgb_frame, small_shape_box_for_color_detection)

            shape_color = shape_color_data['rgb_color']

            thickness = 2
            frame_with_shapes_data = cv2.rectangle(frame_with_shapes_data, start_point, end_point, shape_color,
                                                   thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            shapeFontScale = 0.8
            shapeThickness = 2
            center = (x_middle, y_middle)
            cv2.putText(frame_with_shapes_data, f'{shape_name}', center, font,
                        shapeFontScale, shape_color, shapeThickness, cv2.LINE_AA)

            confidence = current_shape_data[4]
            obj_class = current_shape_data[5]
            obj_name = current_shape_data[6]

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 255)
        thickness = 2
        org = (20, 50)
        cv2.putText(frame_with_shapes_data, f'{frame_index}', org, font, fontScale, color, thickness, cv2.LINE_AA)

        ratio_x = 0.1
        image_width = frame_with_shapes_data.shape[1]
        image_height = frame_with_shapes_data.shape[0]
        image_middle_x = int(0.5 * image_width)
        start_point_x_left_line = int(image_middle_x - ratio_x * image_width)
        start_point_left_line = (start_point_x_left_line, 0)
        end_point_left_line = (start_point_x_left_line, image_height)

        start_point_x_right_line = int(image_middle_x + ratio_x * image_width)
        start_point_right_line = (start_point_x_right_line, 0)
        end_point_right_line = (start_point_x_right_line, image_height)

        start_point = (image_middle_x, 0)
        end_point = (image_middle_x, image_height)
        line_color = (255, 0, 0)
        line_thickness = 3
        #cv2.line(frame_with_shapes_data, start_point, end_point, line_color, line_thickness)
        cv2.line(frame_with_shapes_data, start_point_left_line, end_point_left_line, line_color, line_thickness)
        cv2.line(frame_with_shapes_data, start_point_right_line, end_point_right_line, line_color, line_thickness)

        if shapes_numpy_locations.shape[0] > 0:
            shape_name, shape_color_data = self.get_shape_in_the_middle_of_the_frame(rgb_frame, shapes_numpy_locations, ratio_x)
            if shape_color_data is not None:
                shape_color = shape_color_data['rgb_color']
                shape_name_loc = (150, 50)
                cv2.putText(frame_with_shapes_data, f'{shape_name}', shape_name_loc, font, fontScale, shape_color, thickness,
                            cv2.LINE_AA)

        return frame_with_shapes_data

    def create_template_gray_images(self, shapes_types):
        template_images = {}
        for shape_index, shape_name in enumerate(shapes_types):
            template_full_path = '../../template_shapes/' + shape_name + '.jpg'
            template_rgb_image = cv2.imread(template_full_path)
            template_gray_image = cv2.cvtColor(template_rgb_image, cv2.COLOR_BGR2GRAY)
            # template_gray_image_without_border = self.get_image_without_border(template_gray_image)
            template_images[shape_name] = template_gray_image
        return template_images

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

    def get_color_data_in_box(self, rgb_image, box):
        color_data = {'name': None, 'rgb_color': (0, 0, 0)}
        rgb_sub_image = self.get_sub_image_by_box(rgb_image, box)
        color_code = cv2.COLOR_BGR2HSV
        colors_ranges_data = self.colors_ranges_data
        num_of_colors = len(colors_ranges_data)
        colors_num_of_pixels = np.zeros([num_of_colors])
        for index, single_color_range_info in enumerate(colors_ranges_data):
            lower = single_color_range_info["lower"]
            upper = single_color_range_info["upper"]
            single_color_num_of_pixels = self.count_pixels_in_color_range(lower, upper, rgb_sub_image, color_code)
            colors_num_of_pixels[index] = single_color_num_of_pixels
        max_index = colors_num_of_pixels.argmax()
        shape_color_data = colors_ranges_data[max_index]
        if np.sum(colors_num_of_pixels) == 0:
            return color_data
        color_data['name'] = shape_color_data['name']
        color_data['rgb_color'] = shape_color_data['rgb_color']
        return color_data

    def get_shape_in_the_middle_of_the_frame(self, rgb_frame, shapes_numpy_locations, ratio_x):
        frame_width = rgb_frame.shape[1]
        image_middle_x = int(0.5 * frame_width)
        start_point_x_left_line = int(image_middle_x - ratio_x * frame_width)
        start_point_x_right_line = int(image_middle_x + ratio_x * frame_width)
        num_of_shapes = shapes_numpy_locations.shape[0]
        for shape_index in range(0, num_of_shapes):
            current_shape_data = shapes_numpy_locations[shape_index]
            x_min = current_shape_data[0]
            y_min = current_shape_data[1]
            x_max = current_shape_data[2]
            y_max = current_shape_data[3]
            shape_name = current_shape_data[6]
            small_shape_box_for_color_detection = self.get_partial_box(current_shape_data, ratio=0.45)
            shape_color_data = self.get_color_data_in_box(rgb_frame, small_shape_box_for_color_detection)
            if (x_min <= start_point_x_left_line and x_max >= start_point_x_right_line) or \
                    (x_min >= start_point_x_left_line and x_min <= start_point_x_right_line) or \
                    (x_max >= start_point_x_left_line and x_max <= start_point_x_right_line):
                return shape_name, shape_color_data
        return None, None

    def get_sub_image_by_box(self, rgb_image, box):
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        rgb_sub_image = rgb_image[y_min:y_max+1, x_min:x_max+1, :]
        return rgb_sub_image

    def count_pixels_in_color_range(self, colorLower, colorUpper, frame, code=cv2.COLOR_BGR2HSV):
        hsv_frame = cv2.cvtColor(frame, code)
        mask = cv2.inRange(hsv_frame, colorLower, colorUpper)
        num_of_pixels = cv2.countNonZero(mask)
        return num_of_pixels