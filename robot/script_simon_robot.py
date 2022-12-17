import time

import cv2
import os
import numpy as np
import glob
from shapedetector_robot import ShapeDetector

# greenLower = (42, 101, 96)
# greenUpper = (83, 168, 164)
#
# redLower = (0, 168, 133)
# redUpper = (4, 204, 161)
#
# blueLower = (100, 165, 110)
# blueUpper = (122, 216, 142)
#
# orangeLower = (5, 196, 156)
# orangeUpper = (30, 237, 188)
#
# purpleLower = (122, 100, 106)
# purpleUpper = (147, 139, 160)

min_blue, min_green, min_red = 165, 132, 140
max_blue, max_green, max_red = 203, 225, 222

green_color_info = {"name": 'green', "lower": (42, 101, 96), "upper": (83, 168, 164), "rgb_color": (0, 255, 0)}
# red_color_info = {"name": 'red', "lower": (0, 168, 133) , "upper": (5, 235, 219), "rgb_color": (0, 0, 255)}
red_color_info = {"name": 'red', "lower": (165, 132, 140), "upper": (203, 225, 222), "rgb_color": (0, 0, 255)}
# red_color_info = {"name": 'red', "lower": (163, 193, 150), "upper": (207, 233, 255), "rgb_color": (0, 0, 255)}
# red_color_info = {"name": 'red', "lower": (158, 62, 98), "upper": (215, 246, 249), "rgb_color": (0, 0, 255)}
blue_color_info = {"name": 'blue', "lower": (89, 129, 123), "upper": (165, 241, 255), "rgb_color": (255, 0, 0)}
orange_color_info = {"name": 'orange', "lower": (6, 58, 135), "upper": (26, 228, 255), "rgb_color": (0, 165, 255)}
purple_color_info = {"name": 'purple', "lower": (81, 78, 106), "upper": (152, 160, 173), "rgb_color": (153, 51, 102)}

colors_ranges_info = [green_color_info, red_color_info, blue_color_info, orange_color_info, purple_color_info]

unknown_color_info = {"name": 'unknown', "lower": (0, 0, 0), "upper": (0, 0, 0), "rgb_color": (0, 0, 0)}

shape_to_color = {"circle": {"color_name": 'red', "color": (0, 0, 255)},
                  "octagon": {"color_name": 'yellow', "color": (0, 255, 255)},
                  "pentagon": {"color_name": 'blue', "color": (255, 0, 0)},
                  "rectangle": {"color_name": 'brown', "color": (33, 67, 101)},
                  "square": {"color_name": 'orange', "color": (0, 165, 255)},
                  "rhombus": {"color_name": 'magenta', "color": (255, 0, 255)},
                  "star": {"color_name": 'green', "color": (0, 255, 0)},
                  "triangle": {"color_name": 'purple', "color": (153, 51, 102)},
                  "start": {"color_name": 'black', "color": (0, 0, 0)}}

#input_file_name = 'real_time_floor_rec2.avi'
#input_file_name = 'robomaster_simon_david_house.mp4'
#input_file_name = 'robomaster_ep_pov.MP4'
input_file_name = 'robot_1.mp4'
# input_file_name = 'tello_pov.mp4'
#input_file_name = 'shapes_on_screen.mp4'
input_file_full_path = f'./input_data_simon/videos_and_images/videos/{input_file_name}'
# vid = cv2.VideoCapture("rtsp://192.168.1.28:8901/live")  # For streaming links
vid = cv2.VideoCapture(input_file_full_path)
#vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)


folder_name = 'robomaster_ep_pov'
#folder_name = 'robot_1'
input_folder_full_path = f'./input_data_simon/videos_and_images/images/' + folder_name
input_file_name = '00076.jpg'
# input_file_name = 'Simon_00001.jpg'
image_full_path = input_folder_full_path + '/' + input_file_name

jpg_files = sorted(glob.glob(input_folder_full_path + '/*.jpg'))
#jpg_files = [image_full_path]
num_of_frames = len(jpg_files)
frame_milliseconds = 1




def count_pixels_in_color_range(colorLower, colorUpper, frame, code=cv2.COLOR_BGR2HSV):
    hsv_frame = cv2.cvtColor(frame, code)
    mask = cv2.inRange(hsv_frame, colorLower, colorUpper)
    num_of_pixels = cv2.countNonZero(mask)
    return num_of_pixels


def calc_dist_from_point_to_rectangular_box(point, np_lower, np_upper):
    x_min = np_lower[0]
    x_max = np_upper[0]
    y_min = np_lower[1]
    y_max = np_upper[1]
    z_min = np_lower[2]
    z_max = np_upper[2]
    x = point[0]
    y = point[1]
    z = point[2]
    x_closest_point_on_box = np.clip(x, x_min, x_max)
    y_closest_point_on_box = np.clip(y, y_min, y_max)
    z_closest_point_on_box = np.clip(z, z_min, z_max)
    closest_point_on_box = np.array((x_closest_point_on_box, y_closest_point_on_box, z_closest_point_on_box))
    dist = np.linalg.norm(point - closest_point_on_box)
    return dist


def detect_shape_data(frame, sub_image):
    colorCode = cv2.COLOR_BGR2HSV
    num_of_colors = len(colors_ranges_info)
    colors_num_of_pixels = np.zeros([num_of_colors])
    for index, single_color_range_info in enumerate(colors_ranges_info):
        lower = single_color_range_info["lower"]
        upper = single_color_range_info["upper"]
        single_color_num_of_pixels = count_pixels_in_color_range(lower, upper, frame, colorCode)
        colors_num_of_pixels[index] = single_color_num_of_pixels
    max_index = colors_num_of_pixels.argmax()
    shape_data = colors_ranges_info[max_index]
    if np.sum(colors_num_of_pixels) == 0:
        # distances = np.zeros([num_of_colors])
        # hsv_sub_image = cv2.cvtColor(sub_image, colorCode)
        # average_pixel = np.mean(hsv_sub_image, axis=(0, 1))
        # for index, single_color_range_info in enumerate(colors_ranges_info):
        #     np_lower = np.array(single_color_range_info["lower"])
        #     np_upper = np.array(single_color_range_info["upper"])
        #     dist_from_point_to_rectangular_box = calc_dist_from_point_to_rectangular_box(average_pixel, np_lower, np_upper)
        #     distances[index] = dist_from_point_to_rectangular_box
        # min_index = distances.argmin()
        # shape_data = colors_ranges_info[min_index]
        return unknown_color_info
    return shape_data

def get_shape_in_the_middle_of_the_frame(rgb_frame, shapes_numpy_locations, ratio_x):
    [frame_height, frame_width, channels] = rgb_frame.shape
    frame_middle_x = int(0.5 * frame_width)
    frame_middle_y = int(0.5 * frame_height)



    image_middle_x = int(0.5 * frame_width)
    start_point_x_left_line = int(image_middle_x - ratio_x * frame_width)
    start_point_left_line = (start_point_x_left_line, 0)
    end_point_left_line = (start_point_x_left_line, frame_height)

    start_point_x_right_line = int(image_middle_x + ratio_x * frame_width)
    start_point_right_line = (start_point_x_right_line, 0)
    end_point_right_line = (start_point_x_right_line, frame_height)


    num_of_shapes = shapes_numpy_locations.shape[0]
    x_middles = np.zeros(num_of_shapes)
    y_middles = np.zeros(num_of_shapes)
    for shape_index in range(0, num_of_shapes):
        current_shape_data = shapes_numpy_locations[shape_index]
        x_min = current_shape_data[0]
        y_min = current_shape_data[1]
        x_max = current_shape_data[2]
        y_max = current_shape_data[3]
        # shape_width = x_max - x_min
        # shape_height = y_max - y_min
        shape_name = current_shape_data[6]
        if (x_min <= start_point_x_left_line and x_max >= start_point_x_right_line) or \
            (x_min >= start_point_x_left_line and  x_min <= start_point_x_right_line) or \
                (x_max >= start_point_x_left_line and  x_max <= start_point_x_right_line):
            return shape_name




# def detect_shapes(rgb_frame, frame_index):
#     [frame_height, frame_width, channels] = rgb_frame.shape
#     gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
#     results = model(gray_frame)
#
#     frame_pandas_resutls = results.pandas()
#     shapes_dataframes = frame_pandas_resutls.xyxy[0]
#     shapes_numpy_locations = shapes_dataframes.to_numpy()
#     num_of_shapes = shapes_numpy_locations.shape[0]
#     x_ratio = 0.2
#     y_ratio = 0.2
#     print(f'frame_index = {frame_index}')
#
#
#
#
#
#
#     frame_with_shapes_data = rgb_frame.copy()
#     for shape_index in range(0, num_of_shapes):
#         mask_RGB = np.zeros([frame_height, frame_width, channels], dtype=rgb_frame.dtype)
#         current_shape_data = shapes_numpy_locations[shape_index]
#         x_min = current_shape_data[0]
#         y_min = current_shape_data[1]
#         x_max = current_shape_data[2]
#         y_max = current_shape_data[3]
#
#
#
#
#
#         shape_name = current_shape_data[6]
#         start_point = (int(x_min), int(y_min))
#         end_point = (int(x_max), int(y_max))
#         x_middle = int(0.5 * (x_min + x_max))
#         y_middle = int(0.5 * (y_min + y_max))
#         shape_width = x_max - x_min
#         shape_height = y_max - y_min
#
#
#
#         x_start = int(x_min + x_ratio * shape_width)
#         x_end = int(x_min + (1 - x_ratio) * shape_width)
#
#         y_start = int(y_min + y_ratio * shape_height)
#         y_end = int(y_min + (1 - y_ratio) * shape_height)
#
#         sub_image = rgb_frame[y_start: y_end, x_start: x_end, :]
#
#         mask_RGB[y_start: y_end, x_start: x_end, :] = sub_image
#
#         # if frame_index == 260:
#         #     cv2.imshow('mask_RGB', mask_RGB)
#         #     key = cv2.waitKey(0) & 0xFF
#
#         # shape_data = detect_shape_data(mask_RGB, sub_image)
#         # shape_color_name = shape_data["name"]
#         # shape_color = shape_data["rgb_color"]
#
#         shape_color = shape_to_color[shape_name]["color"]
#
#         thickness = 2
#         frame_with_shapes_data = cv2.rectangle(frame_with_shapes_data, start_point, end_point, shape_color, thickness)
#
#
#
#
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         shapeFontScale = 0.8
#         shapeThickness = 2
#         center = (x_middle, y_middle)
#         cv2.putText(frame_with_shapes_data, f'{shape_name}', center, font,
#                     shapeFontScale, shape_color, shapeThickness, cv2.LINE_AA)
#
#         confidence = current_shape_data[4]
#         obj_class = current_shape_data[5]
#         obj_name = current_shape_data[6]
#         david5 = 5
#     # cv2.imshow('Video Live IP cam', results.render()[0])
#
#
#
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 1
#     color = (255, 0, 255)
#     thickness = 2
#     org = (20, 50)
#     cv2.putText(frame_with_shapes_data, f'{frame_index}', org, font, fontScale, color, thickness, cv2.LINE_AA)
#
#
#     ratio_x = 0.1
#     image_width = frame_with_shapes_data.shape[1]
#     image_height = frame_with_shapes_data.shape[0]
#     image_middle_x = int(0.5 * image_width)
#     start_point_x_left_line = int(image_middle_x - ratio_x * image_width)
#     start_point_left_line = (start_point_x_left_line, 0)
#     end_point_left_line = (start_point_x_left_line, image_height)
#
#     start_point_x_right_line = int(image_middle_x + ratio_x * image_width)
#     start_point_right_line = (start_point_x_right_line, 0)
#     end_point_right_line = (start_point_x_right_line, image_height)
#
#     start_point = (image_middle_x, 0)
#     end_point = (image_middle_x, image_height)
#     line_color = (255, 0, 0)
#     line_thickness = 3
#     cv2.line(frame_with_shapes_data, start_point, end_point, line_color, line_thickness)
#     cv2.line(frame_with_shapes_data, start_point_left_line, end_point_left_line, line_color, line_thickness)
#     cv2.line(frame_with_shapes_data, start_point_right_line, end_point_right_line, line_color, line_thickness)
#
#     shape_name = get_shape_in_the_middle_of_the_frame(rgb_frame, shapes_numpy_locations, ratio_x)
#     if shape_name is None:
#         shape_color = (0, 0, 0)
#     else:
#         shape_color = shape_to_color[shape_name]["color"]
#     shape_name_loc = (100, 50)
#     cv2.putText(frame_with_shapes_data, f'{shape_name}', shape_name_loc, font, fontScale, shape_color, thickness, cv2.LINE_AA)
#
#     return frame_with_shapes_data

# model.cuda()

simon_images_output_folder = './simon_images_with_data'
isExist = os.path.exists(simon_images_output_folder)
if not isExist:
    os.makedirs(simon_images_output_folder)
else:
    files = glob.glob(simon_images_output_folder + '/*.jpg')
    for f in files:
        os.remove(f)

sd = ShapeDetector()
for frame_index, jpg_file in enumerate(jpg_files):
    rgb_image = cv2.imread(jpg_file)
    if rgb_image is None:
        break
    image_data = sd.get_image_data_from_frame(rgb_image)



    rgb_image_with_shapes_data, image_data = sd.detect_shapes(rgb_image, frame_index)
    file_full_path = "{}/{:05d}.jpg".format(simon_images_output_folder, frame_index)
    cv2.imwrite(file_full_path, rgb_image_with_shapes_data)

    cv2.imshow('rgb_image_with_shapes_data', rgb_image_with_shapes_data)
    # print(results.pandas().xyxy[0])
    key = cv2.waitKey(frame_milliseconds) & 0xFF
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

# from threading import Thread
# import cv2, time

# class ThreadedCamera(object):
#     def __init__(self, src=0):
#         self.capture = cv2.VideoCapture(src)
#         self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

#         # FPS = 1/X
#         # X = desired FPS
#         self.FPS = 1/30
#         self.FPS_MS = int(self.FPS * 1000)

#         # Start frame retrieval thread
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()

#     def update(self):
#         while True:
#             if self.capture.isOpened():
#                 (self.status, self.frame) = self.capture.read()
#             time.sleep(self.FPS)

#     def show_frame(self):
#         results = model(self.frame)
#         results.xy
#         cv2.imshow('Video Live IP cam',results.render()[0])
#         # cv2.imshow('frame', self.frame)
#         key = cv2.waitKey(self.FPS_MS) & 0xFF
#         if key ==ord('q'):
#             return


# if __name__ == '__main__':
#     src = 'rtsp://192.168.1.28:8901/live'
#     threaded_camera = ThreadedCamera(src)
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device="cpu")
#     model.conf = 0.5
#     while True:
#         try:
#             threaded_camera.show_frame()
#         except AttributeError:
#             pass
