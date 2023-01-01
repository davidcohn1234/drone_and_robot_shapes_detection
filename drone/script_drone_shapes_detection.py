import cv2
import glob
from shapedetector_tello import ShapeDetector
import os
import sys
sys.path.append('../')
import common_utils
from moviepy.editor import VideoFileClip
import argparse

debug_mode = False


def save_images_to_folders(frame_index, images, main_output_folder, folder_name):
    num_of_images = len(images)
    debugging_images_folder_name = 'debugging_images'
    for image_index in range(num_of_images):
        current_image = images[image_index][0]
        current_folder_name = images[image_index][1]
        images_output_folder = main_output_folder + '/' + debugging_images_folder_name + '/' + folder_name + '/' + current_folder_name
        isExist = os.path.exists(images_output_folder)
        if not isExist:
            os.makedirs(images_output_folder)
        file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index+1)
        cv2.imwrite(file_full_path, current_image)



def create_empty_output_folder(images_output_folder):
    isExist = os.path.exists(images_output_folder)
    if not isExist:
        os.makedirs(images_output_folder)
    else:
        files = glob.glob(images_output_folder + '/*.jpg')
        for f in files:
            os.remove(f)

def detect_shapes_on_frames_from_folder(folder_name, create_gif_video, main_output_folder):
    sd = ShapeDetector()
    if not sd.tesseract_exists:
        print('You need to install tesseract!')
        return
    input_folder_full_path = f'./input_data/' + folder_name
    input_file_name = '00277.jpg'
    image_full_path = input_folder_full_path + '/' + input_file_name
    jpg_files = sorted(glob.glob(input_folder_full_path + '/*.jpg'))
    #jpg_files = [image_full_path]
    frame_milliseconds = 1


    main_images_output_folder = main_output_folder + '/' + 'images'
    videos_output_folder = main_output_folder + '/' + 'videos'
    images_output_folder = main_images_output_folder + '/' + folder_name
    create_empty_output_folder(images_output_folder)
    create_empty_output_folder(videos_output_folder)

    for frame_index, jpg_file in enumerate(jpg_files):
        rgb_image = cv2.imread(jpg_file)
        image_data = sd.get_image_data_from_frame(rgb_image)
        images = sd.get_images_for_debugging(rgb_image)
        if debug_mode:
            save_images_to_folders(frame_index, images, main_output_folder, folder_name)
        shapes_data = image_data['shapes_data']
        shapes_boxes = [single_shape_data['shape_box'] for single_shape_data in shapes_data]
        small_shapes_boxes_for_color_detection = [single_shape_data['small_shape_box_for_color_detection'] for single_shape_data in shapes_data]

        screen_data = image_data['screen_data']
        screen_color_data = image_data['screen_data']['screen_color_data']
        screen_box = image_data['screen_data']['screen_box']
        screen_center = image_data['screen_data']['center_of_screen_box']
        small_box_on_empty_part_of_screen = image_data['screen_data']['small_box_on_empty_part_of_screen']

        rgb_image = sd.write_frame_number_on_image(rgb_image, frame_index + 1)
        # if image_data['image_data_type'] == sd.ImageDateType.QR_COLORS_ONLY or \
        #         image_data['image_data_type'] == sd.ImageDateType.QR_SHAPES_ONLY:
        rgb_image = sd.draw_screen_box_on_image(rgb_image, screen_box)

        if image_data['image_data_type'] == sd.ImageDateType.SHAPES_AND_COLORS:
            rgb_image = sd.draw_shapes_boxes_on_image(rgb_image, shapes_boxes)
            #rgb_image = sd.draw_shapes_boxes_on_image(rgb_image, small_shapes_boxes_for_color_detection)
            rgb_image = sd.write_shapes_names_on_image(rgb_image, shapes_data)
        rgb_image = sd.write_shapes_names_on_image_by_their_order(rgb_image, shapes_data)
        screen_point_for_tello = screen_data['screen_point_for_tello']
        # if screen_point_for_tello is not None:
        #     rgb_image = sd.draw_circle_on_image(rgb_image, screen_point_for_tello, circle_color=(255, 0, 0), radius=10)
        # if screen_center is not None:
        #     rgb_image = sd.draw_circle_on_image(rgb_image, screen_center, circle_color=(0, 0, 255), radius=10)

        rgb_image = sd.draw_small_box_on_empty_part_of_image(rgb_image, small_box_on_empty_part_of_screen)
        rgb_image = sd.write_screen_color_on_image(rgb_image, screen_data)
        #rgb_image = draw_boxes_on_image(rgb_image, shapes_boxes, shape_box_color, shape_box_thickness)
        rgb_image = sd.write_image_type_on_image(rgb_image, image_data)

        file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index+1)
        cv2.imwrite(file_full_path, rgb_image)
        cv2.imshow('rgb_image', rgb_image)
        key = cv2.waitKey(frame_milliseconds) & 0xFF
        if key == ord('q'):
            return

    frame_rate = 20
    video_path = videos_output_folder + '/' + folder_name + '.avi'
    print(f'Creating video {video_path}')
    common_utils.create_video(images_output_folder, 'jpg', video_path, frame_rate)
    print(f'Finised creating video {video_path}')

    if create_gif_video:
        gif_video_path = videos_output_folder + '/' + folder_name + '.gif'
        print(f'Creating gif video {gif_video_path}')
        videoClip = VideoFileClip(video_path)
        videoClip.write_gif(gif_video_path)
        print(f'Finished creating gif video {gif_video_path}')

def parse_command_line():
    arg_parser = argparse.ArgumentParser(description="parameters options")

    arg_parser.add_argument('-ifp', '--input_folder_path',
                            default='012_mix',
                            nargs='?',
                            help='Path to input folder of images',
                            required=False
                            )
    args = arg_parser.parse_args()
    return args

def print_args():
    args = parse_command_line()
    for arg in vars(args):
        print(arg, "= ", getattr(args, arg))

def main():
    args = parse_command_line()
    print_args()
    folder_name = args.input_folder_path
    create_gif_video = False
    main_output_folder = './output'
    detect_shapes_on_frames_from_folder(folder_name, create_gif_video, main_output_folder)

main()
