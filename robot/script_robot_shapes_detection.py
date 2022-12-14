import cv2
import os
import glob
from shapedetector_robot_yolo import ShapeDetectorYolo
from shapedetector_robot_contours import ShapeDetectorContours
import sys
sys.path.append('../')
import common_utils
from moviepy.editor import VideoFileClip
import argparse

class ShapeDetectionType(object):
    DETECT_SHAPE_USING_YOLO = "yolo"
    DETECT_SHAPE_USING_CONTOURS = "contours"

def create_empty_output_folder(images_output_folder):
    isExist = os.path.exists(images_output_folder)
    if not isExist:
        os.makedirs(images_output_folder)
    else:
        files = glob.glob(images_output_folder + '/*.jpg')
        for f in files:
            os.remove(f)

def detect_shapes_on_frames_from_folder(shape_detection_type, folder_name, create_gif_video):
    if shape_detection_type == ShapeDetectionType.DETECT_SHAPE_USING_CONTOURS:
        sd = ShapeDetectorContours()
    else:
        sd = ShapeDetectorYolo()
    input_folder_full_path = f'./input_data/images/' + folder_name
    print()
    print(f'Detecting shapes on folder {input_folder_full_path}')
    jpg_files = sorted(glob.glob(input_folder_full_path + '/*.jpg'))
    frame_milliseconds = 1
    main_output_folder = './output'
    if shape_detection_type == ShapeDetectionType.DETECT_SHAPE_USING_CONTOURS:
        main_output_folder_with_detection_type = main_output_folder + '/contours'
    else:
        main_output_folder_with_detection_type = main_output_folder + '/yolo'
    main_images_output_folder = main_output_folder_with_detection_type + '/' + 'images'
    videos_output_folder = main_output_folder_with_detection_type + '/' + 'videos'
    images_output_folder = main_images_output_folder + '/' + folder_name
    create_empty_output_folder(images_output_folder)
    create_empty_output_folder(videos_output_folder)


    for frame_index, jpg_file in enumerate(jpg_files):
        rgb_image = cv2.imread(jpg_file)
        if rgb_image is None:
            break
        rgb_image_with_shapes_data = sd.detect_shapes(rgb_image, frame_index)
        file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index)
        cv2.imwrite(file_full_path, rgb_image_with_shapes_data)
        cv2.imshow('rgb_image_with_shapes_data', rgb_image_with_shapes_data)
        key = cv2.waitKey(frame_milliseconds) & 0xFF
        if key == ord('q'):
            return
    cv2.destroyAllWindows()
    print(f'Finished detecting shapes on folder {input_folder_full_path}')

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
                            default='robomaster_ep_pov',
                            nargs='?',
                            help='Path to input folder of images',
                            required=False
                            )
    arg_parser.add_argument('-sdt', '--shape_detection_type',
                            default=ShapeDetectionType.DETECT_SHAPE_USING_CONTOURS,
                            nargs='?',
                            choices=[ShapeDetectionType.DETECT_SHAPE_USING_CONTOURS,
                                     ShapeDetectionType.DETECT_SHAPE_USING_YOLO],
                            help='Algorithm for shape detection (contours/yolo). default set to {0}'.format(ShapeDetectionType.DETECT_SHAPE_USING_CONTOURS),
                            required=False
                            )
    args = arg_parser.parse_args()
    return args

def print_args():
    args = parse_command_line()
    vars_args = vars(args)
    for arg in vars_args:
        print(arg, "= ", getattr(args, arg))

def main():
    args = parse_command_line()
    print_args()
    folder_name = args.input_folder_path
    shape_detection_type_str = args.shape_detection_type
    create_gif_video = False
    detect_shapes_on_frames_from_folder(shape_detection_type_str, folder_name, create_gif_video)

main()
