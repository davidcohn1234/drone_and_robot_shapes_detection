import cv2
import os
import glob
from enum import Enum
from shapedetector_robot_yolo import ShapeDetectorYolo
from shapedetector_robot_contours import ShapeDetectorContours
import common_utils


class ShapeDetectionType(Enum):
    DETECT_SHAPE_USING_YOLO = 0
    DETECT_SHAPE_USING_CONTOURS = 1

def create_empty_output_folder(images_output_folder):
    isExist = os.path.exists(images_output_folder)
    if not isExist:
        os.makedirs(images_output_folder)
    else:
        files = glob.glob(images_output_folder + '/*.jpg')
        for f in files:
            os.remove(f)

def detect_shapes_on_frames_from_folder(shape_detection_type, folder_name):
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
    main_images_output_folder = main_output_folder + '/' + 'images'
    videos_output_folder = main_output_folder + '/' + 'videos'
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
            break
    cv2.destroyAllWindows()
    print(f'Finished detecting shapes on folder {input_folder_full_path}')

    frame_rate = 5
    video_path = videos_output_folder + '/' + folder_name + '.avi'
    print(f'Creating video {video_path}')
    common_utils.create_video(images_output_folder, 'jpg', video_path, frame_rate)
    print(f'Finised creating video {video_path}')

def main():
    folder_name = 'robomaster_ep_pov'
    shape_detection_type = ShapeDetectionType.DETECT_SHAPE_USING_CONTOURS
    detect_shapes_on_frames_from_folder(shape_detection_type, folder_name)

main()
