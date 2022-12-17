import cv2
import os
import glob
from shapedetector_robot import ShapeDetector

def create_empty_output_folder(images_output_folder):
    isExist = os.path.exists(images_output_folder)
    if not isExist:
        os.makedirs(images_output_folder)
    else:
        files = glob.glob(images_output_folder + '/*.jpg')
        for f in files:
            os.remove(f)

def main():
    folder_name = 'robomaster_ep_pov'
    input_folder_full_path = f'./input_data/' + folder_name
    jpg_files = sorted(glob.glob(input_folder_full_path + '/*.jpg'))
    frame_milliseconds = 1
    images_output_folder = './images_with_data'
    create_empty_output_folder(images_output_folder)

    sd = ShapeDetector()
    for frame_index, jpg_file in enumerate(jpg_files):
        rgb_image = cv2.imread(jpg_file)
        if rgb_image is None:
            break
        rgb_image_with_shapes_data, image_data = sd.detect_shapes(rgb_image, frame_index)
        file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index)
        cv2.imwrite(file_full_path, rgb_image_with_shapes_data)
        cv2.imshow('rgb_image_with_shapes_data', rgb_image_with_shapes_data)
        key = cv2.waitKey(frame_milliseconds) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

main()
