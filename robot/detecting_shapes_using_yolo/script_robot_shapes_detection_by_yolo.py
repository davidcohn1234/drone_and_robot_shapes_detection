import cv2
import os
from shapedetector_robot_yolo import ShapeDetector

def main():
    input_file_name = 'robot_2.mp4'
    input_file_full_path = f'../input_data/videos/{input_file_name}'
    vid = cv2.VideoCapture(input_file_full_path)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    frame_index = 0
    images_output_folder = './simon_images_with_data'
    isExist = os.path.exists(images_output_folder)
    if not isExist:
        os.makedirs(images_output_folder)

    sd = ShapeDetector()
    while True:
        frame_index += 1
        _, rgb_image = vid.read()
        if rgb_image is None:
            break
        frame_with_shapes_data = sd.detect_shapes(rgb_image, frame_index)
        file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index)
        cv2.imwrite(file_full_path, frame_with_shapes_data)
        cv2.imshow('frame_with_shapes_data', frame_with_shapes_data)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

main()