import glob
import cv2

def create_video(frames_path, frame_extension, video_path, frame_rate):
    frames_files_full_paths = glob.glob(f'{frames_path}/*.{frame_extension}')
    frames_files_full_paths = sorted(frames_files_full_paths)
    #frames_files_full_paths = frames_files_full_paths[1200:]
    first_frame_full_path = frames_files_full_paths[0]
    first_frame = cv2.imread(first_frame_full_path)
    first_frame_shape = first_frame.shape
    height = first_frame_shape[0]
    width = first_frame_shape[1]
    frameSize = (width, height)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, frameSize)
    num_of_frames = len(frames_files_full_paths)
    for idx, filename in enumerate(frames_files_full_paths):
        print(f'creating video: frame {idx + 1} out of {num_of_frames}')
        img = cv2.imread(filename)
        out.write(img)
    out.release()

frame_rate = 5
video_path = './drone/output/videos/012_mix.avi'
images_output_folder = './drone/output/images/012_mix/'
create_video(images_output_folder, 'jpg', video_path, frame_rate)