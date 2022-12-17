import cv2
import numpy as np
import imutils
import imutils.object_detection
import glob
import operator
from shapedetector_tello import ShapeDetector
import os
from pytesseract import pytesseract




# def draw_boxes_on_image(rgb_image, boxes, box_color, box_thickness):
#     for box in boxes:
#         x1 = int(box[0])
#         y1 = int(box[1])
#         x2 = int(box[2])
#         y2 = int(box[3])
#         cv2.rectangle(rgb_image, (x1, y1), (x2, y2), box_color, box_thickness)
#     return rgb_image

# def write_frame_number_on_image(img_rgb, frame_index, num_of_frames):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     color = (255, 0, 255)
#     thickness = 2
#     org = (20, 50)
#     cv2.putText(img_rgb, f'{frame_index}/{num_of_frames}', org, font, font_scale, color, thickness, cv2.LINE_AA)
#     return img_rgb


def detect_shapes_in_boxes(gray_image, boxes):
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        shape_width = x2 - x1
        shape_height = y2 - y1

        image_with_single_shape = gray_image[y1:y2, x1:x2]
        cv2.imshow('image_with_single_shape', image_with_single_shape)
        cv2.waitKey(0)



# def check_if_box_inside_screen(shape_box, screen_box):
#     x1_screen = screen_box[0]
#     y1_screen = screen_box[1]
#     x2_screen = screen_box[2]
#     y2_screen = screen_box[3]
#     screen_width = x2_screen - x1_screen
#     screen_height = y2_screen - y1_screen
#
#     x1_shape = shape_box[0]
#     y1_shape = shape_box[1]
#     x2_shape = shape_box[2]
#     y2_shape = shape_box[3]
#     shape_width = x2_shape - x1_shape
#     shape_height = y2_shape - y1_shape
#
#     if x1_screen < x1_shape < x2_screen and \
#         x1_screen < x2_shape < x2_screen and \
#         y1_screen < y1_shape < y2_screen and \
#         y1_screen < y2_shape < y2_screen and \
#         shape_width < screen_width and \
#         shape_height < screen_height:
#         return True
#     else:
#         return False

def check_if_box_contains_number(shape_box, gray_image):
    (x1, y1, x2, y2) = shape_box
    gray_image_with_single_shape = gray_image[y1:y2, x1:x2]

    (template_thresh, binary_image_with_single_shape_white_background) = cv2.threshold(
        gray_image_with_single_shape, 128, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_image_with_single_shape = cv2.bitwise_not(binary_image_with_single_shape_white_background)

    text = pytesseract.image_to_string(binary_image_with_single_shape, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    #text = text.lower()
    cv2.imshow('binary_image_with_single_shape', binary_image_with_single_shape)
    cv2.waitKey(0)

# def detect_shapes_boxes_in_screen(contours, screen_box, gray_image):
#     boxes = list()
#     for single_contour in contours:
#         x, y, w, h = cv2.boundingRect(single_contour)
#         diff_width_height = abs(w-h)
#         if max_square_size > w > min_square_size and \
#                 max_square_size > h > min_square_size and \
#                 diff_width_height < 10:
#             x1 = x
#             x2 = x1 + w
#             y1 = y
#             y2 = y1 + h
#             shape_box = (x1, y1, x2, y2)
#             is_box_inside_screen = check_if_box_inside_screen(shape_box, screen_box)
#             if is_box_inside_screen:
#                 #is_number_inside_box = check_if_box_contains_number(shape_box, gray_image)
#                 is_number_inside_box = False
#                 if is_number_inside_box == False:
#                     boxes.append(shape_box)
#
#     boxes = imutils.object_detection.non_max_suppression(np.array(boxes))
#     return boxes

# def get_all_contours_boxes(contours):
#     boxes = list()
#     local_max_screen_width = 900
#     local_min_screen_width = 500
#     local_max_screen_height = 500
#     local_min_screen_height = 200
#     for single_contour in contours:
#         x, y, w, h = cv2.boundingRect(single_contour)
#         if local_max_screen_width > w > local_min_screen_width and \
#                 local_max_screen_height > h > local_min_screen_height:
#             x1 = x
#             x2 = x1 + w
#             y1 = y
#             y2 = y1 + h
#             box = (x1, y1, x2, y2)
#             boxes.append(box)
#     return boxes



# base_model = VGG16(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
#
# def extract(img):
#     img = img.resize((224, 224)) # Resize the image
#     img = img.convert('RGB') # Convert the image color space
#     x = image.img_to_array(img) # Reformat the image
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     feature = model.predict(x)[0] # Extract Features
#     return feature / np.linalg.norm(feature)
#
# def find_closest_match(template_images_folder_path, gray_image_with_single_shape):
#     template_images_paths = sorted(glob.glob(template_images_folder_path + '/*.jpg'))
#     # Iterate through images and extract Features
#     #images = ["img1.png", "img2.png", "img3.png", "img4.png", "img5.png"... + 2000 more]
#     all_features = np.zeros(shape=(len(template_images_paths), 4096))
#
#     for i in range(len(template_images_paths)):
#         image = Image.open(template_images_paths[i])
#         feature = extract(img=image)
#         all_features[i] = np.array(feature)
#
#     # Match image
#     im = Image.fromarray(gray_image_with_single_shape)
#     query = extract(img=im)  # Extract its features
#     dists = np.linalg.norm(all_features - query, axis=1)  # Calculate the similarity (distance) between images
#     ids = np.argsort(dists)[:5]  # Extract 5 images that have lowest distance
#     return ids









def get_image_with_contours(binary_image):
    (height, width) = binary_image.shape
    img_rgb = np.zeros((height, width, 3), np.uint8)
    img_rgb[:, :, 0] = binary_image
    img_rgb[:, :, 1] = binary_image
    img_rgb[:, :, 2] = binary_image
    findContoursResults = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(findContoursResults)
    sd = ShapeDetector()

    cv2.imshow('binary_image', binary_image)
    cv2.waitKey()

    middle_image_pixel = (int(0.5*width), int(0.5*height))



    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.3
    text_color = (0, 0, 0)
    text_thickness = 1

    middle_image_font = cv2.FONT_HERSHEY_COMPLEX
    middle_image_font_scale = 0.6
    middle_image_text_color = (0, 0, 0)
    middle_image_text_thickness = 1

    for single_contour in contours:
        if single_contour.shape[0] <= 2:
            return single_contour.shape[0]

        peri = cv2.arcLength(single_contour, True)
        single_approx_contour = single_contour
        single_approx_contour = cv2.approxPolyDP(single_approx_contour, 0.01 * peri, True)
        single_approx_contour = sd.reduce_sides_of_contour_by_too_big_angles_and_too_little_side_lengths(single_approx_contour)

        contourColor = (0, 0, 255)
        cv2.drawContours(img_rgb, [single_approx_contour], -1, contourColor, 2)

        num_of_contour_points = single_approx_contour.shape[0]
        circle_radius = 3
        circle_color = (0, 255, 0)
        circle_thickness = -1

        cv2.putText(img_rgb,
                      f'{num_of_contour_points} coords', middle_image_pixel, middle_image_font,
                      middle_image_font_scale, middle_image_text_color, middle_image_text_thickness, cv2.LINE_AA)

        for counterPointIndex in range(0, num_of_contour_points):
            current_contour_point = single_approx_contour[counterPointIndex, :]
            current_contour_point_tuple = (current_contour_point[0], current_contour_point[1])
            cv2.circle(img_rgb, current_contour_point_tuple, circle_radius, circle_color, circle_thickness)

            cv2.putText(img_rgb,
                          f'{counterPointIndex+1}', current_contour_point_tuple, font,
                          font_scale, text_color, text_thickness, cv2.LINE_AA)



    return img_rgb


# def check_if_image_contains_one_of_the_shapes(gray_image_with_single_shape, template_gray_images):
#     num_of_template_shapes = len(template_gray_images)
#     max_values = {}
#     sum_vals_all_shapes = {}
#     template_images_folder_path = 'original_shapes_jpg'
#     # ids = find_closest_match(template_images_folder_path, gray_image_with_single_shape)
#
#     sd = ShapeDetector()
#     non_zero_elements_for_all_templates = np.zeros((num_of_template_shapes))
#     first_key = list(template_gray_images.keys())[0]
#     first_template_gray_image = template_gray_images[first_key]
#     template_height, template_width = first_template_gray_image.shape[:2]
#     template_dim = (template_height, template_width)
#
#     resized_gray_image_with_single_shape = cv2.resize(gray_image_with_single_shape, template_dim,
#                                                       interpolation=cv2.INTER_AREA)
#
#     resized_gray_image_with_single_shape_without_border = sd.get_image_without_border(resized_gray_image_with_single_shape)
#
#
#
#     (template_thresh, resized_binary_image_with_single_shape_white_background_without_border) = cv2.threshold(
#         resized_gray_image_with_single_shape_without_border, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     resized_binary_image_with_single_shape_without_border = cv2.bitwise_not(
#         resized_binary_image_with_single_shape_white_background_without_border)
#
#     for template_shape_index, template_image_name in enumerate(template_gray_images):
#         template_gray_image = template_gray_images[template_image_name]
#         (template_thresh, binary_template_white_background) = cv2.threshold(template_gray_image, 128, 255,
#                                                                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         binary_template = cv2.bitwise_not(binary_template_white_background)
#
#         binary_template_without_border = sd.get_image_without_border(binary_template)
#
#         diff_images_without_border = resized_binary_image_with_single_shape_without_border.astype(
#             float) - binary_template_without_border.astype(float)
#         #unique_diff_images = np.unique(diff_images_without_border)
#         abs_diff_images_without_border = abs(diff_images_without_border).astype(np.uint8)
#         #unique_abs_diff_images = np.unique(abs_diff_images_without_border)
#         #sum_vals_specific_shape = np.sum(abs_diff_images_without_border)
#
#         resized_binary_image_with_single_shape_without_border_with_vertices, single_approx_contour = sd.plot_contour_and_vertices_on_image(
#             resized_binary_image_with_single_shape_without_border)
#         binary_template_without_border_with_vertices, _ = sd.plot_contour_and_vertices_on_image(
#             binary_template_without_border)
#
#         non_zero_elements_for_current_template = np.count_nonzero(abs_diff_images_without_border)
#         non_zero_elements_for_all_templates[template_shape_index] = non_zero_elements_for_current_template
#
#         # cv2.imshow('abs_diff_images_without_border', abs_diff_images_without_border)
#         # cv2.imshow('resized_gray_image_with_single_shape_without_border',
#         #            resized_binary_image_with_single_shape_without_border_with_vertices)
#         # cv2.imshow('template_gray_image_without_border', binary_template_without_border_with_vertices)
#         # cv2.waitKey(0)
#     min_non_zero_elements = np.min(non_zero_elements_for_all_templates)
#     if min_non_zero_elements >= 8000:
#         return False
#     else:
#         return True


def get_boxes_max_vals(gray_image_with_single_shape, template_gray_images, dim):
    num_of_template_shapes = len(template_gray_images)
    max_values = {}
    sum_vals_all_shapes = {}
    template_images_folder_path = 'original_shapes_jpg'
    #ids = find_closest_match(template_images_folder_path, gray_image_with_single_shape)
    delta_pixels_to_remove_border = 20

    sd = ShapeDetector()
    non_zero_elements_for_all_templates = np.zeros((num_of_template_shapes))
    for template_shape_index, template_image_name in enumerate(template_gray_images):
        template_gray_image = template_gray_images[template_image_name]
        #resized_template_gray_image = cv2.resize(template_gray_image, dim, interpolation=cv2.INTER_AREA)

        (template_thresh, binary_template_white_background) = cv2.threshold(template_gray_image, 128, 255,
                                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary_template = cv2.bitwise_not(binary_template_white_background)

        template_height, template_width = template_gray_image.shape[:2]
        template_dim = (template_height, template_width)
        resized_gray_image_with_single_shape = cv2.resize(gray_image_with_single_shape, template_dim, interpolation=cv2.INTER_AREA)
        (template_thresh, resized_binary_image_with_single_shape_white_background) = cv2.threshold(resized_gray_image_with_single_shape, 128, 255,
                                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        resized_binary_image_with_single_shape = cv2.bitwise_not(resized_binary_image_with_single_shape_white_background)

        resized_binary_image_with_single_shape_without_border = resized_binary_image_with_single_shape[
                                                                delta_pixels_to_remove_border:template_height-delta_pixels_to_remove_border,
                                                                delta_pixels_to_remove_border:template_width-delta_pixels_to_remove_border]
        binary_template_without_border = binary_template[
                                                                delta_pixels_to_remove_border:template_height-delta_pixels_to_remove_border,
                                                                delta_pixels_to_remove_border:template_width-delta_pixels_to_remove_border]

        diff_images_without_border = resized_binary_image_with_single_shape_without_border.astype(float) - binary_template_without_border.astype(float)
        unique_diff_images = np.unique(diff_images_without_border)
        abs_diff_images_without_border = abs(diff_images_without_border).astype(np.uint8)
        unique_abs_diff_images = np.unique(abs_diff_images_without_border)
        sum_vals_specific_shape = np.sum(abs_diff_images_without_border)

        resized_binary_image_with_single_shape_without_border_with_vertices, single_approx_contour = sd.plot_contour_and_vertices_on_image(resized_binary_image_with_single_shape_without_border)
        binary_template_without_border_with_vertices, _ = sd.plot_contour_and_vertices_on_image(binary_template_without_border)

        M = cv2.moments(single_approx_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])


        non_zero_elements_for_current_template = np.count_nonzero(abs_diff_images_without_border)
        non_zero_elements_for_all_templates[template_shape_index] = non_zero_elements_for_current_template
        print(f'non_zero_elements_for_current_template = {non_zero_elements_for_current_template}')

        cv2.imshow('abs_diff_images_without_border', abs_diff_images_without_border)
        cv2.imshow('resized_gray_image_with_single_shape_without_border', resized_binary_image_with_single_shape_without_border_with_vertices)
        cv2.imshow('template_gray_image_without_border', binary_template_without_border_with_vertices)
        cv2.waitKey(0)

        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                   'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        result_image = cv2.matchTemplate(
            image=resized_binary_image_with_single_shape,
            templ=binary_template,
            method=cv2.TM_CCOEFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_image)
        max_val1 = np.max(result_image)
        max_values[template_image_name] = max_val
        sum_vals_all_shapes[template_image_name] = sum_vals_specific_shape
    min_non_zero_elements = np.min(non_zero_elements_for_all_templates)
    sorted_max_values = dict(sorted(max_values.items(), key=operator.itemgetter(1), reverse=True))
    sorted_sum_vals_all_shapes = dict(sorted(sum_vals_all_shapes.items(), key=operator.itemgetter(1), reverse=False))
    #sorted_max_values = {k: v for k, v in sorted(max_values.items(), key=lambda item: item[1])}
    return sorted_sum_vals_all_shapes

# def get_boxes_of_single_shape_by_matchTemplate(img_gray, resized_template_gray_image, thresh):
#     template_width, template_height = resized_template_gray_image.shape[:2]
#     match = cv2.matchTemplate(
#         image=img_gray,
#         templ=resized_template_gray_image,
#         method=cv2.TM_CCOEFF_NORMED)
#
#     # Select rectangles with
#     # confidence greater than threshold
#     (y_points, x_points) = np.where(match >= thresh)
#     max_val = np.max(match)
#
#     # initialize our list of rectangles
#     boxes = list()
#
#     # loop over the starting (x, y)-coordinates again
#     for (x, y) in zip(x_points, y_points):
#         # update our list of rectangles
#         boxes.append((x, y, x + template_width, y + template_height))
#
#     # apply non-maxima suppression to the rectangles
#     # this will create a single bounding box
#     boxes = imutils.object_detection.non_max_suppression(np.array(boxes))
#     if len(boxes) == 0:
#         return [], None
#
#     num_of_boxes = boxes.shape[0]
#
#     box_centers = np.zeros((num_of_boxes, 2), int)
#     # loop over the final bounding boxes
#     for (box_index, (x_min, y_min, x_max, y_max)) in enumerate(boxes):
#         # draw the bounding box on the image
#         x_center = int(0.5 * (x_min + x_max))
#         y_center = int(0.5 * (y_min + y_max))
#         box_center = np.array((x_center, y_center), int)
#         box_centers[box_index, :] = box_center
#     return boxes, box_centers

def detect_shape_scores(shape_box, gray_image_with_single_shape, template_gray_images, dim):
    sorted_max_values_for_single_shape = get_boxes_max_vals(gray_image_with_single_shape, template_gray_images, dim)
    return sorted_max_values_for_single_shape
    # thresh = 0.7
    # for template_image_name in template_gray_images:
    #     template_gray_image = template_gray_images[template_image_name]
    #     # resize image
    #     # INTER_NEAREST
    #     # INTER_LINEAR
    #     # INTER_AREA
    #     # INTER_CUBIC
    #     # INTER_LANCZOS4
    #     resized_template_gray_image = cv2.resize(template_gray_image, dim, interpolation=cv2.INTER_AREA)
    #     (boxes, box_centers) = get_boxes_of_single_shape_by_matchTemplate(gray_image_with_single_shape,
    #                                                                       resized_template_gray_image, thresh)
    #     david = 5




# def detect_shapes_names(shapes_boxes, rgb_image, gray_image, template_gray_images):
#     num_of_shapes = len(shapes_boxes)
#     shapes_data = [None] * num_of_shapes
#     sd = ShapeDetector()
#     for shape_index, shape_box in enumerate(shapes_boxes):
#         x_min = int(shape_box[0])
#         y_min = int(shape_box[1])
#         x_max = int(shape_box[2])
#         y_max = int(shape_box[3])
#
#         shape_width = x_max - x_min
#         shape_height = y_max - y_min
#
#         max_dim = max(shape_width, shape_height)
#         dim = (max_dim, max_dim)
#         gray_image_with_single_shape = gray_image[y_min:y_max, x_min:x_max]
#         rgb_image_with_single_shape = rgb_image[y_min:y_max, x_min:x_max, :]
#
#
#         resized_rgb_image_with_single_shape = sd.get_image_in_template_dimensions(template_gray_images, rgb_image_with_single_shape)
#
#         resized_rgb_image_with_single_shape_without_border = sd.get_image_without_border(resized_rgb_image_with_single_shape)
#
#
#
#         resized_binary_image_with_single_shape_without_border = sd.convert_gray_image_to_binary_image_with_same_dims_as_template_image(template_gray_images,
#                                                                             gray_image_with_single_shape)
#         # cv2.imshow('gray_image_with_single_shape', gray_image_with_single_shape)
#         # cv2.waitKey()
#         #sorted_max_values_for_single_shape = detect_shape_scores(shape_box, gray_image_with_single_shape, template_gray_images, dim)
#         is_shape_legal = check_if_image_contains_one_of_the_shapes(gray_image_with_single_shape, template_gray_images)
#         if is_shape_legal == False:
#             shape_name = 'not_shape'
#             single_shape_data = {'shape_name': shape_name, 'shape_color_data': sd.unknown_color_info, 'shape_box': shape_box}
#             shapes_data[shape_index] = single_shape_data
#             continue
#         #shape_name = list(sorted_max_values_for_single_shape.keys())[0]
#         shape_contour_with_max_vertices = sd.detect_shape_contour(resized_binary_image_with_single_shape_without_border)
#         shape_name = sd.detect_shape_type_by_image(shape_contour_with_max_vertices)
#         shape_color_data = sd.detect_shape_color(resized_rgb_image_with_single_shape_without_border, shape_contour_with_max_vertices)
#         #single_shape_data = {'shape_name': shape_name, 'shape_box': shape_box, 'sorted_max_values': sorted_max_values_for_single_shape}
#         single_shape_data = {'shape_name': shape_name, 'shape_color_data': shape_color_data, 'shape_box': shape_box}
#         shapes_data[shape_index] = single_shape_data
#     return shapes_data

    # david = 5
    #
    # x_center = int(0.5 * (x_min + x_max))
    # y_center = int(0.5 * (y_min + y_max))
    # box_center = np.array((x_center, y_center), int)
    #
    #
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale = 0.6
    # thickness = 2
    # shape_color = 0
    # cv2.putText(gray_image, f'{shape_name}', box_center, font,
    #             fontScale, shape_color, thickness, cv2.LINE_AA)
    #
    #
    # cv2.imshow('gray_image', gray_image)
    # cv2.waitKey(0)






# def create_template_gray_images(image_data):
#     shapes_data = image_data['shapes_data']
#     template_images = {}
#     sd = ShapeDetector()
#     for shape_index, shape_name in enumerate(shapes_data):
#         template_full_path = 'original_shapes_jpg/' + shape_name + '.jpg'
#         template_rgb_image = cv2.imread(template_full_path)
#         template_gray_image = cv2.cvtColor(template_rgb_image, cv2.COLOR_BGR2GRAY)
#         #template_gray_image_without_border = sd.get_image_without_border(template_gray_image)
#         template_images[shape_name] = template_gray_image
#     return template_images


# def get_box_center(box):
#     x_min = int(box[0])
#     y_min = int(box[1])
#     x_max = int(box[2])
#     y_max = int(box[3])
#     x_center = int(0.5 * (x_min + x_max))
#     y_center = int(0.5 * (y_min + y_max))
#     box_center = np.array((x_center, y_center), int)
#     return box_center


# def write_screen_color_on_image(rgb_image, screen_data):
#     screen_color_data = screen_data['screen_color_data']
#     small_box_on_empty_part_of_screen = screen_data['small_box_on_empty_part_of_screen']
#     box_center = get_box_center(small_box_on_empty_part_of_screen)
#     screen_color_name = screen_color_data['name']
#     screen_color = screen_color_data['rgb_color']
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     fontScale = 0.6
#     thickness = 2
#     text_to_write = screen_color_name + ' screen'
#     cv2.putText(rgb_image, text_to_write, box_center, font,
#                 fontScale, screen_color, thickness, cv2.LINE_AA)
#     return rgb_image

# def write_shapes_names_on_image(rgb_image, shapes_data):
#     for single_shape_data in shapes_data:
#         shape_box = single_shape_data['shape_box']
#         #sorted_max_values_for_single_shape = single_shape_data['sorted_max_values']
#         x_min = int(shape_box[0])
#         y_min = int(shape_box[1])
#         x_max = int(shape_box[2])
#         y_max = int(shape_box[3])
#         x_center = int(0.5 * (x_min + x_max))
#         y_center = int(0.5 * (y_min + y_max))
#         box_center = np.array((x_center, y_center), int)
#
#         #shape_name = list(sorted_max_values_for_single_shape.keys())[0]
#         shape_name = single_shape_data['shape_name']
#         shape_color_data = single_shape_data['shape_color_data']
#
#         shape_color_name = shape_color_data["name"]
#         shape_color = shape_color_data["rgb_color"]
#
#
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         fontScale = 0.6
#         thickness = 2
#         cv2.putText(rgb_image, f'{shape_name}', box_center, font,
#                     fontScale, shape_color, thickness, cv2.LINE_AA)
#         cv2.putText(rgb_image, f'{shape_color_name}', box_center + (0, 30), font,
#                     fontScale, shape_color, thickness, cv2.LINE_AA)
#
#
#     return rgb_image

# def write_shapes_names_on_image_by_their_order(rgb_image, shapes_data):
#     x = 700
#     y = 100
#     for single_shape_data in shapes_data:
#         shape_name = single_shape_data['shape_name']
#         shape_color_data = single_shape_data['shape_color_data']
#
#         shape_color_name = shape_color_data["name"]
#         shape_color = shape_color_data["rgb_color"]
#
#
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         fontScale = 1.0
#         thickness = 2
#         cv2.putText(rgb_image, f'{shape_name}', (x, y), font,
#                     fontScale, shape_color, thickness, cv2.LINE_AA)
#         y += 25
#         cv2.putText(rgb_image, f'{shape_color_name}', (x, y), font,
#                     fontScale, shape_color, thickness, cv2.LINE_AA)
#
#         y += 50
#
#
#     return rgb_image



def main():
    folder_name = '012_mix'
    input_file_name = '00542.jpg'




    #folder_name = 'back_screen_shapes_colors'
    # folder_name = 'shapes_and_colors_no_sticker_1'
    #folder_name = 'shapes_on_screen'
    #folder_name = 'simon_shapes_with_paper_01'
    # input_folder_full_path = f'./input_data_simon/videos_and_images/images/' + folder_name + '/original'
    input_folder_full_path = f'./input_data/' + folder_name



    #input_folder_full_path = f'./input_data_simon/videos_and_images/images_simon_room/' + folder_name
    #input_file_name = '00245.jpg'
    #input_file_name = 'Simon_00001.jpg'
    image_full_path = input_folder_full_path + '/' + input_file_name




    jpg_files = sorted(glob.glob(input_folder_full_path + '/*.jpg'))
    #jpg_files = [image_full_path]
    num_of_frames = len(jpg_files)
    frame_milliseconds = 1

    simon_images_output_folder = './output_images/' + folder_name
    #simon_images_output_folder = './output_images/shapes_and_colors'

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


        # rgb_image = np.zeros(rgb_image.shape, np.uint8)
        # rgb_image[50:150, 60, 1] = 255
        # rgb_image[50:150, 200, 1] = 255
        # rgb_image[50, 60:200, 1] = 255
        # rgb_image[150, 60:200, 1] = 255
        # rgb_image[80, 60:100, 1] = 255
        # rgb_image[120, 60:100, 1] = 255
        # rgb_image[80:120, 100, 1] = 255

        image_data = sd.get_image_data_from_frame(rgb_image)
        # if len(image_data['shapes_data']) == 8:
        #     image_data['shapes_data'][0]['shape_name'] = 'rectangle'
        #     image_data['shapes_data'][0]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[0] #brown
        #
        #     image_data['shapes_data'][1]['shape_name'] = 'octagon'
        #     image_data['shapes_data'][1]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[1] #yellow
        #
        #     image_data['shapes_data'][2]['shape_name'] = 'rhombus'
        #     image_data['shapes_data'][2]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[7] #pink
        #
        #     image_data['shapes_data'][3]['shape_name'] = 'triangle'
        #     image_data['shapes_data'][3]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[4] #purple
        #
        #     image_data['shapes_data'][4]['shape_name'] = 'square'
        #     image_data['shapes_data'][4]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[2] #orange
        #
        #     image_data['shapes_data'][5]['shape_name'] = 'circle'
        #     image_data['shapes_data'][5]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[3] #red
        #
        #     image_data['shapes_data'][6]['shape_name'] = 'pentagon'
        #     image_data['shapes_data'][6]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[6] #blue
        #
        #     image_data['shapes_data'][7]['shape_name'] = 'star'
        #     image_data['shapes_data'][7]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[5] #green

        # if len(image_data['shapes_data']) == 8:
        #     image_data['shapes_data'][0]['shape_name'] = 'circle'
        #     image_data['shapes_data'][0]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[3] #red
        #
        #     image_data['shapes_data'][1]['shape_name'] = 'triangle'
        #     image_data['shapes_data'][1]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[4] #purple
        #
        #     image_data['shapes_data'][2]['shape_name'] = 'pentagon'
        #     image_data['shapes_data'][2]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[6] #blue
        #
        #     image_data['shapes_data'][3]['shape_name'] = 'octagon'
        #     image_data['shapes_data'][3]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[1] #yellow
        #
        #     image_data['shapes_data'][4]['shape_name'] = 'rhombus'
        #     image_data['shapes_data'][4]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[7] #pink
        #
        #     image_data['shapes_data'][5]['shape_name'] = 'star'
        #     image_data['shapes_data'][5]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[5] #green
        #
        #     image_data['shapes_data'][6]['shape_name'] = 'rectangle'
        #     image_data['shapes_data'][6]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[0] #brown
        #
        #     image_data['shapes_data'][7]['shape_name'] = 'square'
        #     image_data['shapes_data'][7]['shape_color_data'] = sd.colors_ranges_info_tello_without_stick[2] #orange
        #if image_data['image_data_type'] == self.ImageDateType.QR_COLORS_ONLY:
        shapes_data = image_data['shapes_data']
        shapes_boxes = [single_shape_data['shape_box'] for single_shape_data in shapes_data]
        small_shapes_boxes_for_color_detection = [single_shape_data['small_shape_box_for_color_detection'] for single_shape_data in shapes_data]

        screen_data = image_data['screen_data']
        if image_data is None:
            david = 6
        elif image_data['screen_data'] is None:
            david = 8
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






        file_full_path = "{}/{:05d}.jpg".format(simon_images_output_folder, frame_index+1)
        cv2.imwrite(file_full_path, rgb_image)
        cv2.imshow('rgb_image', rgb_image)
        cv2.waitKey(frame_milliseconds)

main()
