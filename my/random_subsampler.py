import os.path
import random
from typing import Tuple, List

import cv2
import numpy as np
import math


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"


def transform_point(point: Point, matrix: np.matrix) -> Point:
    transformed_point = np.array(np.matmul(matrix, np.array([point.x, point.y, 1]).T))
    return Point(int(transformed_point[0][0]), int(transformed_point[0][1]))


def subsample_image(filename: str, crop_width: int, crop_height: int, rnd: random.Random) -> Tuple[
    np.ndarray, Point, np.matrix]:
    img = cv2.imread(filename)
    img_h, img_w = img.shape[:2]
    point, angle = get_random_point_and_angle(img_w, img_h, crop_width, crop_height, rnd)

    rotated_img, affine_mat = rotate_image(img, angle / (2 * math.pi) * 360)
    rotated_point = transform_point(point, affine_mat)
    cropped_img = rotated_img[rotated_point.y:rotated_point.y + crop_height,
                  rotated_point.x:rotated_point.x + crop_width]

    # cv2.circle(rotated_img, (rotated_point.x, rotated_point.y), 5, (0,255,0), -1)
    # cv2.imshow("hhh", rotated_img)
    # cv2.waitKey(0)
    return cropped_img, rotated_point, affine_mat


def get_random_point_and_angle(img_width: int, img_height: int, cropped_width: int, cropped_height: int,
                               rnd: random.Random) -> Tuple[Point, float]:
    while True:
        start_x = rnd.randint(5, img_width - 5)
        start_y = rnd.randint(5, img_height - 5)
        angle = rnd.random() * 2 * math.pi
        end_x = math.ceil(start_x + math.cos(angle) * cropped_width)
        end_y = math.ceil(start_y + math.sin(angle) * cropped_width)

        if end_x > img_width or end_x < 0 or end_y > img_height or end_y < 0:
            continue

        angle_to_other = (angle + math.pi / 2) % (2 * math.pi)
        end_x_2 = math.ceil(start_x + math.cos(angle_to_other) * cropped_height)
        end_y_2 = math.ceil(start_y + math.sin(angle_to_other) * cropped_height)
        if end_x_2 > img_width or end_x_2 < 0 or end_y_2 > img_height or end_y_2 < 0:
            continue

        end_x_3 = end_x + (end_x_2 - start_x)
        end_y_3 = end_y + (end_y_2 - start_y)
        if end_x_3 > img_width or end_x_3 < 0 or end_y_3 > img_height or end_y_3 < 0:
            continue

        # ordered = order_points(np.array([[start_x, start_y], [end_x, end_y], [end_x_2, end_y_2], [end_x_3, end_y_3]]))

        # return [[start_x, start_y], [end_x, end_y], [end_x_2, end_y_2], [end_x_3, end_y_3]], angle
        return Point(start_x, start_y), angle


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result, affine_mat


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def parse_labels(path: str, width: int, height: int) -> List[Tuple[Point, Point, int]]:
    with open(path, 'r') as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        split = line.split(" ")
        cls = int(split[0])
        bbox = [float(split[1]) * width, float(split[2]) * height, float(split[3]) * width,
                float(split[4]) * height]
        bbox = np.array(bbox, dtype=float)
        bbox = xywh2xyxy(bbox)
        top_left = Point(int(bbox[0]), int(bbox[1]))
        bottom_right = Point(int(bbox[2]), int(bbox[3]))
        labels.append((top_left, bottom_right, cls))
    return labels


def serilize_label(top_left: Point, bottom_right: Point, cls: int, img_width: int, img_height: int) -> str:
    # Label is class center_x center_y label_width label_height (all relative)
    label_width = bottom_right.x - top_left.x
    label_height = bottom_right.y - top_left.y
    center_x = top_left.x + (label_width / 2)
    center_y = top_left.y + (label_height / 2)

    return f"{cls} {center_x / img_width} {center_y / img_height} {label_width / img_width} {label_height / img_height}"


def is_inside(point: Point, img_top_left: Point, width: int, height: int) -> bool:
    return is_inside2(point.x, point.y, img_top_left, width, height)


def is_inside2(x, y, img_top_left: Point, width: int, height: int) -> bool:
    return img_top_left.x <= x < img_top_left.x + width and img_top_left.y <= y < img_top_left.y + height


def get_overlapping_bbox(img_top_left: Point, img_width: int, img_height: int, unrotated_top_left: Point,
                         unrotated_bottom_right: Point, transform: np.matrix) -> Tuple[Point, Point] or None:
    unrotated_top_right = Point(unrotated_bottom_right.x, unrotated_top_left.y)
    unrotated_bottom_left = Point(unrotated_top_left.x, unrotated_bottom_right.y)

    unrotated_points = [unrotated_top_left, unrotated_top_right, unrotated_bottom_right, unrotated_bottom_left,
                        unrotated_top_left]
    # unrotated_points = [unrotated_top_left, unrotated_top_right, unrotated_bottom_right, unrotated_bottom_left]
    rotated_points = [transform_point(x, transform) for x in unrotated_points]
    inside_points = []
    last_inside = False
    last_point = None
    for point in rotated_points:
        this_inside = False
        if is_inside(point, img_top_left, img_width, img_height):
            inside_points.append(point)
            this_inside = True

        if last_inside != this_inside and last_point is not None:
            # Crossed the image edge, find the first pixel that is inside the image along the line from this point to the previous one
            # Go from the point outside to the point inside
            direction = np.array([last_point.x - point.x, last_point.y - point.y])
            start = np.array([point.x, point.y]).astype(float)
            if this_inside:
                direction *= -1
                start = np.array([last_point.x, last_point.y]).astype(float)
            for _ in range(100):
                start += direction * 1 / 100
                if (is_inside2(int(start[0]), int(start[1]), img_top_left, img_width, img_height)):
                    inside_point = Point(int(start[0]), int(start[1]))
                    inside_points.append(inside_point)
                    break
            pass

        last_inside = this_inside
        last_point = point

    if len(inside_points) == 0:
        return None

    mask = np.zeros((img_height, img_width, 1), dtype=np.uint8)
    offset_points = list(map(lambda p: Point(p.x - img_top_left.x, p.y - img_top_left.y), rotated_points[:-1]))
    pt_array = np.array(list(map(lambda p: [p.x, p.y], offset_points)), np.int32)
    pt_array = pt_array.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pt_array], (255, 255, 255))

    # if there are very few pixels in the label, ignore it
    num_pixels_of_label = np.count_nonzero(mask)
    if num_pixels_of_label < 100:
        return None

    # Get the bounding box of all the points
    xs = sorted(map(lambda p: p.x, inside_points))
    ys = sorted(map(lambda p: p.y, inside_points))
    min_x, max_x = xs[0], xs[-1]
    min_y, max_y = ys[0], ys[-1]
    return Point(min_x - img_top_left.x, min_y - img_top_left.y), Point(max_x - img_top_left.x,
                                                                        max_y - img_top_left.y), mask


def process_img(img_path, img_type, new_dataset_base, rnd, only_negative):
    img_name = os.path.basename(img_path)
    img_number = int(img_name.replace(".PNG", "").replace("frame_", ""))
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    cropped_width = 256
    cropped_height = 256

    labels = parse_labels(img_path.replace(".PNG", ".txt").replace("images", "labels"), img_width, img_height)
    cropped_img, top_left, transform = subsample_image(img_path, cropped_width, cropped_height, rnd)
    new_labels = []
    masks = []
    for label in labels:
        overlap = get_overlapping_bbox(top_left, cropped_width, cropped_height, label[0], label[1], transform)
        if overlap is None:
            continue

        label_top_left, label_bottom_right, mask = overlap

        # cls = label[2]
        cls = 0
        new_label = serilize_label(label_top_left, label_bottom_right, cls, cropped_width, cropped_height)
        new_labels.append(new_label)
        masks.append(mask)

    if only_negative and len(new_labels) != 0:
        # try again until we get an image with no labels inside
        return process_img(img_path, img_type, new_dataset_base, rnd, only_negative)

    label_path = os.path.join(new_dataset_base, img_type, "labels")
    os.makedirs(label_path, exist_ok=True)
    with open(os.path.join(label_path, f"frame_{img_number}.txt"), "w") as f:
        f.write("\n".join(new_labels))

        # new_parsed = parse_labels(f"/home/m/datasets/Novi11Cropped/label{i}.txt", cropped_width, cropped_height)
        # for n in new_parsed:
        #    cv2.circle(cropped_img, (n[0].x, n[0].y), 5, (255, 0,0), -1)
        #    cv2.circle(cropped_img, (n[1].x, n[1].y), 5, (0, 255, 0), - 1)
    img_path = os.path.join(new_dataset_base, img_type, "images")
    os.makedirs(img_path, exist_ok=True)
    cv2.imwrite(os.path.join(img_path, f"frame_{img_number}.PNG"), cropped_img)

    masks_path = os.path.join(new_dataset_base, img_type, "masks")
    os.makedirs(masks_path, exist_ok=True)
    if len(masks) == 0:
        mask = np.zeros((img_height, img_width, 1), dtype=np.uint8)
    else:
        mask = masks[0]

    for i in range(1, len(masks)):
        other_mask = masks[i]
        mask[other_mask == 255] = 255

    cv2.imwrite(os.path.join(masks_path, f"frame_{img_number}.PNG"), mask)


if __name__ == "__main__":
    rnd = random.Random()
    rnd.seed(42)

    base_path = "/home/m/datasets/Novi11Fair"
    new_path = "/home/m/datasets/Novi11CroppedUnsupervised"
    only_negative_train = True
    img_types = ["test", "train", "valid"]
    for img_type in img_types:
        img_dir = os.path.join(base_path, img_type, "images")
        images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if ".PNG" in f]
        for image in images:
            produce_only_negative = only_negative_train and img_type == "train"
            process_img(image, img_type, new_path, rnd, produce_only_negative)
