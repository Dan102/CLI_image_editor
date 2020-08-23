"""
PEP8 enhanced CLI image processing script
"""
import argparse
from enum import Enum
from typing import List
from collections import OrderedDict, defaultdict
import sys
# Unfortunately import PIL alone is not enough, as Image is submodule and PIL finds it on its own
# PIL.Image doesn't work
from PIL import Image
import PIL
import numpy as np

RGB_PIXEL_MAX_VALUE = 255
BLUR_KERNEL = np.array([[1] * 9 for x in range(9)]) / (9*9)
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
DEFAULT_OUTPUT_FILE = "altered.jpg"
MAX_ITERATIONS = 200


class Mirroring(Enum):
    """
    Enum for mirroring image.
    """
    HORIZONTAL = 0
    VERTICAL = 1


class Color(Enum):
    """
    Enum for changing image color components.
    """
    RED = 0
    GREEN = 1
    BLUE = 2


class HorizontalPosition(Enum):
    """
    Enum for predefined position for image cropping
    """
    LEFT = 0
    CENTER = 1
    RIGHT = 2


class VerticalPosition(Enum):
    """
    Enum for predefined position for image cropping.
    """
    TOP = 0
    CENTER = 1
    BOTTOM = 2


def load_img_to_np_array(image_name: str) -> np.array:
    """
    Loads image from file into numpy array.
    """
    try:
        with Image.open(image_name) as source_image:
            np_img = np.array(source_image)
            return np_img
    except FileNotFoundError:
        print("ERROR: Specified source file " + image_name + " doesn't exists.")
        sys.exit(100)
    except PIL.UnidentifiedImageError:
        print("ERROR: Specified source file " + image_name + " doesn't seem to be an image.")
        sys.exit(101)


def save_img_from_np_array(image: np.array, image_name: str):
    """
    Saves image from numpy array into file.
    """
    try:
        changed_img = Image.fromarray(image)
        changed_img.save(image_name)
        changed_img.close()
    except PIL.UnidentifiedImageError:
        print("ERROR: Specified target file " + image_name + " couldn't be saved.")
        sys.exit(102)
    except ValueError:
        print("ERROR: Specified target file extension " + image_name + " is probably"
              " not valid image extension.")
        sys.exit(103)



# REQUIRED METHODS #
def rotate_img_right(image: np.array) -> np.array:
    """
    Rotate image by 90 degrees into right.
    """
    return np.rot90(image, 1, axes=(1, 0))


def mirror_img(image: np.array, mir_type: Mirroring = Mirroring.HORIZONTAL) -> np.array:
    """
    Mirror image by axis x.
    """
    return np.flip(image, axis=mir_type.value)


def get_inverse_img(image: np.array) -> np.array:
    """
    Get inverse value for every pixel.
    """
    return RGB_PIXEL_MAX_VALUE - image


def get_greyscale_img(image: np.array) -> np.array:
    """
    Turn image into greyscale, every color component will be the same.
    """
    # I gave preference to condition because Exception is too generic (out of index)
    if len(image.shape) != 3:
        print("ERROR: cannot convert image to greyscale as its already greyscale.")
        sys.exit(104)
    img_color_means = (np.mean(image, axis=2)).astype(np.uint8)
    return np.repeat(img_color_means[:, :, None], 3, axis=2)


def change_img_brightness(image: np.array, percentage: int = 50) -> np.array:
    """
    Add to each pixel certaini value.
    """
    brighter_img = (image + (percentage / 100) * RGB_PIXEL_MAX_VALUE)
    return np.clip(brighter_img, 0, 255).astype(np.uint8)


def lighten_img(image: np.array, percentage: int = 50) -> np.array:
    """
    Make each pixel brighter by adding certain value.
    """
    # Percentage argument checked in argument processing
    return change_img_brightness(image, percentage)


def darken_img(image: np.array, percentage: int = 50) -> np.array:
    """
    Make each pixel darker by subsctracting certain value.
    """
    # Percentage argument checked in argument processing
    return change_img_brightness(image, -percentage)


def apply_filter(image: np.array, kernel: np.array) -> np.array:
    """
    Apply filter to image. Filter must have odd shape. On the corner, missing
    values are considered to be 0.
    """
    # Filter must have odd shape
    dimension = len(np.shape(image))
    img_h, img_w = image.shape[0:2]
    kernel_h, kernel_w = kernel.shape
    h_side, w_side = kernel_h//2, kernel_w//2
    padding_tuple = [(h_side, h_side), (w_side, w_side)]
    if dimension == 3:
        padding_tuple.append((0, 0))
    # Pad filter with zeroes
    img_pad = np.pad(image.astype(np.float), padding_tuple, "constant", constant_values=(0, 0))
    all_rolled_images = []
    # Create views which are rolled to all possible direction
    for i in range(kernel_h):
        rolled_img_view = np.roll(img_pad, h_side - i, axis=0)
        for j in range(kernel_w):
            rolled_img_with_filter = np.roll(rolled_img_view, w_side - j, axis=1) * kernel[i, j]
            all_rolled_images.append(rolled_img_with_filter)
    convoluted_img = np.sum(all_rolled_images, axis=0)
    not_padded_img = convoluted_img[h_side: img_h + h_side, w_side: img_w + w_side]
    return np.clip(not_padded_img, 0, 255).astype(np.uint8)


def sharpen_img(image: np.array) -> np.array:
    """
    Sharpen image by applying sharpen kernel filter
    """
    return apply_filter(image, np.array(SHARPEN_KERNEL))


# EXTENSION METHODS #
def blur_img(image: np.array) -> np.array:
    """
    Apply filter with kernel consisting of 1, to make every pixel average of surrounding pixels
    """
    return apply_filter(image, BLUR_KERNEL)


def reduce_color_components(image: np.array, colors: List[Color] = [Color.RED],
                            percentage: int = 100) -> np.array:
    """
    Reduce certain color components (Red or Green or Blue) in intensity (relatively to current
    level). 100% reduction will remove it entirely. No need to check percentage in <0, 100> as it
    is checked when arguments are processed.
    """
    # I gave preference to condition because Exception is too generic (out of index)
    if len(image.shape) != 3:
        print("ERROR: cannot reduce color componets in image as its greyscale image and only has"
              " one color component.")
        sys.exit(105)
    color_indexes_to_reduce = [c.value for c in colors]
    img_copy = image.copy()
    img_copy[:, :, color_indexes_to_reduce] = img_copy[:, :, color_indexes_to_reduce] *\
                                              (1 - percentage / 100)
    return img_copy


def get_single_color_img(image: np.array, color: Color = Color.RED) -> np.array:
    """
    Leave only one color component intact in the image and get rid of both others.
    """
    # I gave preference to condition because Exception is too generic (out of index)
    if len(image.shape) != 3:
        print("ERROR: cannot reduce color componets in image as its greyscale image and only has"
              " one color component.")
        sys.exit(106)
    return reduce_color_components(image, [c for c in Color if c.value != color.value], 100)


def k_means_clustering(image: np.array, k: int) -> np.array:
    """
    My implementation of K-Means clustering algorithm, to cluster image into k colors.
    The algorithm does not always generate same results, as starting point is chosen randomly
    (which is part of the standard implementation) but it is use for its reasonable performance
    and results. This algorithm (in my case) aims to choose K colors which "represents the image
    best". Although it does not minimize exactly distance (it minimize SQUARED distance), i
    usually gives good results. The algorithm is capped by max_iteration (random initialization
    and the beginning doesnt have to lead to anything) and can be partially reset during its run
    as some centroids (center of one of the k means) can vanish.
    """
    # I gave preference to condition because Exception is too generic (out of index)
    if len(image.shape) != 3:
        print("ERROR: cannot reduce color variety in 2d image as in implementation I decided that"
              " reducing color variety in greyscale images would make them non recognizable")
        sys.exit(107)
    height, width = image.shape[0:2]
    reshaped_1d_img = image.reshape(height * width, image.shape[2]).astype(np.int)
    unique_colors = np.unique(reshaped_1d_img, axis=0)
    centroids_indexes = np.random.choice(range(unique_colors.shape[0]), k, replace=False)
    centroids = unique_colors[centroids_indexes, :]
    cluster_labels = np.zeros(height*width)
    i = 0
    print("K-means clustering algorithm: ")
    while i < MAX_ITERATIONS:
        # Assign each pixel to closest centroids
        centroids_for_broadcast = centroids.reshape(k, 1, image.shape[2])
        distances_by_axes = reshaped_1d_img - centroids_for_broadcast
        total_distances_from_centroids = np.linalg.norm(distances_by_axes, axis=2)
        new_cluster_labels = np.argmin(total_distances_from_centroids.T, axis=1)
        if np.array_equal(new_cluster_labels, cluster_labels):
            break
        cluster_labels = new_cluster_labels
        # Get new centroids
        for j in range(k):
            class_indexes = (new_cluster_labels == j)
            # Unfortunately cluster can become empty during processing, then asses new centroid
            if np.sum(class_indexes) > 0:
                new_centroid = np.mean(reshaped_1d_img[class_indexes, :], axis=0)
                centroids[j, :] = new_centroid
            # If we assign labels always in given order, this will for sure converge
            else:
                random_i = np.random.choice(range(unique_colors.shape[0]), 1)
                centroids[j, :] = reshaped_1d_img[random_i, :]
        i += 1
        if i % 10 == 1:
            print(f"   {i}th iteration of maximum {MAX_ITERATIONS}")
    if i == MAX_ITERATIONS:
        print("ERROR: Max iteration reached, operation aborted. Try increasing maximum iteration"
              " limit with -mi MAX_ITERATION")
        sys.exit(108)
    return centroids, cluster_labels


def reduce_color_variety(image: np.array, colors: int = 8) -> np.array:
    """
    Aims to reduce image to certain number of colors (which were present in the given image) in the
    way that difference between old colors and new colors will be as good as possible (It minimizes
    sum of SQUARED distance instead of normal distance, but the heuristic still gives good results,
    as minimizing sum of normal distance would be too computationally demanding.)
    """
    # I gave preference to condition because Exception is too generic (out of index)
    if len(image.shape) != 3:
        print("ERROR: cannot reduce color variety in 2d image as in implementation I decided that"
              " reducing color variety in greyscale images would make them non recognizable")
        sys.exit(109)
    if len(np.unique(image)) <= colors:
        print("ERROR: Operation for reducing color variety couldn't be processed as there is less"
              " or same amount of colors in the image as desired number of colors.")
        sys.exit(110)
    height, width = image.shape[0:2]
    centroids, cluster_labels = k_means_clustering(image, colors)
    reshaped_1d_img = centroids[cluster_labels]
    return (reshaped_1d_img.reshape(height, width, image.shape[2])).astype(np.uint8)


def add_gaussian_noise_img(image: np.array, percentage: int = 30) -> np.array:
    """
    Add random noise to every pixel in the image
    """
    # Percentage argument checked in argument processing
    gaussian_noise = np.random.normal(loc=0, scale=(percentage / 100 * 255), size=image.shape)
    return np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)


def crop_image(image: np.array, height: int, width: int, y_start: int, x_start: int) -> np.array:
    """
    Crop image to certain height and width, given the coordinates of left upper corner of new image
     in relation to the old image.
    """
    img_h, img_w = image.shape[0:2]
    # NUMPY doesn't throw any exception (operation is valid, it just doesn't do anything if new
    # image size is bigger than old) but we need to inform user that his operation didn't do what
    # he wanted
    y_end = y_start + height
    x_end = x_start + width
    if img_h < y_end or img_w < x_end:
        print("ERROR: Operation for cropping image couldn't be processed as the new image"
              " size located at new origine is bigger than whats left from source image.")
        sys.exit(111)
    if len(image.shape) == 2:
        return image[y_start: y_end, x_start: x_end]
    return image[y_start: y_end, x_start: x_end, :]


def crop_image_predefined(image: np.array, height: int, width: int,
                          pos1: VerticalPosition = VerticalPosition.CENTER,
                          pos2: HorizontalPosition = HorizontalPosition.CENTER) -> np.array:
    """
    Crop image to certain height and width, given predefined possibilities of new image position
     related to the old image (eg. crop image in the way that new image will be taken from the
     centre of the old image)
    """
    vertical = pos1
    horizontal = pos2
    # So arguments can be send in any order (eg. only set one of them by program parameter)
    if isinstance(pos1, HorizontalPosition):
        vertical, horizontal = pos2, pos1
        if isinstance(pos2, HorizontalPosition):
            vertical = VerticalPosition.CENTER
    y_start, x_start = 0, 0
    img_height, img_width = image.shape[0:2]
    if vertical == VerticalPosition.CENTER:
        y_start = (img_height - height) // 2
    elif vertical == VerticalPosition.BOTTOM:
        y_start = img_height - height
    if horizontal == HorizontalPosition.CENTER:
        x_start = (img_width - width) // 2
    elif horizontal == HorizontalPosition.RIGHT:
        x_start = img_width - width
    return crop_image(image, height, width, y_start, x_start)


class OrderedStore(argparse.Action):
    """
    Own class for argParser action, because there is a need to keep the order of arguments
    intact (not ensured by argparser itself! as list of same operations in different order
    can cause different results) It was convenient to add another structure which keeps info
    about how many times was each parameter set
    """
    def __call__(self, parser, namespace, values, option_string=None):
        if not 'ord_args' in namespace:
            setattr(namespace, 'ord_args', OrderedDict())
            setattr(namespace, 'count_args', defaultdict(int))
        if (self.dest in namespace.ord_args) and (values != []):
            print("Undefined behavior, same parametric operation called twice")
            sys.exit(112)
        namespace.ord_args[self.dest] = values

        namespace.count_args[self.dest] += 1
        setattr(namespace, self.dest, values)


# This is for elegant casting in Argparser. I didnt find other way as i need BOTH
# NUMERICAL representation and both STRING representation for enum
def string_to_color_enum(color_str: str) -> Color:
    """
    Own method for casting string to enum Color, used in validating input
    """
    print(color_str)
    try:
        return Color[color_str.upper()]
    except KeyError:
        print("ERROR: Color for operation 'single color' not recognized."
              " Choices are {'red', 'green', 'blue'}")
        sys.exit(113)


def string_to_ver_pos_enum(ver_pos_str: str) -> VerticalPosition:
    """
    Own method for casting string to enum Vertical Position, used in validating input
    """
    try:
        return VerticalPosition[ver_pos_str.upper()]
    except KeyError:
        print("ERROR: Vertical position for crop in 'cvp' not recognized."
              " Choices are: " + str([vp.name.lower() for vp in VerticalPosition]))
        sys.exit(114)


def string_to_hor_pos_enum(hor_pos_str: str) -> HorizontalPosition:
    """
    Own method for casting string to enum Horizontal Position, used in validating input
    """
    try:
        return HorizontalPosition[hor_pos_str.upper()]
    except KeyError:
        print("ERROR: Horizontal position for crop in 'chp' not recognized."
              " Choices are: " + str([hp.name.lower() for hp in HorizontalPosition]))


if __name__ == "__main__":

    ARGUMENTS_OPERATIONS_SWITCH = {
        "r": rotate_img_right,
        "rotate": rotate_img_right,
        "m": mirror_img,
        "mirror": mirror_img,
        "i": get_inverse_img,
        "inverse": get_inverse_img,
        "bw": get_greyscale_img,
        "l": lighten_img,
        "lighten": lighten_img,
        "d": darken_img,
        "darken": darken_img,
        "s": sharpen_img,
        "sharpen": sharpen_img,
        "b": blur_img,
        "blur": blur_img,
        "sc": get_single_color_img,
        "singlecolor": get_single_color_img,
        "rv": reduce_color_variety,
        "reducevariety": reduce_color_variety,
        "gn": add_gaussian_noise_img,
        "gaussiannoise": add_gaussian_noise_img,
        "c": crop_image,
        "crop": crop_image,
        "cropsize": crop_image_predefined,
    }

    parser = argparse.ArgumentParser(description="Tool for processing images in command line."
                                                 " Commands without parameters can be set"
                                                 " multiple times.")
    operations_queue = []
    parser.add_argument('-mi', '--max_iteration', action='store', default=MAX_ITERATIONS,
                        type=int,
                        help="Set different max iteration value for K-means algorithm used in"
                             " reducing colors. Default value is 100")
    parser.add_argument('-r', '--rotate', action=OrderedStore, nargs=0,
                        help="Rotate image by 90 degrees to right. Can be set multiple times"
                             " in one run of program.")
    parser.add_argument('-m', '--mirror', action=OrderedStore, nargs=0, help="Mirror image by"
                                                                             " axis x.")
    parser.add_argument('-i', '--inverse', action=OrderedStore, nargs=0,
                        help="Inverse all the pixels' colors in the image (create negative).")
    parser.add_argument('--bw', action=OrderedStore, nargs=0,
                        help="Convert pixels' color in the image into shades of gray.")
    parser.add_argument('-l', '--lighten',
                        action=OrderedStore,
                        nargs=1,
                        type=int,
                        choices=range(101),
                        metavar="PERCENTAGE",
                        help="Lighten the image. The required parameter must be in percentage,"
                             " and only integer in interval <0; 100>.")
    parser.add_argument('-d', '--darken',
                        action=OrderedStore,
                        nargs=1,
                        type=int,
                        choices=range(101),
                        metavar="PERCENTAGE",
                        help="Darken the image. The required parameter must be in percentage,"
                             " and only integer in interval <0; 100>.")
    parser.add_argument('-s', '--sharpen', action=OrderedStore, nargs=0, help="Make the edges"
                                                                              " in the image"
                                                                              " sharper.")
    parser.add_argument('-b', '--blur', action=OrderedStore, nargs=0, help="Blur the image.")
    parser.add_argument('-rv', '--reducevariety', action=OrderedStore, type=int,
                        metavar="REDUCED_COLORS_NUMBER",
                        help="Reduce number of colors in the image to the given number. K-mean"
                             " clustering is used as algorithm to try to find reduced number of"
                             " colors in the picture which represents the image best. The"
                             " required parameter must be integer")
    parser.add_argument('-sc', '--singlecolor', action=OrderedStore, type=string_to_color_enum,
                        metavar=set([color.name.lower() for color in Color]),
                        help="Leave only one color component in the image, reduce others to 0.")
    parser.add_argument('-gn', '--gaussiannoise', action=OrderedStore, type=int, choices=range(101),
                        metavar="PERCENTAGE",
                        help="Add random (gaussian) noise to every pixel in the image. Parameter"
                             " is in percentage, which is relative to max value of each pixel"
                             " (255), only integer in interval <0; 100> accepted.")
    parser.add_argument('-c', '--crop', action=OrderedStore, nargs=4, type=int,
                        metavar=("HEIGHT", "WIDTH", "Y_START", "X_START"),
                        help="Crop the image. Given parameters are: height, width, y_start and"
                             " x_start (top left corner and size of the new image, relative to"
                             " the old image.")
    parser.add_argument('--cropsize', action=OrderedStore, nargs=2, type=int,
                        help="Predefined crop, set size of the new image and then with --cvp"
                             " and --chp where to crop (horizontal and vertical). Default is"
                             " to position the new cropped image to the center")
    parser.add_argument('--cvp', action=OrderedStore, type=string_to_ver_pos_enum,
                        metavar=set([vp.name.lower() for vp in VerticalPosition]),
                        help="Predefined crop, set vertical position for crop. To have effect"
                             " must be used with --cropsize parameter")
    parser.add_argument('--chp', action=OrderedStore, type=string_to_hor_pos_enum,
                        metavar=set([hp.name.lower() for hp in HorizontalPosition]),
                        help="Predefined crop, set horizontal position for crop. To have effect,"
                             " must be used with --cropsize parameter")
    parser.add_argument('source_img_path', metavar="SOURCE_IMAGE_PATH",
                        help="Path (relative or absolute) to the source image which should"
                             " be processed.")
    parser.add_argument('target_img_path', nargs='?', default=DEFAULT_OUTPUT_FILE,
                        metavar="TARGET_IMAGE_PATH",
                        help="Path (relative or abs) to where the result of source image"
                             " processing will be saved.")

    args = parser.parse_args()

    MAX_ITERATIONS = args.max_iteration

    if "ord_args" not in args:
        print("ERROR: No opperations specified.")
        sys.exit(115)
    specified_operations = []
    for operation in args.ord_args.keys():
        specified_operations += [operation] * args.count_args[operation]
    print("Specified operations in order: ", specified_operations)

    img = load_img_to_np_array(args.source_img_path)
    if 'cropsize' in args.ord_args:
        if 'cvp' in args.ord_args:
            args.ord_args['cropsize'].append(args.ord_args['cvp'])
            del args.ord_args['cvp']
        if 'chp' in args.ord_args:
            args.ord_args['cropsize'].append(args.ord_args['chp'])
            del args.ord_args['chp']
    for operation, operation_parameters in args.ord_args.items():
        all_parameters = [img]
        if isinstance(operation_parameters, list):
            all_parameters.extend(operation_parameters)
        else:
            all_parameters.append(operation_parameters)
        for op in range(args.count_args[operation]):
            all_parameters[0] = img
            img = ARGUMENTS_OPERATIONS_SWITCH[operation](*all_parameters)

    save_img_from_np_array(img, args.target_img_path)
