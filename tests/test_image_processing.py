import numpy as np
import pytest
from numpy.testing import assert_equal
import program.image_processing as image_processing


def repeat_2d_image_array_into_3d(img):
    return np.repeat(img[:, :, None], 3, axis=2)


TEST_IMAGE_LOCATION = 'tests/test.jpg'
TEST_MATRIX_2D = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 11, 12]])
TEST_MATRIX_3D = repeat_2d_image_array_into_3d(TEST_MATRIX_2D)
TEST_MATRIX_3D_NON_GREY = TEST_MATRIX_3D + np.array([0, 1, 2])  # r+0, g+1, b+2


@pytest.fixture
def image():
    return image_processing.load_img_to_np_array(TEST_IMAGE_LOCATION)


# REQUIRED METHODS #
def test_rotate_img_right():
    rotated_img_2d = image_processing.rotate_img_right(TEST_MATRIX_2D)
    rotated_img_target_2d = np.array([[10, 7, 4, 1],
                                      [11, 8, 5, 2],
                                      [12, 9, 6, 3]])
    rotated_img_3d = image_processing.rotate_img_right(TEST_MATRIX_3D)
    rotated_img_target_3d = repeat_2d_image_array_into_3d(rotated_img_target_2d)
    assert_equal(rotated_img_2d, rotated_img_target_2d)
    assert_equal(rotated_img_3d, rotated_img_target_3d)


def test_mirror_img():
    mirrored_img_2d = image_processing.mirror_img(TEST_MATRIX_2D)
    mirrored_img_target_2d = np.array([[10, 11, 12],
                                       [7, 8, 9],
                                       [4, 5, 6],
                                       [1, 2, 3]])
    mirrored_img_3d = image_processing.mirror_img(TEST_MATRIX_3D)
    mirrored_img_target_3d = repeat_2d_image_array_into_3d(mirrored_img_target_2d)
    assert_equal(mirrored_img_2d, mirrored_img_target_2d)
    assert_equal(mirrored_img_3d, mirrored_img_target_3d)


def test_get_inverse_img():
    inverse_img_2d = image_processing.get_inverse_img(TEST_MATRIX_2D)
    inverse_img_target_2d = np.array([[255-1, 255-2, 255-3],
                                      [255-4, 255-5, 255-6],
                                      [255-7, 255-8, 255-9],
                                      [255-10, 255-11, 255-12]])
    inverse_img_3d = image_processing.get_inverse_img(TEST_MATRIX_3D)
    inverse_img_target_3d = repeat_2d_image_array_into_3d(inverse_img_target_2d)
    assert_equal(inverse_img_2d, inverse_img_target_2d)
    assert_equal(inverse_img_3d, inverse_img_target_3d)


def test_get_greyscale_img():
    # Testing 2D images doesnt make sense, they are greyscale from beginning
    greyscale_img_3d = image_processing.get_greyscale_img(TEST_MATRIX_3D_NON_GREY)
    greyscale_img_target_3d = TEST_MATRIX_3D + 1
    assert_equal(greyscale_img_3d, greyscale_img_target_3d)


def test_lighten_img():
    lighten_img_2d = image_processing.lighten_img(TEST_MATRIX_2D, 50)
    lighten_img_target_2d = TEST_MATRIX_2D + image_processing.RGB_PIXEL_MAX_VALUE // 2
    lighten_img_3d = image_processing.lighten_img(TEST_MATRIX_3D, 50)
    lighten_img_target_3d = repeat_2d_image_array_into_3d(lighten_img_target_2d)
    assert_equal(lighten_img_2d, lighten_img_target_2d)
    assert_equal(lighten_img_3d, lighten_img_target_3d)


def test_darken_img():
    darken_img_2d = image_processing.darken_img(TEST_MATRIX_2D, 100)
    darken_img_target_2d = np.zeros(TEST_MATRIX_2D.shape)
    darken_img_3d = image_processing.darken_img(TEST_MATRIX_3D, 100)
    darken_img_target_3d = repeat_2d_image_array_into_3d(darken_img_target_2d)
    assert_equal(darken_img_2d, darken_img_target_2d)
    assert_equal(darken_img_3d, darken_img_target_3d)


def apply_filter_for_single_index_manually(img: np.array, kernel: np.array, y: int, x: int) -> int:
    filtered_value = 0
    img_h, img_w = img.shape[0:2]
    kernel_h, kernel_w = kernel.shape[0:2]
    # Manual for loop, only for testing (same as manually counting every pixel)
    for i in range(kernel_h):
        for j in range(kernel_w):
            img_y = y + i - kernel_h // 2
            img_x = x + j - kernel_w // 2
            if img_x < 0 or img_y < 0 or img_x >= img_w or img_y >= img_h:
                continue
            filtered_value += img[img_y, img_x] * kernel[i, j]
    filtered_value = 255 if filtered_value > 255 else filtered_value
    filtered_value = 0 if filtered_value < 0 else filtered_value
    return int(filtered_value)


def test_apply_filter():
    # Testing that numbers on sides will be actualized as if missing values were 0
    new_filter = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sharpen_img_target_1_1 = apply_filter_for_single_index_manually(TEST_MATRIX_2D,
                                                                    new_filter, 0, 0)
    sharpen_img_2d = image_processing.apply_filter(TEST_MATRIX_2D, new_filter)
    sharpen_img_3d = image_processing.apply_filter(TEST_MATRIX_3D, new_filter)
    assert sharpen_img_2d[0, 0] == sharpen_img_target_1_1
    assert sharpen_img_3d[0, 0, 0] == sharpen_img_target_1_1


def test_sharpen_img():
    sharpen_img_2d = image_processing.sharpen_img(TEST_MATRIX_2D)
    sharpen_img_target_1_1 = apply_filter_for_single_index_manually(TEST_MATRIX_2D,
                                                                    image_processing.SHARPEN_KERNEL, 1, 1)
    sharpen_img_3d = image_processing.sharpen_img(TEST_MATRIX_3D)
    assert sharpen_img_2d[1, 1] == sharpen_img_target_1_1
    assert sharpen_img_3d[1, 1, 0] == sharpen_img_target_1_1


# EXTENSION METHODS #
def test_blur_img():
    print(image_processing.BLUR_KERNEL)
    blurred_img_2d = image_processing.blur_img(TEST_MATRIX_2D)
    # Blur filter is bigger than whole matrix, but other pixels count as 0.
    blurred_img_target_1_1 = apply_filter_for_single_index_manually(TEST_MATRIX_2D,
                                                                    image_processing.BLUR_KERNEL, 1, 1)
    blurred_img_3d = image_processing.blur_img(TEST_MATRIX_3D)
    assert blurred_img_2d[1, 1] == blurred_img_target_1_1
    assert blurred_img_3d[1, 1, 0] == blurred_img_target_1_1


def test_reduce_color_components():
    single_color_img_3d = image_processing.reduce_color_components(TEST_MATRIX_3D,
                                                                   [image_processing.Color.RED], 100)
    assert_equal(single_color_img_3d[:, :, 0], np.zeros(TEST_MATRIX_2D.shape))


def test_get_single_color_img():
    single_color_img_3d = image_processing.get_single_color_img(TEST_MATRIX_3D,
                                                                image_processing.Color.RED)
    assert_equal(single_color_img_3d[:, :, 1], np.zeros(TEST_MATRIX_2D.shape))
    assert_equal(single_color_img_3d[:, :, 2], np.zeros(TEST_MATRIX_2D.shape))


def test_reduce_color_variety(image):
    target_colors = 3
    reduced_image = image_processing.reduce_color_variety(TEST_MATRIX_3D, target_colors)
    # Testing if only 3 colors left
    assert len(np.unique(reduced_image)) == target_colors


def test_add_gaussian_noise_img():
    gaussian_noise_img_2d = image_processing.add_gaussian_noise_img(TEST_MATRIX_2D, 50)
    gaussian_noise_img_3d = image_processing.add_gaussian_noise_img(TEST_MATRIX_3D, 50)
    assert not np.array_equal(gaussian_noise_img_2d, TEST_MATRIX_2D)
    assert not np.array_equal(gaussian_noise_img_3d, TEST_MATRIX_3D)


def test_crop_image():
    cropped_img_2d = image_processing.crop_image(TEST_MATRIX_2D, 2, 2, 1, 1)
    cropped_img_target_2d = np.array([[5, 6],
                                      [8, 9]])
    cropped_img_3d = image_processing.crop_image(TEST_MATRIX_3D, 2, 2, 1, 1)
    cropped_img_target_3d = repeat_2d_image_array_into_3d(cropped_img_target_2d)
    assert_equal(cropped_img_2d, cropped_img_target_2d)
    assert_equal(cropped_img_3d, cropped_img_target_3d)


def test_crop_image_predefined():
    cropped_img_2d = image_processing.crop_image_predefined(TEST_MATRIX_2D, 2, 2,
                                                            image_processing.VerticalPosition.CENTER,
                                                            image_processing.HorizontalPosition.RIGHT)
    cropped_img_target_2d = np.array([[5, 6],
                                      [8, 9]])
    cropped_img_3d = image_processing.crop_image_predefined(TEST_MATRIX_3D, 2, 2,
                                                            image_processing.VerticalPosition.CENTER,
                                                            image_processing.HorizontalPosition.RIGHT)
    cropped_img_target_3d = repeat_2d_image_array_into_3d(cropped_img_target_2d)
    assert_equal(cropped_img_2d, cropped_img_target_2d)
    assert_equal(cropped_img_3d, cropped_img_target_3d)
