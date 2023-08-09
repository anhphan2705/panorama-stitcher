import cv2
import glob
import time
import argparse


class InvalidDirectoryError(Exception):
    pass


class StitchingFailedError(Exception):
    pass


class SalvageFailedError(Exception):
    pass


def show_image(header, image):
    """
    Displays an image in a window with the specified header.

    Args:
        header (str): Header for the image window.
        image (numpy.ndarray): Image array to be displayed.
    """
    print("[Console] Showing image")
    cv2.imshow(header, image)
    cv2.waitKey()


def write_image(directory, image):
    """
    Saves an image to the specified directory.

    Args:
        directory (str): Path to the directory where the image will be saved.
        image (numpy.ndarray): Image array to be saved.
    """
    print("[Console] Saving image")
    cv2.imwrite(directory, image)


def get_images(directory):
    """
    Loads images from the specified directory.

    Args:
        directory (str): Path to the directory containing the images.

    Returns:
        List of loaded images (list of numpy.ndarray).

    Raises:
        InvalidDirectoryError: If the directory is invalid or no images are found.
    """
    try:
        print("[Console] Accessing folder")
        image_paths = glob.glob(directory)
        print(image_paths)
        if len(image_paths) == 0:
            raise InvalidDirectoryError("[ERROR] Invalid directory or no images found.")
        images = []
        # Add images to memory
        print("[Console] Loading Images")
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Unable to load image: {image_path}")
            else:
                images.append(image)
        print(f"[INFO] Loaded {len(images)} image(s)")
        return images
    except Exception as e:
        print(str(e))
        raise e


def get_gray_image(image):
    """
    Converts an RGB image to grayscale.

    Args:
        image (numpy.ndarray): RGB image array.

    Returns:
        Grayscale image (numpy.ndarray).
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_stitch_image(image_list):
    """
    Stitches a list of images together to create a panoramic image.

    Args:
        image_list (list): List of images to be stitched.

    Returns:
        Stitched image (numpy.ndarray).

    Raises:
        Exception: If stitching fails.
    """
    print("[Console] Stitching Images")
    # Stitch images
    since = time.time()
    stitcher = cv2.Stitcher.create()
    stitch_status, stitched_image = stitcher.stitch(image_list)
    elapsed = time.time() - since
    # Check stitching status
    if stitch_status == 0:
        print(f"[INFO] Stitch successful in {elapsed:2f}s")
        return stitched_image
    else:
        raise Exception("[INFO] Stitch failed")


def get_threshold_image(gray_image):
    """
    Applies thresholding to a grayscale image.

    Args:
        gray_image (numpy.ndarray): Grayscale image array.

    Returns:
        Thresholded image (numpy.ndarray).
    """
    print("[Console] Thresholding image")
    return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)[1]


def get_image_2D_dim(image):
    """
    Returns the dimensions (height and width) of an image.

    Args:
        image (numpy.ndarray): Image array.

    Returns:
        Tuple containing the height and width of the image.
    """
    return image.shape[:2]


def get_mask_image(image):
    """
    Creates a mask image by converting the input image to grayscale, applying thresholding, and blurring.

    Args:
        image (numpy.ndarray): Image array.

    Returns:
        Mask image (numpy.ndarray).
    """
    print("[Console] Masking image")
    gray_image = get_gray_image(image)
    # Threshold + Blur + Threshold = Remove all the random black pixel in the white part of the first threshold
    threshold_image = get_threshold_image(gray_image)
    threshold_image = cv2.GaussianBlur(threshold_image, (5, 5), 0)
    threshold_image = get_threshold_image(threshold_image)
    return threshold_image


def crop_image(image, factor):
    """
    Crops an image inward proportionally based on the specified factor.

    Args:
        image (numpy.ndarray): Image array to be cropped.
        factor (float): Proportion of the image to be cropped (0.0 to 1.0).

    Returns:
        Tuple containing the crop location (lower height, upper height, left width, right width)
        and the cropped image (numpy.ndarray).
    """
    (h, w) = get_image_2D_dim(image)
    # Crop horizontally (width)
    amount_crop = w * (1 - factor)
    w_right = int(w - amount_crop // 2)
    w_left = int(amount_crop // 2)
    # Crop vertically (height)
    amount_crop = h * (1 - factor)
    h_upper = int(h - amount_crop // 2)
    h_lower = int(amount_crop // 2)
    return (h_lower, h_upper, w_left, w_right), image[h_lower:h_upper, w_left:w_right]


def is_black_ver_line(image, start_h, end_h, w):
    """
    Checks if there is a black pixel in a straight vertical line within the specified image region.

    Args:
        image (numpy.ndarray): Image array.
        start_h (int): Starting height of the region.
        end_h (int): Ending height of the region.
        w (int): Width of the vertical line.

    Returns:
        True if a black pixel is found, False otherwise.
    """
    for value in range(start_h, end_h):
        if all(image[value, w] == [0, 0, 0]):
            return True
    return False


def is_black_hor_line(image, start_w, end_w, h):
    """
    Checks if there is a black pixel in a straight horizontal line within the specified image region.

    Args:
        image (numpy.ndarray): Image array.
        start_w (int): Starting width of the region.
        end_w (int): Ending width of the region.
        h (int): Height of the horizontal line.

    Returns:
        True if a black pixel is found, False otherwise.
    """
    for value in range(start_w, end_w):
        if all(image[h, value] == [0, 0, 0]):
            return True
    return False


def is_black_pixel_outline(threshold_image):
    """
    Checks if there are black pixels on the four sides of the thresholded image.

    Args:
        threshold_image (numpy.ndarray): Thresholded image array.

    Returns:
        True if black pixels are found on the outline, False otherwise.
    """
    (height, width) = get_image_2D_dim(threshold_image)
    # Lower side (0, w)
    if is_black_hor_line(threshold_image, 0, width, 0):
        return True
    # Upper side (h, w)
    if is_black_hor_line(threshold_image, 0, width, height - 1):
        return True
    # Left side (h, 0)
    if is_black_ver_line(threshold_image, 0, height, 0):
        return True
    # Right side (h, w)
    if is_black_ver_line(threshold_image, 0, height, width - 1):
        return True
    return False


def expand_from_crop_image(image, crop_location):
    """
    Expands the cropped image by searching for the nearest black pixels on each side.

    Args:
        image (numpy.ndarray): Image array to be expanded.
        crop_location (tuple): Crop location (lower height, upper height, left width, right width).

    Returns:
        Tuple containing the expanded location and the expanded image (numpy.ndarray).
    """
    print("[Console] Salvaging usable cropped portions")
    since = time.time()
    height, width = get_image_2D_dim(image)
    h_lower, h_upper, w_left, w_right = crop_location
    mask_img = get_mask_image(image)
    # Left side (h, 0)
    for w in range(w_left, -1, -1):
        if is_black_ver_line(mask_img, h_lower, h_upper, w):
            w_left = w + 5
            break
    # Right side (h, w)
    for w in range(w_right, width):
        if is_black_ver_line(mask_img, h_lower, h_upper, w):
            w_right = w - 5
            break
    # Lower side (0, w)
    for h in range(h_lower, -1, -1):
        if is_black_hor_line(mask_img, w_left, w_right, h):
            h_lower = h + 5
            break
    # Upper side (w, 0)
    for h in range(h_upper, height):
        if is_black_hor_line(mask_img, w_left, w_right, h):
            h_upper = h - 5
            break
    if crop_location is not (h_lower, h_upper, w_left, w_right):
        elapsed = time.time() - since
        print(f"[INFO] Salvaging usable image portion success in {elapsed:2f}s")
        return (h_lower, h_upper, w_left, w_right), image[
            h_lower:h_upper, w_left:w_right
        ]
    else:
        print("[INFO] Salvage failed")
        return (None, None)


def remove_black_outline(image):
    """
    Crops the image inward proportionally until all the black pixel outlines are removed.

    Args:
        image (numpy.ndarray): Image array to be processed.

    Returns:
        Tuple containing the crop location and the cropped image (numpy.ndarray).
    """
    print("[Console] Cropping Image")
    since = time.time()
    mask = get_mask_image(image)
    # Cropping image
    is_cropped = False
    for crop_factor in range(100, -1, -1):
        crop_factor = 0.01 * crop_factor
        trial_mask = crop_image(mask, crop_factor)[1]
        if not is_black_pixel_outline(trial_mask):
            print(f"[Console] Crop image with factor of {crop_factor}")
            is_cropped = True
            break
    elapsed = time.time() - since
    # Showing result
    if is_cropped:
        print(f"[INFO] Crop successful in {elapsed:2f}s")
        return crop_image(image, crop_factor)
    else:
        print("[INFO] Image is not suitable to be cropped")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program will stitch multiple images into one big panorama."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="(String) Path to a folder that holds a sequence of images.",
        default="./images/real",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="(String) Path to folder where outputs should be written to",
        default="./output",
    )
    parser.add_argument(
        "-c",
        "--crop",
        type=bool,
        help="(bool) False to turn of cropping, True otherwise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    INPUT_PATH = args.input
    OUT_PATH = args.output
    IS_CROP = args.crop

    main_since = time.time()
    images = get_images(INPUT_PATH + "/*")
    stitched_image = get_stitch_image(images)

    if IS_CROP:
        crop_location, cropped_image = remove_black_outline(stitched_image)
        expand_location, expanded_img = expand_from_crop_image(stitched_image, crop_location)
        main_elapsed = time.time() - main_since
        if expanded_img is not None:
            print(f"[INFO] Done in {main_elapsed:2f}s")
            write_image(OUT_PATH + "/pano.jpg", expanded_img)
            show_image("Pano", expanded_img)
        else:
            print("Something went wrong")
    else:
        main_elapsed = time.time() - main_since
        print(f"[INFO] Done in {main_elapsed:2f}s")
        write_image(OUT_PATH + "/stitched.jpg", stitched_image)
        show_image("Stitched", stitched_image)
