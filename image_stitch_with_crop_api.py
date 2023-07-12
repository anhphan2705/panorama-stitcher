from fastapi import FastAPI, UploadFile, Response
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()


def convert_byte_to_arr(byte_image):
    """
    Converts a byte image to a NumPy array.

    Args:
        byte_image (bytes): The byte representation of the image.

    Returns:
        numpy.ndarray: The NumPy array representation of the image.
    """
    arr_image = np.array(Image.open(BytesIO(byte_image)))
    return arr_image


def convert_arr_to_byte(arr_image):
    """
    Converts a NumPy array to a byte image.

    Args:
        arr_image (numpy.ndarray): The NumPy array representation of the image.

    Returns:
        bytes: The byte representation of the image.
    """
    arr_image_cvt = cv2.cvtColor(arr_image, cv2.COLOR_RGB2BGR)
    success, byte_image = cv2.imencode(".jpg", arr_image_cvt)
    if success:
        return byte_image.tobytes()
    else:
        raise Exception("Cannot convert array image to byte image")


def show_image(header, image):
    """
    Displays an image using OpenCV.

    Args:
        header (str): The header/title of the image window.
        image (numpy.ndarray): The image to be displayed.
    """
    cv2.imshow(header, image)
    cv2.waitKey()


def write_image(directory, image):
    """
    Saves an image to disk.

    Args:
        directory (str): The directory path to save the image.
        image (numpy.ndarray): The image to be saved.
    """
    cv2.imwrite(directory, image)


def get_gray_image(image):
    """
    Converts an RGB image to grayscale.

    Args:
        image (numpy.ndarray): The RGB image.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_stitch_image(image_list):
    """
    Stitches a list of images together.

    Args:
        image_list (List[numpy.ndarray]): List of images to be stitched.

    Returns:
        numpy.ndarray: The stitched image.
    Raises:
        Exception: If stitching fails.
    """
    stitcher = cv2.Stitcher.create()
    stitch_status, stitched_image = stitcher.stitch(image_list)
    if stitch_status == 0:
        return stitched_image
    else:
        raise Exception("Stitch failed")


def get_threshold_image(gray_image):
    """
    Applies a threshold to convert a grayscale image to binary.

    Args:
        gray_image (numpy.ndarray): The grayscale image.

    Returns:
        numpy.ndarray: The thresholded binary image.
    """
    return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)[1]


def get_image_2D_dim(image):
    """
    Gets the dimensions of an image.

    Args:
        image (numpy.ndarray): The image.

    Returns:
        Tuple[int, int]: The height and width of the image.
    """
    return image.shape[:2]


def get_mask_image(image):
    """
    Creates a binary mask image from an RGB image.

    Args:
        image (numpy.ndarray): The RGB image.

    Returns:
        numpy.ndarray: The binary mask image.
    """
    gray_image = get_gray_image(image)
    threshold_image = get_threshold_image(gray_image)
    return threshold_image


def crop_image(image, factor):
    """
    Crops an image based on a given factor.

    Args:
        image (numpy.ndarray): The image to be cropped.
        factor (float): The crop factor.

    Returns:
        Tuple[int, int, int, int], numpy.ndarray: The crop location and the cropped image.
    """
    (h, w) = get_image_2D_dim(image)
    amount_crop = w * (1 - factor)
    w_right = int(w - amount_crop // 2)
    w_left = int(amount_crop // 2)
    amount_crop = h * (1 - factor)
    h_upper = int(h - amount_crop // 2)
    h_lower = int(amount_crop // 2)
    return (h_lower, h_upper, w_left, w_right), image[h_lower:h_upper, w_left:w_right]


def is_black_ver_line(image, start_h, end_h, w):
    """
    Checks if there is a black pixel in a straight vertical line.

    Args:
        image (numpy.ndarray): The image.
        start_h (int): The starting height of the line.
        end_h (int): The ending height of the line.
        w (int): The width of the line.

    Returns:
        bool: True if a black pixel is found, False otherwise.
    """
    for value in range(start_h, end_h):
        if all(image[value, w] == [0, 0, 0]):
            return True
    return False


def is_black_hor_line(image, start_w, end_w, h):
    """
    Checks if there is a black pixel in a straight horizontal line.

    Args:
        image (numpy.ndarray): The image.
        start_w (int): The starting width of the line.
        end_w (int): The ending width of the line.
        h (int): The height of the line.

    Returns:
        bool: True if a black pixel is found, False otherwise.
    """
    for value in range(start_w, end_w):
        if all(image[h, value] == [0, 0, 0]):
            return True
    return False


def is_black_pixel_outline(threshold_image):
    """
    Finds if there are black pixels on the four sides of an image.

    Args:
        threshold_image (numpy.ndarray): The binary threshold image.

    Returns:
        bool: True if black pixels are found on the outline, False otherwise.
    """
    (height, width) = get_image_2D_dim(threshold_image)
    if is_black_hor_line(threshold_image, 0, width, 0):
        return True
    if is_black_hor_line(threshold_image, 0, width, height - 1):
        return True
    if is_black_ver_line(threshold_image, 0, height, 0):
        return True
    if is_black_ver_line(threshold_image, 0, height, width - 1):
        return True
    return False


def expand_from_crop_image(image, crop_location):
    """
    Expands the cropped image until it hits a black pixel.

    Args:
        image (numpy.ndarray): The image to be expanded.
        crop_location (Tuple[int, int, int, int]): The crop location.

    Returns:
        Tuple[int, int, int, int], numpy.ndarray: The expanded crop location and the expanded image.
    """
    height, width = get_image_2D_dim(image)
    h_lower, h_upper, w_left, w_right = crop_location
    mask_img = get_mask_image(image)
    for w in range(w_left, -1, -1):
        if is_black_ver_line(mask_img, h_lower, h_upper, w):
            w_left = w + 5
            break
    for w in range(w_right, width):
        if is_black_ver_line(mask_img, h_lower, h_upper, w):
            w_right = w - 5
            break
    for h in range(h_lower, -1, -1):
        if is_black_hor_line(mask_img, w_left, w_right, h):
            h_lower = h + 5
            break
    for h in range(h_upper, height):
        if is_black_hor_line(mask_img, w_left, w_right, h):
            h_upper = h - 5
            break
    if crop_location != (h_lower, h_upper, w_left, w_right):
        return (h_lower, h_upper, w_left, w_right), image[
            h_lower:h_upper, w_left:w_right
        ]
    else:
        return (None, None)


def remove_black_outline(image):
    """
    Removes the black outline from an image.

    Args:
        image (numpy.ndarray): The image to remove the black outline.

    Returns:
        Tuple[int, int, int, int], numpy.ndarray: The crop location and the cropped image without black outline.
    """
    mask = get_mask_image(image)
    is_cropped = False
    for crop_factor in range(100, -1, -1):
        crop_factor = 0.01 * crop_factor
        trial_mask = crop_image(mask, crop_factor)[1]
        if not is_black_pixel_outline(trial_mask):
            is_cropped = True
            break
    if is_cropped:
        return crop_image(image, crop_factor)
    else:
        return None


@app.get("/")
def welcome_page():
    """
    Serves the root route ("/") and displays a welcome message with a link to the API documentation.
    """
    return HTMLResponse(
        """
        <h1>Welcome to Banana</h1>
        <p>Click the button below to go to /docs/:</p>
        <form action="/docs" method="get">
            <button type="submit">Visit Website</button>
        </form>
    """
    )


@app.post("/stitch_image")
async def stitch_app(in_images: list[UploadFile]):
    """
    API endpoint to stitch multiple images together.

    Args:
        in_images (List[UploadFile]): List of images to be stitched.

    Returns:
        Response: The stitched image as a response.

    Raises:
        Exception: If stitching fails or the image cannot be cropped.
    """
    images = []
    for in_image in in_images:
        byte_image = await in_image.read()
        arr_image = convert_byte_to_arr(byte_image)
        images.append(arr_image)

    stitched_image = get_stitch_image(images)
    crop_location, cropped_image = remove_black_outline(stitched_image)

    if cropped_image is not None:
        expand_location, expanded_img = expand_from_crop_image(
            stitched_image, crop_location
        )
        if expanded_img is not None:
            byte_stitched_image = convert_arr_to_byte(expanded_img)
            return Response(byte_stitched_image, media_type="image/jpg")

        byte_stitched_image = convert_arr_to_byte(cropped_image)
        return Response(byte_stitched_image, media_type="image/jpg")

    byte_stitched_image = convert_arr_to_byte(stitched_image)
    return Response(byte_stitched_image, media_type="image/jpg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
