# Image Stitching and Cropping

This Python script performs image stitching, cropping, and expansion operations using OpenCV. It takes a directory of input images, stitches them together to create a panoramic image, removes the black pixel outline around the stitched image, crops the image inward proportionally until all the black pixels are removed, and expands the cropped image to salvage usable portions.

## Prerequisites

- Python 3.x
- OpenCV library

## Installation

1. Clone the repository or download the script files.

2. Install the required dependencies by running the following command:

   ```
   pip install opencv-python
   ```

## Usage

1. Place your input images in the `images/*/` directory.

2. Open the Python script `image_stitch_wtih_crop_source.py` and modify the following line to specify the input image directory:

   ```python
   images = get_images("./images/*/*.jpg")
   ```

3. Run the script using the following command:

   ```
   python ./image_stitch_wtih_crop_source.py
   ```

4. The stitched image will be saved as `output/stitched_img.jpg` and displayed on the screen.

5. If the black pixel outline removal is successful, the cropped image will be saved as `output/cropped_img.jpg` and displayed on the screen.

6. If the cropped image expansion is successful, the expanded image will be saved as `output/expanded_img.jpg` and displayed on the screen.

## Code Structure

The code is structured as follows:

- `show_image(header, image)`: Displays an image in a window with the specified header.

- `write_image(directory, image)`: Saves an image to the specified directory.

- `get_images(directory)`: Loads images from the specified directory.

- `get_gray_image(image)`: Converts an RGB image to grayscale.

- `get_stitch_image(image_list)`: Stitches a list of images together to create a panoramic image.

- `get_threshold_image(gray_image)`: Applies thresholding to a grayscale image.

- `get_image_2D_dim(image)`: Returns the dimensions (height and width) of an image.

- `get_mask_image(image)`: Creates a mask image by converting the input image to grayscale, applying thresholding, and blurring.

- `crop_image(image, factor)`: Crops an image inward proportionally based on the specified factor.

- `is_black_ver_line(image, start_h, end_h, w)`: Checks if there is a black pixel in a straight vertical line within the specified image region.

- `is_black_hor_line(image, start_w, end_w, h)`: Checks if there is a black pixel in a straight horizontal line within the specified image region.

- `is_black_pixel_outline(threshold_image)`: Checks if there are black pixels on the four sides of the thresholded image.

- `expand_from_crop_image(image, crop_location)`: Expands the cropped image by searching for the nearest black pixels on each side.

- `remove_black_outline(image)`: Crops the image inward proportionally until all the black pixel outlines are removed.

- Main section: Loads images, performs image stitching, removes black pixel outline, crops the image, and expands the cropped image if necessary. It saves and displays the resulting images.

## Notes

- Make sure the input images are in the proper sequence for accurate stitching.

- The script assumes the input images are in JPEG format. Modify the file extension in the `get_images` function if using a different format.

- The script may take some time to process depending on the number and size of input images.

- Ensure that the input image directory exists and contains valid image files.

- The output images will be saved in the `output/` directory. Make sure the directory exists or modify the save paths in the code if desired.

## Contributing

Contributions are welcome! If you have any suggestions or improvements for this code, please feel free to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- OpenCV: https://opencv.org/