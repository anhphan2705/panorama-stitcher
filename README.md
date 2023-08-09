# Panorama Stitcher

## Introduction

The Pano Stitcher script is a Python-based utility that enables the seamless stitching of a sequence of images to create a panoramic view. It utilizes the OpenCV library for image processing and provides optional cropping and salvaging functionalities to enhance the final stitched result. It will work even with unorganized and duplicated images.

## Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [API Usage](#api-usage)
- [Script Usage](#script-usage)
- [Options](#options)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- Stitch multiple images together to create a panoramic view.
- Optional cropping to remove black pixel outlines and enhance the visual quality.
- Salvaging usable cropped portions to optimize the final stitched result.
- Command-line interface for easy execution and customization.
- Utilizes OpenCV for efficient image processing.
- Include API for ease of use

## Requirements

- Python 3.x
- OpenCV (cv2) library
- Numpy library
- argparse library
- FastAPI library
- Pillow library

## Installation

1. Install the required libraries using the following command:

```bash
pip install opencv-python numpy argparse
```

2. If you are using API, please install these extras libraries using the following command:

```bash
pip install fastapi uvicorn pillow
```

## API Usage

1. Navigate to the API directory.

2. Launch the FastAPI app by executing the following command:

```bash
uvicorn pano_stitcher_api:app --host 0.0.0.0 --port 8000
```

3. Access the FastAPI documentation by navigating to `http://localhost:8000` in your web browser.

4. Utilize the `/stitch_image` endpoint to upload images for stitching and processing. The script will return the stitched and processed image.

5. You can share the API with someone else on the same network using `http://your-ipv4:8000` with your own IPv4

## Script Usage

1. Clone or download this repository to your local machine.
2. Organize your input images in the same folder.
3. Navigate to the script directory.
4. Run the script with the following command:

```bash
python pano_stitcher.py -i 'input_folder_path' -o 'output_folder_path'
```

5. To see more option with cropping, use

```bash
python pano_stitcher.py -h
```

## Options

- `-h`, `--help`: Show help message and exit.
- `-i INPUT`, `--input INPUT`: Path to the folder containing the input images. Default: `./images/real`.
- `-o OUTPUT`, `--output OUTPUT`: Path to the folder where output images should be saved. Default: `./output`.
- `-c`, `--crop`: Use this flag to enable cropping and salvaging. By default, cropping is enabled.
- `--no-crop`: Use this flag to disable cropping and salvaging. By default, cropping is enabled.

## Examples

1. Basic Stitching:
  
```bash
python pano_stitcher.py -i './input_images' -o './output_images'
```

2. Stitching with Cropping Disabled:

```bash
python pano_stitcher.py -i './input_images' -o './output_images' --no-crop
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- OpenCV: https://opencv.org/
- FastAPI: https://fastapi.tiangolo.com/