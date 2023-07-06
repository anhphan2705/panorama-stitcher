from fastapi import FastAPI, UploadFile, Response
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

def convert_byte_to_arr(byte_image):
    print("[Console] Converting image from byte to np.arr")
    arr_image = np.array(Image.open(BytesIO(byte_image)))
    return arr_image

def convert_arr_to_byte(arr_image):
    print("[Console] Converting image from np.arr to byte")
    arr_image_cvt = cv2.cvtColor(arr_image, cv2.COLOR_RGB2BGR)
    success, byte_image = cv2.imencode('.jpg', arr_image_cvt)
    return byte_image.tobytes()

def show_image(header, image):
    print("[Console] Showing image")
    cv2.imshow(header, image)
    cv2.waitKey()
    
def write_image(directory, image):
    print("[Console] Saving image")
    cv2.imwrite(directory, image)

def get_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def get_stitch_image(image_list):
    print("[Console] Stitching Images")
    # Stitch
    stitcher = cv2.Stitcher.create()
    stitch_status, stitched_image = stitcher.stitch(image_list)
    # Output
    if stitch_status == 0:
        print("[Console] Stitch successfully")
        return stitched_image
    else:
        raise Exception("[Console] Stitch failed")
    
def get_threshold_image(gray_image):
    print("[Console] Thresholding image")
    return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)[1]

def get_image_2D_dim(image):
    return image.shape[:2]

def get_mask_image(image):
    print("[Console] Masking image")
    gray_image = get_gray_image(image)
    # Threshold + Blur + Threshold = Remove all the random black pixel in the white part of the first threshold
    threshold_image = get_threshold_image(gray_image)
    threshold_image_b = cv2.GaussianBlur(threshold_image, (5, 5), 0)
    threshold_image = get_threshold_image(threshold_image_b)
    return threshold_image

def crop_image(image, factor):
    (h, w) = get_image_2D_dim(image)
    # Crop horizontally (width)
    amount_crop = w * (1-factor)
    w_right = int(w - amount_crop//2)
    w_left = int(amount_crop//2)
    # Crop vertically (height)
    amount_crop = h * (1-factor)
    h_upper = int(h - amount_crop//2)
    h_lower = int(amount_crop//2)
    return (h_lower, h_upper, w_left, w_right), image[h_lower:h_upper, w_left:w_right]

def is_black_ver_line(image, start_h, end_h, w):
    # Check if there is a black pixel in a straight vertical line
    for value in range(start_h, end_h):
        if all(image[value, w] == [0, 0, 0]):
            return True
    return False

def is_black_hor_line(image, start_w, end_w, h):
    # Check if there is a black pixel in a straight horizontal line
    for value in range(start_w, end_w):
        if all(image[h, value] == [0, 0, 0]):
            return True
    return False
        
def is_black_pixel_outline(threshold_image):
    # Find if there is black pixel on 4 sides
    (height, width) = get_image_2D_dim(threshold_image)
    # Lower side (0, w)
    if is_black_hor_line(threshold_image, 0, width, 0): return True
    # Upper side (h, w)
    if is_black_hor_line(threshold_image, 0, width, height-1): return True
    # Left side (h, 0)
    if is_black_ver_line(threshold_image, 0, height, 0): return True
    # Right side (h, w)
    if is_black_ver_line(threshold_image, 0, height, width-1): return True
    return False

def expand_from_crop_image(image, crop_location):
    # Assuming it is proportionally cropped, expand each side until it hit a black pixel
    # This will get the most image info in the expense of original image ratio
    # w +- 5 instead of 1 for any error margin it may have
    print("[Console] Salvaging usable cropped portions")
    height, width = get_image_2D_dim(image)
    h_lower, h_upper, w_left, w_right = crop_location
    mask_img = get_mask_image(image)
    # Left side (h, 0)
    for w in range(w_left, -1, -1):
        if is_black_ver_line(mask_img, h_lower, h_upper, w):
            w_left = w+5
            break
    # Right side (h, w)
    for w in range(w_right, width):
        if is_black_ver_line(mask_img, h_lower, h_upper, w): 
            w_right = w-5
            break
    # Lower side (0, w)
    for h in range(h_lower, -1, -1):
        if is_black_hor_line(mask_img, w_left, w_right, h): 
            h_lower = h+5
            break
    # Upper side (w, 0)
    for h in range(h_upper, height):
        if is_black_hor_line(mask_img, w_left, w_right, h): 
            h_upper = h-5
            break
    if crop_location is not (h_lower, h_upper, w_left, w_right):
        print("[Console] Salvaging usable image portion success")
        return (h_lower, h_upper, w_left, w_right), image[h_lower:h_upper, w_left:w_right]
    else:
        print("[Console] Salvage failed")
        return (None, None)

def remove_black_outline(image):
    print("[Console] Cropping Image")
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
    # Showing result
    if is_cropped:
        print("[Console] Crop successfully")
        return crop_image(image, crop_factor)
    else:
        print("[Console] Image is not suitable to be cropped")
        return None

@app.get("/")
async def home():
    print("hello")

@app.post("/stitch_image")
async def stitch_app(in_images:list[UploadFile]):
    #Add images to process
    images = []
    for in_image in in_images:
        byte_image = await in_image.read()
        arr_image = convert_byte_to_arr(byte_image)
        images.append(arr_image)
    # Stitch
    stitched_image = get_stitch_image(images)
    crop_location, cropped_image = remove_black_outline(stitched_image)
    if cropped_image is not None:
        expand_location, expanded_img = expand_from_crop_image(stitched_image, crop_location)
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