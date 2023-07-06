import cv2
import glob

def show_image(header, image):
    print("[CONSOLE] Showing image")
    cv2.imshow(header, image)
    cv2.waitKey()
    
def write_image(directory, image):
    print("[CONSOLE] Saving image")
    cv2.imwrite(directory, image)

def get_images(directory):
    print("[CONSOLE] Accessing folder")
    image_paths = glob.glob(directory)
    print(image_paths)
    if len(image_paths) == 0:
        raise Exception("[CONSOLE] Invalid directory")
    images = []
    # Add image to memory
    print("[CONSOLE] Loading Images")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        images.append(image)
    print(f"[CONSOLE] Loaded {len(images)} image(s)")
    return images

def get_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def get_stitch_image(image_list):
    print("[CONSOLE] Stitching Images")
    # Stitch
    stitcher = cv2.Stitcher.create()
    stitch_status, stitched_image = stitcher.stitch(image_list)
    # Output
    if stitch_status == 0:
        print("[CONSOLE] Stitch successfully")
        return stitched_image
    else:
        raise Exception("[Console] Stitch failed")
    
def get_threshold_image(gray_image):
    print("[CONSOLE] Thresholding image")
    return cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)[1]

def get_image_2D_dim(image):
    return image.shape[:2]

def get_mask_image(image):
    print("[CONSOLE] Masking image")
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

def expand_from_crop_image(image, cropped_image, crop_location):
    # Assuming it is proportionally cropped, expand each side until it hit a black pixel
    # This will get the most image info in the expense of original image ratio
    print("[CONSOLE] Salvaging usable cropped portions")
    height, width = get_image_2D_dim(cropped_image)
    print(get_image_2D_dim(cropped_image))
    h_lower, h_upper, w_left, w_right = crop_location
    print(crop_location)
    mask_img = get_mask_image(image)
    # Left side (h, 0)
    for w in range(w_left, -1, -1):
        if is_black_ver_line(mask_img, h_lower, h_upper, w):
            w_left = w-1
            print("w1", w)
            break
    # Right side (h, w)
    for w in range(w_right, width):
        if is_black_ver_line(mask_img, h_lower, h_upper, w): 
            w_right = w-1
            print("w2", w)
            break
    # Lower side (0, w)
    for h in range(h_lower, -1, -1):
        if is_black_hor_line(mask_img, w_left, w_right, h): 
            h_lower = h-1
            print("h1", h)
            break
    # Upper side (w, 0)
    for h in range(h_upper, height):
        if is_black_hor_line(mask_img, w_left, w_right, h): 
            h_upper = h-1
            print("h2", h)
            break
    print(h_lower, h_upper, w_left, w_right)
    return (h_lower, h_upper, w_left, w_right), image[h_lower:h_upper, w_left:w_right]

    
def remove_black_outline(image):
    print("[CONSOLE] Cropping Image")
    mask = get_mask_image(image)
    # Cropping image
    is_cropped = False
    for crop_factor in range(100, -1, -1):
        crop_factor = 0.01 * crop_factor
        trial_mask = crop_image(mask, crop_factor)[1]
        if not is_black_pixel_outline(trial_mask):
            print(f"[CONSOLE] Crop image with factor of {crop_factor}")
            is_cropped = True
            break
    # Showing result
    if is_cropped:
        print("[CONSOLE] Crop successfully")
        return crop_image(image, crop_factor)
    else:
        print("[CONSOLE] Image is not suitable to be cropped")
        return None

# Main
images = get_images("./images/real/*.jpg")
stitched_image = get_stitch_image(images)
crop_location, cropped_image = remove_black_outline(stitched_image)
expand_location, expanded_img = expand_from_crop_image(stitched_image, cropped_image, crop_location)
# Output
print(get_image_2D_dim(expanded_img))
show_image("Product", expanded_img)
write_image("./output/stitched_img.jpg", stitched_image)
write_image("./output/cropped_img.jpg", cropped_image)
write_image("./output/expanded_img.jpg", expanded_img)


