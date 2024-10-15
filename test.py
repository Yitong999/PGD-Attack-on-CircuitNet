import cv2

# Load the image using OpenCV
image = cv2.imread('German/Train/0/00000_00000_00001.png')

# Get the shape of the image
height, width, channels = image.shape

print(f"Image height: {height}")
print(f"Image width: {width}")
print(f"Number of channels (usually 3 for RGB images): {channels}")
