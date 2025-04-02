import cv2
import numpy as np
import pickle
import os

# Global variables
current_image = 0
current_layer = 0
point_list = []
layers = {}
flip_sequence = ''
threshold = 100

# Load image and initialize variables
fn = r"C:\Projects\retina\30mM gluc 660 nm 60xw retina2 stack1 parte 1.tiff"
zoom_factor = 0.97
layer_limits = ['GCL', 'IPL']
layer_limits = list(np.array([[x, x + '_exclude'] for x in layer_limits] + [['Save', '*']]).flatten())
img = cv2.imread(fn)
img = cv2.resize(img, (0, 0), fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LANCZOS4)
red, green, blue = cv2.split(img)
images = [img, red, green, blue]
original_images = images.copy()  # Save a copy of the original images

if os.path.exists(fn.replace('.tif', '_layers.pkl')):
    with open(fn.replace('.tif', '_layers.pkl'), 'rb') as f:
        layers = pickle.load(f)

# Mouse click callback function
def click(event, x, y, flags, param):
    if layer_limits[current_layer] not in ('*', 'Save'):
        if event == cv2.EVENT_LBUTTONDOWN:
            point_list.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and point_list:
            point_list.pop()

# Threshold update function
def update_threshold(val):
    global threshold
    threshold = val

# Refresh display function
def refresh_display():
    display = images[current_image].copy()
    if point_list:
        for point in point_list:
            cv2.circle(display, tuple(point), 5, (255, 255, 255))
        for i in range(len(point_list) - 1):
            cv2.line(display, tuple(point_list[i]), tuple(point_list[i + 1]), (255, 255, 255))
    for layer in layers:
        if 'exclude' in layer:
            for excluded_area in layers[layer]:
                display = cv2.drawContours(display, [excluded_area], 0, (255, 255, 255), 2)
        else:
            cv2.putText(display, layer, (0, layers[layer][0][1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255))
            display = cv2.drawContours(display, [layers[layer]], 0, (255, 255, 255), 2)
    cv2.putText(display, layer_limits[current_layer], (0, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
    cv2.imshow('image', display)

# Initialize OpenCV windows and trackbars
cv2.namedWindow("image")
cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
cv2.resizeWindow("threshold", 300, 60)  # Adjust the size of the threshold window to fit the trackbar
cv2.setMouseCallback("image", click)
cv2.createTrackbar('Threshold', 'threshold', 0, 255, update_threshold)
cv2.setTrackbarPos('Threshold', 'threshold', threshold)

# Position the threshold window to the right of the image window
cv2.moveWindow("threshold", img.shape[1], 0)

def handle_keys():
    global current_image, current_layer, layers, images

    k = cv2.waitKey(1)
    if k == 27:  # ESC key to end
        return False
    elif k == 96:  # Key `, toggle channels
        current_image = (current_image + 1) % 4
    elif 49 <= k <= 52:  # Keys 1-4, show each channel
        current_image = k - 49
    elif k == 13:  # ENTER key, accept layer boundary
        handle_enter_key()
    elif k == 8:  # BACKSPACE key, accept excluded area
        handle_backspace_key()
    elif k == 32:  # SPACE key, apply threshold and count pixels
        apply_threshold_and_count_pixels()
    elif k == 45:  # Key -, clear layer data
        layers = {}
    elif k == 43:  # Key +, reload layer data
        reload_layer_data()
    elif k == 100:  # Key d, define layers
        define_layers()
    elif k == 56:  # Key 8, flip vertical
        flip_vertical()
    elif k == 57:  # Key 9, flip 90º clockwise
        flip_90_clockwise()
    elif k == 55:  # Key 7, flip 90º counter-clockwise
        flip_90_counter_clockwise()
    elif k == 54:  # Key 6, undo threshold
        undo_threshold()
    else:
        if k != -1:  # No key
            print(f'Unused key {k}')
    return True

def handle_enter_key():
    global current_layer, layers, point_list
    if layer_limits[current_layer] == 'Save':
        with open(fn.replace('.tif', '_layers.pkl'), 'wb') as f:
            pickle.dump(layers, f)
        current_layer = (current_layer + 1) % len(layer_limits)
    elif layer_limits[current_layer] != '*':
        if 'exclude' not in layer_limits[current_layer]:
            print(f"Setting layer {layer_limits[current_layer]} with points: {point_list}")
            layers[layer_limits[current_layer]] = np.array(point_list)
        point_list.clear()
        current_layer = (current_layer + 1) % len(layer_limits)

def handle_backspace_key():
    global current_layer, layers, point_list
    if 'exclude' in layer_limits[current_layer]:
        if layer_limits[current_layer] not in layers:
            layers[layer_limits[current_layer]] = []
        print(f"Setting exclude layer {layer_limits[current_layer]} with points: {point_list}")
        layers[layer_limits[current_layer]].append(np.array(point_list))
        point_list.clear()

def apply_threshold_and_count_pixels():
    global images
    results = []
    for layer in layers:
        if 'exclude' not in layer:
            mask = np.zeros(images[current_image].shape, dtype=np.uint8)
            mask = cv2.fillPoly(mask, [layers[layer]], 1)
            if layer + '_exclude' in layers:
                for excluded in layers[layer + '_exclude']:
                    mask = cv2.fillPoly(mask, [excluded], 0)
            total_pixels = np.sum(mask) / zoom_factor
            img_thresh = cv2.threshold(images[current_image], threshold, 1, cv2.THRESH_BINARY)[1]
            mask_thresh = mask * img_thresh
            thresh_pixels = np.sum(mask_thresh) / zoom_factor
            result = [layer, total_pixels, thresh_pixels]
            results.append('\t'.join(map(str, result)))
    images[current_image] = img_thresh * 255
    print('\n'.join(results))
    with open(fn.replace('.tif', '_ch' + str(current_image) + '_results.txt'), 'w') as outfile:
        outfile.write('\n'.join(results))

def reload_layer_data():
    global layers
    if os.path.exists(fn.replace('.tif', '_layers.pkl')):
        with open(fn.replace('.tif', '_layers.pkl'), 'rb') as f:
            layers = pickle.load(f)

def define_layers():
    global layers, current_layer
    layers = {}
    current_layer = 0

def flip_vertical():
    global images, flip_sequence
    images[current_image] = cv2.flip(images[current_image], 0)
    flip_sequence += 'V'
    with open(fn.replace('.tif', '_flips.txt'), 'w') as f:
        f.write(flip_sequence)

def flip_90_clockwise():
    global images, flip_sequence
    images[current_image] = cv2.rotate(images[current_image], cv2.ROTATE_90_CLOCKWISE)
    flip_sequence += 'R'
    with open(fn.replace('.tif', '_flips.txt'), 'w') as f:
        f.write(flip_sequence)

def flip_90_counter_clockwise():
    global images, flip_sequence
    images[current_image] = cv2.rotate(images[current_image], cv2.ROTATE_90_COUNTERCLOCKWISE)
    flip_sequence += 'L'
    with open(fn.replace('.tif', '_flips.txt'), 'w') as f:
        f.write(flip_sequence)

def undo_threshold():
    global images
    images = original_images.copy()
    print("Threshold undone.")

while True:
    refresh_display()
    if not handle_keys():
        break

cv2.destroyAllWindows()
