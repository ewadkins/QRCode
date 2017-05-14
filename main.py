import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import sys
import base64
import time

from locate import locate
from calibrate import calibrate
from autofocus import autofocus_step, autofocus_batch

# Displays results
def display(locations, image, filtered, peaks, thresholded, kernel):
    
    image_width = image.shape[1]
    image_height = image.shape[0]
    
    kernel_width = kernel.shape[1]
    kernel_height = kernel.shape[0]

    def draw_rectangles(ax, locs, w, h):
        for i in range(len(locs)):
            x = locs[i][0] - w / 2
            y = locs[i][1] - h / 2
            ax.add_patch(patches.Rectangle((x, y), w, h, lw=1, ec='r', fc='none'))

    f1 = plt.figure(figsize=(16, 4))

    ax1 = f1.add_subplot(151)
    im1 = ax1.imshow(image, cmap='gray')
    draw_rectangles(ax1, locations, kernel_width, kernel_height)
    f1.colorbar(im1)

    ax2 = f1.add_subplot(152)
    im2 = ax2.imshow(filtered, cmap='gray')
    draw_rectangles(ax2, locations, kernel_width, kernel_height)
    f1.colorbar(im2)

    ax3 = f1.add_subplot(153)
    im3 = ax3.imshow(peaks, cmap='gray')
    draw_rectangles(ax3, locations, kernel_width, kernel_height)
    f1.colorbar(im3)

    ax4 = f1.add_subplot(154)
    im4 = ax4.imshow(thresholded, cmap='gray')
    draw_rectangles(ax4, locations, kernel_width, kernel_height)
    f1.colorbar(im4)

    ax5 = f1.add_subplot(155)
    im5 = ax5.imshow(kernel, cmap='gray')
    f1.colorbar(im5)

    def get_crop(loc, w, h):
        return image[max(0, int(loc[1])-h/2) : min(int(loc[1])+h/2, image_height), max(0, int(loc[0])-w/2) : min(int(loc[0])+w/2, image_width)]

    if len(locations):
        f2 = plt.figure(figsize=(3*len(locations), 6))

        plot_height = 2
        plot_width = len(locations)
        for i in range(len(locations)):
            ax = f2.add_subplot(plot_height, plot_width, i + 1)
            ax.imshow(image, cmap='gray')
            draw_rectangles(ax, [locations[i]], kernel_width, kernel_height)
        for i in range(len(locations)):
            ax = f2.add_subplot(plot_height, plot_width, i + len(locations) + 1)
            ax.imshow(get_crop(locations[i], kernel_width, kernel_height), cmap='gray')

    plt.show()

###################################################################################################

# Detection

def _detect(image, kernel_path, calibration_path, show_display):
    # Load kernel
    kernel = ndimage.imread(kernel_path, flatten=True)
    
    # Load calibration settings
    with open(calibration_path) as f:
        content = f.readlines()
        size = int(content[0])
        rotation = float(content[1])

    # Resizing image
    kernel = misc.imresize(kernel, (size, size)).astype(float)

    results, filtered, peaks, thresholded, modified_kernel = locate(kernel, image, rotation=rotation)
    locations = []
    magnitudes = [0]
    if len(results):
        locations, magnitudes = zip(*results)
    if (show_display):
        display(locations, image, filtered, peaks, thresholded, modified_kernel)

###################################################################################################

# Calibration

def _calibrate(image, kernel_path, calibration_path):
    # Load kernel
    kernel = ndimage.imread(kernel_path, flatten=True)
    
    size, rotation = calibrate(kernel, image)
    f = open(calibration_path, 'w')
    f.write(str(size) + '\n' + str(rotation) + '\n')

###################################################################################################

# Autofocus (batch)

def _autofocus_batch(image_directory, kernel_path, calibration_path, show_display):
    # Load kernel
    kernel = ndimage.imread(kernel_path, flatten=True)
    
    # Load images
    images = []
    # Loads images and sorts based on number included in name
    for image_path in glob.glob(image_directory + '/*.png'):
        image_name = image_path[len(image_directory)+1:]
        image_id = int(filter(type(image_name).isdigit, image_name))
        image = ndimage.imread(image_path, flatten=True)
        images.append((image_id, image))
    images.sort()
    images = map(lambda x: x[1], images)
    
    # Load calibration settings
    with open(calibration_path) as f:
        content = f.readlines()
        size = int(content[0])
        rotation = float(content[1])

    # Resizing image
    kernel = misc.imresize(kernel, (size, size)).astype(float)
    
    
    x = []
    intensities = []
    for i in range(len(images)):
        x.append(i)
        intensities.append(np.mean(images[i]))
    if show_display:
        print np.size(intensities)
        plt.plot(x, intensities)
        plt.show()
    
    focused = autofocus_batch(images, kernel, rotation=rotation)
    print focused
    
    
    if show_display:
        plt.imshow(images[focused], cmap='gray')
        plt.show()

###################################################################################################

# Autofocus (stream)

def _autofocus():
    def output(s):
        sys.stdout.write('#' + str(s) + '\n')
        sys.stdout.flush()

    def get_input():
        return sys.stdin.readline()
    
    # Load kernel
    kernel = ndimage.imread(kernel_path, flatten=True)
    
    # Load calibration settings
    with open(calibration_path) as f:
        content = f.readlines()
        size = int(content[0])
        rotation = float(content[1])

    # Resizing image
    kernel = misc.imresize(kernel, (size, size)).astype(float)

    #### Start closed loop process
    output(-1) # Request first image
    while True:
        line = get_input() # Get response
        
        parts = line.split(' ')
        width = int(parts[0])
        height = int(parts[1])
        data = parts[2]
        
        image = getImageFromData(data, width, height)
        
        # Get delta
        delta = 1
        delta = autofocus_step(image, kernel, rotation=rotation)
        time.sleep(0.25)
        
        output(delta) # Communicate delta
        
###################################################################################################

# Input handling

def getImageFromPath(path):
    return ndimage.imread(path, flatten=True)

def getImageFromData(data, width, height):
    buf = base64.b64decode(data)
    return np.reshape(np.frombuffer(buf, np.float32), (height, width))

def getDataFromPath(path):
    image = ndimage.imread(path, flatten=True)
    return base64.b64encode(image)
        
###################################################################################################

# Args handling

#parser = argparse.ArgumentParser()
#parser.add_argument("--calibrate", 
#                    help="the path of the image to calibrate with")
#parser.add_argument("--detect", 
#                    help="the path of the image to locate QR codes in")
#parser.add_argument("--autofocusbatch", 
#                    help="the path of the image directory for batch focusing")
#parser.add_argument('--display', action='store_true',
#                    help="include with --train to display relvant NN graphs")
#parser.set_defaults(display=False)
#args = parser.parse_args()
#
#kernel_path = "filter_160x160.png"
#calibration_path = 'calibration.txt'
#    
#if args.calibrate:
#    _calibrate(args.calibrate, kernel_path, calibration_path)
#elif args.detect:
#    _detect(args.detect, kernel_path, calibration_path, args.display)
#elif args.autofocusbatch:
#    _autofocus_batch(args.autofocusbatch, kernel_path, calibration_path, args.display)

kernel_path = "filter_160x160.png"
calibration_path = 'calibration.txt'

#data = getDataFromPath('focusImages/focus128.png')
#print data
#print getImageFromData(data)

#print sys.argv[5]

try:
    if sys.argv[1] == 'calibrate':
        if sys.argv[2] == 'data':
            width = int(sys.argv[3])
            height = int(sys.argv[4])
            data = sys.argv[5]
            image = getImageFromData(data, width, height)
            _calibrate(image, kernel_path, calibration_path)
        elif sys.argv[2] == 'path':
            path = sys.argv[3]
            image = getImageFromPath(path)
            _calibrate(image, kernel_path, calibration_path)
    elif sys.argv[1] == 'detect':
        if sys.argv[2] == 'data':
            width = sys.argv[3]
            height = sys.argv[4]
            data = sys.argv[5]
            image = getImageFromData(data, width, height)
            _detect(image, kernel_path, calibration_path, True)
        elif sys.argv[2] == 'path':
            path = sys.argv[3]
            image = getImageFromPath(path)
            _detect(image, kernel_path, calibration_path, True)
    elif sys.argv[1] == 'autofocus':
        _autofocus()
        
except Exception, err:
    print err
    print
    print
    print
    print 'To calibrate:'
    print '\tpython main.py calibrate data <width> <height> <img_data>'
    print '\tpython main.py calibrate path <img_path>'
    print 'To detect QR codes:'
    print '\tpython main.py detect data <width> <height> <img_data>'
    print '\tpython main.py detect path <img_path>'
    print 'To begin autofocus process:'
    print '\tpython main.py autofocus'
    print
    print 'Note: Must calibrate on example image before detection or autofocusing'