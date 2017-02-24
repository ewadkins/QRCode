import numpy as np
from scipy import ndimage
from scipy import signal
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from time import time

#np.set_printoptions(threshold=np.inf)

kernel_filename = "filter_40x40.png"
#kernel_filename = "qrcode.png"
image_filename = "img6.png"
#image_filename = 'img1_straight.jpg'
#image_filename = 'img1_straight2.jpg'

# Load kernel and image
kernel = ndimage.imread(kernel_filename, flatten=True)
image = ndimage.imread(image_filename, flatten=True)

start = time()

# Process kernel
kernel = 255 - kernel
kernel /= 255
indifference_indices = np.where(np.logical_and(kernel >= 0.40, kernel <= 0.60))
kernel -= 0.3 # Weight the kernel (give more preference to either white or black, this is key)
kernel /= np.abs(np.sum(kernel))
mid_val = (np.max(kernel) + np.min(kernel)) / 2 # Set indifferent regions to the middle value
kernel[indifference_indices] = mid_val

#kernel = ndimage.interpolation.rotate(kernel, -21.5, cval=mid_val)
kernel = ndimage.interpolation.rotate(kernel, -21.5, cval=mid_val)

def peak_filter(image, size, sigma): 
    kernel = gaussian_filter(size, sigma);
    kernel -= (np.max(kernel) + np.min(kernel)) / 2
    kernel -= 0.0003
    #kernel -= 0.00001
    kernel /= np.abs(np.sum(kernel))
    #flipped_kernel = np.fliplr(np.flipud(kernel))
    #padded_image = np.pad(image, [(kernel.shape[1]/2, kernel.shape[1]/2 - even_width), (kernel.shape[0]/2, kernel.shape[0]/2 - even_height)], mode='edge')
    #return signal.fftconvolve(padded_image, flipped_kernel, mode='valid')
    return ndimage.convolve(image, kernel, mode='nearest')

def gaussian_filter(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

#kernel = peak_filter(kernel, 10, 4)

flipped_kernel = np.fliplr(np.flipud(kernel)) # Flip kernel for use in convolution

kernel_width = kernel.shape[1]
kernel_height = kernel.shape[0]

#kernel = 1./9 * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

print 'Loaded kernel'

# Process image
image = 255 - image

image_width = image.shape[1]
image_height = image.shape[0]

print 'Loaded image'

# Manual convolution (debugging)
#filtered = np.zeros(image.shape);
#for i in range(len(image)):
#    for j in range(len(image[0])):
#        print i, j
#        for k in range(len(kernel)):
#            for l in range(len(kernel[0])):
#                a = max(0, min(i - len(kernel) / 2 + k, len(image) - 1))
#                b = max(0, min(j - len(kernel[0]) / 2 + l, len(image[0]) - 1))
#                filtered[i][j] += image[a][b] * kernel[k][l]
#                #print ((a, b), image[a][b], kernel[k][l])

# Performs the convolution without flipping the kernel
test1 = time()
#filtered = ndimage.correlate(image, kernel, mode='nearest')

even_width = 1 if kernel.shape[1] % 2 == 0 else 0
even_height = 1 if kernel.shape[0] % 2 == 0 else 0
padded_image = np.pad(image, [(kernel.shape[1]/2, kernel.shape[1]/2 - even_width), (kernel.shape[0]/2, kernel.shape[0]/2 - even_height)], mode='edge')

filtered = signal.fftconvolve(padded_image, flipped_kernel, mode='valid')

###### TODO Work on combining kernel with peak filter before convolution over entire image

#filtered = signal.correlate(image, kernel, mode='same')
#filtered = signal.correlate2d(image, kernel, mode='same', boundary='symm') # Works but much slower than ndimage
test2 = time()
print 'Took ' + str(int((test2 - test1) * 1000)) + ' ms'

#
#def peak_filter(image, size, sigma): 
#    kernel = gaussian_filter(size, sigma);
#    kernel -= (np.max(kernel) + np.min(kernel)) / 2
#    kernel -= 0.0003
#    kernel /= np.abs(np.sum(kernel))
#    #flipped_kernel = np.fliplr(np.flipud(kernel))
#    #padded_image = np.pad(image, [(kernel.shape[1]/2, kernel.shape[1]/2 - even_width), (kernel.shape[0]/2, kernel.shape[0]/2 - even_height)], mode='edge')
#    #return signal.fftconvolve(padded_image, flipped_kernel, mode='valid')
#    return ndimage.convolve(image, kernel, mode='nearest')
#
#def gaussian_filter(size, sigma):
#    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
#    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
#    return g/g.sum()

test3 = time()
# Find peaks in filtered
peaks = peak_filter(filtered, 10, 4)
#peaks = filtered

test4 = time()
print 'Took ' + str(int((test4 - test3) * 1000)) + ' ms'

def threshold_image(image, threshold):
    thresholded = np.zeros(image.shape)
    thresholded[np.where(image > (threshold * np.max(image) + (1 - threshold) * np.min(image)))] = 1
    return thresholded

threshold = 0.3
regions = None

test5 = time()
# For thresholds [0.50, 0.55, ..., 0.95, 1.00]
for threshold in [x * 0.05 for x in range(10, 21)]:
    print 'Thresholding with ' + str(threshold)
    thresholded = threshold_image(peaks, threshold)
    #plt.imshow(thresholded)
    #plt.pause(1)
    blobs, num_blobs = ndimage.label(thresholded)
    
    # If there are too many possible
    if num_blobs > 8:
        print 'Too many regions'
        continue
    
    regions = []
    too_large = False
    for i in range(num_blobs):
        blob_indices = np.where(blobs == i + 1)
        blob_size = len(blob_indices[0])
        if blob_size >= 200:
            print 'Too large of a region'
            too_large = True
            break
        regions.append([blob_size, i + 1, blob_indices, None, True])
    if too_large:
        continue
    break

# Check for case of no real blobs
if len(regions) == 1 and len(regions[0][2][0]) == 0:
    regions = []
        
# Calculate center of regions
for i in range(len(regions)):
    regions[i][3] = np.average(regions[i][2], axis=1)[::-1]
    
regions.sort()
regions.reverse()

test6 = time()
print 'Took ' + str(int((test6 - test5) * 1000)) + ' ms'

test7 = time()
for i in range(len(regions)):
    out_of_bounds_x = regions[i][3][0] < kernel_width / 4 or image_width - regions[i][3][0] < kernel_width / 4
    out_of_bounds_y = regions[i][3][1] < kernel_height / 4 or image_height - regions[i][3][1] < kernel_height / 4
    if out_of_bounds_x or out_of_bounds_y:
        regions[i][4] = False
        thresholded[regions[i][2]] = 0

for i in range(len(regions)):
    for j in range(i + 1, len(regions)):
        if regions[i][4] and np.linalg.norm(regions[i][3] - regions[j][3]) < np.linalg.norm(kernel.shape):
            regions[j][4] = False
            thresholded[regions[j][2]] = 0
                        
#plt.imshow(thresholded)
#plt.pause(1)
#plt.close()


locations = []
for i in range(len(regions)):
    if regions[i][4]:
        locations.append(regions[i][3])

test8 = time()
print 'Took ' + str(int((test8 - test6) * 1000)) + ' ms'
        
#filtered = np.sign(filtered) * filtered ** 2

print locations

end = time()

print 'Took ' + str(int((end - start) * 1000)) + ' ms'

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
