import numpy as np
from scipy import ndimage
from scipy import signal
from scipy import misc
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from time import time

def gaussian_filter(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

###ndimage.convolve([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0],[0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], [[1, 1, 1], [1, 1, 1],[1, 1, 1]], mode='nearest')

def locate(kernel, image, rotation=0, partial=False):
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
    kernel = ndimage.interpolation.rotate(kernel, rotation, cval=mid_val)

    def peak_filter(image, size, sigma): 
        kernel = gaussian_filter(size, sigma);
        kernel -= (np.max(kernel) + np.min(kernel)) / 2
        kernel -= 0.0003
        #kernel -= 0.0004
        #kernel -= 0.0001
        kernel /= np.abs(np.sum(kernel))
        #kernel = np.fliplr(np.flipud(kernel))
        #flipped_kernel = np.fliplr(np.flipud(kernel))
        return ndimage.convolve(image, kernel, mode='nearest')
        #return signal.fftconvolve(image, kernel, mode='valid')
        #image = np.pad(image, [(kernel.shape[1]/2, kernel.shape[1]/2), (kernel.shape[0]/2, kernel.shape[0]/2)], mode='constant')
        #result = ndimage.convolve(image, kernel, mode='nearest')
        #return result[kernel.shape[0]/2:result.shape[0] - kernel.shape[0]/2, kernel.shape[1]/2:result.shape[1] - kernel.shape[1]/2]
    
    #kernel = peak_filter(kernel, 10, 4)

    flipped_kernel = np.fliplr(np.flipud(kernel)) # Flip kernel for use in convolution
    
    #flipped_kernel = peak_filter(flipped_kernel, 10, 4)
    
    kernel_width = kernel.shape[1]
    kernel_height = kernel.shape[0]

    #kernel = 1./9 * np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    ###print 'Loaded kernel'

    # Process image
    image = 255 - image

    image_width = image.shape[1]
    image_height = image.shape[0]

    ###print 'Loaded image'

    # Performs the convolution without flipping the kernel
    test1 = time()
    #filtered = ndimage.correlate(image, kernel, mode='nearest')

    even_width = 1 if kernel.shape[1] % 2 == 0 else 0
    even_height = 1 if kernel.shape[0] % 2 == 0 else 0
    padded_image = np.pad(image, [(kernel.shape[1]/2, kernel.shape[1]/2 - even_width), (kernel.shape[0]/2, kernel.shape[0]/2 - even_height)], mode='edge')

    ###print 'Calculating convolution'

    # Fastest way to calculate convolution for large images
    filtered = signal.fftconvolve(padded_image, flipped_kernel, mode='valid')

    ###### TODO Work on combining kernel with peak filter before convolution over entire image

    #filtered = signal.correlate(image, kernel, mode='same')
    #filtered = signal.correlate2d(image, kernel, mode='same', boundary='symm') # Works but much slower than ndimage
    test2 = time()
    ###print 'Took ' + str(int((test2 - test1) * 1000)) + ' ms'

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
                
    if partial:
        return [], filtered, peaks, None, kernel

    test4 = time()
    ###print 'Took ' + str(int((test4 - test3) * 1000)) + ' ms'

    def threshold_image(image, threshold):
        thresholded = np.zeros(image.shape)
        thresholded[np.where(image > (threshold * np.max(image) + (1 - threshold) * np.min(image)))] = 1
        return thresholded

    threshold = 0.3
    regions = None

    test5 = time()
    # For thresholds [0.50, 0.55, ..., 0.95, 1.00]
    for threshold in [x * 0.05 for x in range(10, 21)]:
        ###print 'Thresholding with ' + str(threshold)
        thresholded = threshold_image(peaks, threshold)
        #plt.imshow(thresholded)
        #plt.pause(1)
        blobs, num_blobs = ndimage.label(thresholded)

        # If there are too many possible
        if num_blobs > 8:
            ###print 'Too many regions'
            continue

        regions = []
        too_large = False
        for i in range(num_blobs):
            blob_indices = np.where(blobs == i + 1)
            blob_size = len(blob_indices[0])
            if blob_size >= 200:
                ###print 'Too large of a region'
                too_large = True
                break
            regions.append([blob_size, i + 1, blob_indices, None, True])
        if too_large:
            continue
        break

    # Check for case of no real blobs
    if len(regions) == 1 and len(regions[0][2][0]) == 0:
        regions = []

    # Calculate center of regions or max of region
    for i in range(len(regions)):
        #regions[i][3] = np.average(regions[i][2], axis=1)[::-1]
        region_peaks = np.ma.array(peaks, mask=True)
        region_peaks.mask[regions[i][2]] = False
        ind = np.argmax(region_peaks)
        regions[i][3] = np.array([ind % peaks.shape[1], ind / peaks.shape[1]])

    regions.sort()
    regions.reverse()

    test6 = time()
    ###print 'Took ' + str(int((test6 - test5) * 1000)) + ' ms'

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


    results = []
    for i in range(len(regions)):
        if regions[i][4]:
            x, y = np.round(regions[i][3]).astype(int)
            results.append((regions[i][3], peaks[y][x]))

    test8 = time()
    ###print 'Took ' + str(int((test8 - test6) * 1000)) + ' ms'

    end = time()

    ###print 'Took ' + str(int((end - start) * 1000)) + ' ms'
    
    return results, filtered, peaks, thresholded, kernel

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
    
kernel_path = "filter_160x160.png"
image_path = "img6.png"
#image_path = "test6.png"

# Load kernel and image
original_kernel = ndimage.imread(kernel_path, flatten=True)
image = ndimage.imread(image_path, flatten=True)

assert original_kernel.shape[0] == original_kernel.shape[1]

size_low = 30
size_high = 90
size_step = 5
size_decrease = 6

rotation_low = -180
rotation_high = 180
rotation_step = 5
rotation_decrease = 8
rotation_min_step = 0.01

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

test1 = time()
        
size = None
rotation = None
partial = True
while True:
    break
    master_results = ([], [])
    max_magnitudes = ([], [])
    orientation_results = []
    for s in range(size_low, size_high + size_step, size_step):
        kernel = misc.imresize(original_kernel, (s, s)).astype(float)
        rotation_results = ([], [])
        max_magnitude = 0
        for r in frange(rotation_low, rotation_high + rotation_step, rotation_step):
            results, filtered, peaks, thresholded, modified_kernel = locate(kernel, image, rotation=r, partial=partial)
            locations = []
            magnitudes = [0]
            if len(results):
                locations, magnitudes = zip(*results)
            magnitude = np.max(peaks) if partial else np.max(magnitudes)
            print s, '\t', r, '\t', magnitude
            rotation_results[0].append(r)
            rotation_results[1].append(magnitude)
            master_results[0].append((s, r))
            master_results[1].append(magnitude)
            if magnitude > max_magnitude:
                max_magnitude = magnitude
        orientation_results.append(rotation_results)
        max_magnitudes[0].append(s)
        max_magnitudes[1].append(max_magnitude)
    
    print 'Size:', size_low, size_high, size_step
    print 'Rotation:', rotation_low, rotation_high, rotation_step
    
    max_ind = np.argmax(master_results[1])
    size, rotation = master_results[0][max_ind]
    val = master_results[1][max_ind]
    print 'Maxes:', size, rotation, val
    
    done = size_step == 1 and rotation_step == rotation_min_step

    if size_step > 1 or size_low != size_high:
        size_range = (size_high - size_low) / int(size_decrease)
        size_low = size - size_range / 2
        size_high = size + size_range / 2
        size_step = max(1, int(round(float(size_step) / size_decrease)))
        print 'New size:', size_low, size_high, size_step
    
    if rotation_step > rotation_min_step or rotation_low != rotation_high:
        rotation_range = (rotation_high - rotation_low) / float(rotation_decrease)
        rotation_low = rotation - rotation_range / 2
        rotation_high = rotation + rotation_range / 2
        rotation_step = max(rotation_min_step, rotation_step / float(rotation_decrease))
        print 'New rotation:', rotation_low, rotation_high, rotation_step

#    for i in range(len(orientation_results)):
#        plt.plot(orientation_results[i][0], orientation_results[i][1], label=str(i))
#    plt.show()
#
#    plt.plot(max_magnitudes[0], max_magnitudes[1])
#    plt.show()
    
    if done:
        break

test2 = time()
print 'Took ' + str(int((test2 - test1) * 1000)) + ' ms'
        
size = 41
rotation = -19.8553024902

print size
print rotation

kernel = misc.imresize(original_kernel, (size, size)).astype(float)
results, filtered, peaks, thresholded, modified_kernel = locate(kernel, image, rotation=rotation)
locations = []
magnitudes = [0]
if len(results):
    locations, magnitudes = zip(*results)
display(locations, image, filtered, peaks, thresholded, modified_kernel)
