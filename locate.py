import numpy as np
from scipy import ndimage
from scipy import signal
from scipy import misc
from scipy.ndimage import filters

def notch_filter_2d(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g /= g.sum()
    g -= (np.max(g) + np.min(g)) / 2
    return g

def locate(kernel, image, rotation=0, partial=False):
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
        kernel = notch_filter_2d(size, sigma);
        kernel -= 0.0003
        kernel /= np.abs(np.sum(kernel))
        return ndimage.convolve(image, kernel, mode='nearest')
    
    flipped_kernel = np.fliplr(np.flipud(kernel)) # Flip kernel for use in convolution
        
    kernel_width = kernel.shape[1]
    kernel_height = kernel.shape[0]

    # Process image
    image = 255 - image

    image_width = image.shape[1]
    image_height = image.shape[0]

    even_width = 1 if kernel.shape[1] % 2 == 0 else 0
    even_height = 1 if kernel.shape[0] % 2 == 0 else 0
    padded_image = np.pad(image, [(kernel.shape[1]/2, kernel.shape[1]/2 - even_width), (kernel.shape[0]/2, kernel.shape[0]/2 - even_height)], mode='edge')

    # Fastest way to calculate convolution for large images
    filtered = signal.fftconvolve(padded_image, flipped_kernel, mode='valid')

    # Find peaks in filtered
    peaks = peak_filter(filtered, 10, 4)
                    
    if partial:
        return [], filtered, peaks, None, kernel

    def threshold_image(image, threshold):
        thresholded = np.zeros(image.shape)
        thresholded[np.where(image > (threshold * np.max(image) + (1 - threshold) * np.min(image)))] = 1
        return thresholded

    threshold = 0.3
    regions = None

    # For thresholds
    for threshold in [x * 0.05 for x in range(10, 21)]:
        thresholded = threshold_image(peaks, threshold)
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
            # If too large of a region
            if blob_size >= 200:
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
        region_peaks = np.ma.array(peaks, mask=True)
        region_peaks.mask[regions[i][2]] = False
        ind = np.argmax(region_peaks)
        regions[i][3] = np.array([ind % peaks.shape[1], ind / peaks.shape[1]])

    regions.sort()
    regions.reverse()

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

    results = []
    for i in range(len(regions)):
        if regions[i][4]:
            x, y = np.round(regions[i][3]).astype(int)
            results.append((regions[i][3], peaks[y][x]))
    
    return results, filtered, peaks, thresholded, kernel