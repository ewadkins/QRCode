import numpy as np
from scipy import misc

from locate import locate

def calibrate(original_kernel, image):

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

    size = None
    rotation = None
    while True:
        master_results = ([], [])
        max_magnitudes = ([], [])
        orientation_results = []
        for s in range(size_low, size_high + size_step, size_step):
            kernel = misc.imresize(original_kernel, (s, s)).astype(float)
            rotation_results = ([], [])
            max_magnitude = 0
            for r in frange(rotation_low, rotation_high + rotation_step, rotation_step):
                results, filtered, peaks, thresholded, modified_kernel = locate(kernel, image, rotation=r, partial=True)
                locations = []
                magnitudes = [0]
                if len(results):
                    locations, magnitudes = zip(*results)
                magnitude = np.max(peaks)
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

        if done:
            break
    
    return size, rotation