import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from locate import locate

def autofocus_batch(images, kernel, rotation=0):
    
    low = 0
    high = len(images)
    step = 5
    decrease = 10
    
    fig = plt.figure()
    while True:
        magnitudes = ([], [])
        for i in filter(lambda x: x < len(images), range(low, high + step, step)):
            results, filtered, peaks, thresholded, modified_kernel = locate(kernel, images[i], rotation=rotation, partial=True)
            
            #edge_kernel = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
            #flipped_kernel = np.fliplr(np.flipud(edge_kernel))
            #even_width = 1 if kernel.shape[1] % 2 == 0 else 0
            #even_height = 1 if kernel.shape[0] % 2 == 0 else 0
            #padded_image = np.pad(images[i], [(kernel.shape[1]/2, kernel.shape[1]/2 - even_width), (kernel.shape[0]/2, kernel.shape[0]/2 - even_height)], mode='edge')
            
            magnitude = np.max(peaks)
            #edges = signal.fftconvolve(padded_image, flipped_kernel, mode='valid')
            #magnitude = np.max(edges)
            magnitudes[0].append(i)
            magnitudes[1].append(magnitude)
            print i, '\t', magnitude
            
            #fig.clear()
            #plt.imshow(edges, cmap='gray')
            #plt.pause(0.0001)
        
        max_ind = np.argmax(magnitudes[1])
        val = magnitudes[0][max_ind]
        print 'Params: ', low, high, step
        print 'Max: ', val, magnitudes[1][max_ind]
        
        plt.plot(magnitudes[0], magnitudes[1])
        plt.show()
        
        if step == 1:# and low == high:
            break

        if step > 1 or low != high:
            _range = (high - low) / int(decrease)
            low = val - _range / 2
            high = val + _range / 2
            step = max(1, int(round(float(step) / decrease)))
            print 'New:', low, high, step

    return val

def autofocus():
    pass