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


step1 = 10
margin1 = 4
step2 = 1
margin2 = 2
index = 0
def autofocus_step(image, state, kernel, rotation=0):
    # Initial state
    global index
    last = None
    decreasingCount = 0
    direction = -1
    phase = 1
    xs = []
    ys = []
    
    if state is not None:
        index, last, decreasingCount, direction, phase, xs, ys = map(eval, state.split(':'))
        
    def state_string():
        return ':'.join(map(str, [index, last, decreasingCount, direction, phase, xs, ys])).replace(' ', '')
    
    def choose(x):
        global index
        xs.append(index)
        ys.append(val)
        index += x
        return x, state_string()
    
    if phase == 5:
        return 0, ''
    
    val = np.mean(image) if phase < 3 else np.max(locate(kernel, image, rotation=rotation, partial=True)[2])
    
    if last is None: # If first point
        last = val
        return choose(-(step1 if phase < 3 else step2)) # Back away from surface, and collect another data point
    decreasing = val < last
        
    last = val
        
    if phase == 1 or phase == 3: # Determine direction of peak
        direction *= -1 if decreasing else 1 # Change direction if next data point is decreasing val
        decreasingCount = 0
        phase += 1 # Continue to phase 2
        return choose((step1 if phase < 3 else step2) * direction * (2 if decreasing else 1))
    
    if phase == 2 or phase == 4: # Find bounds while collecting data
        if decreasing:
            decreasingCount += 1
        else:
            decreasingCount = max(0, decreasingCount - 1)
        if decreasingCount >= (margin1 if phase < 3 else margin2):
            # if consecutive data points decrease in val, change direction
            # Fit to quadratic and find max and return max displacement
            if phase == 2:
                a, b, c = np.polyfit(xs, ys, 2)
                result = -b / (2 * a)
            else:
                result = sorted(zip(xs, ys), lambda a, b: int(np.sign(b[1] - a[1])))[0][0]
            phase += 1
            xs = [] # Reset for filter search
            ys = []
            last = None
            displacement = result - index
            delta = result - index
            index += displacement
            return delta, state_string()
        return choose((step1 if phase < 3 else step2) * direction)
