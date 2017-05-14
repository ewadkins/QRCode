import numpy as np
from scipy import ndimage
from scipy import misc
import base64
from subprocess import Popen, PIPE
import sys
import re
import glob
import matplotlib.pyplot as plt

pos = 50

def getDataFromPath(path):
    image = ndimage.imread(path, flatten=True)
    height = len(image)
    width = len(image[0])
    return width, height, base64.b64encode(image)

def getImageFromData(data, width, height):
    buf = base64.b64decode(data)
    return np.reshape(np.frombuffer(buf, np.float32), (height, width))
    
    
# The following code creates a subprocess, whose stdin and stdout this process is able to write to and read from. 
# Then, we listen for commands from the subprocess that tell us how to adjust the camera. After adjusting it, we then send the subprocess image data, and the process repeats. Once the subprocess responds with a 0, it is in focus.

def processOutput(delta, p):
    global pos
    pos += delta
    ### Get next image somehow (from the camera, but for testing I grab it from a file)
    path = 'focusImages/focus' + str(int(pos)) + '.png'
    width, height, data = getDataFromPath(path)
    print path
    ###
    # Send main image data
    response = str(width) + ' ' + str(height) + ' ' + data
    p.stdin.write(response + '\n')
    ###
    ### Unimportant
    plt.imshow(getImageFromData(data, width, height), cmap="gray")
    plt.pause(0.00001)
    ###
    
p = Popen(['ipython', 'main.py', '--', 'autofocus'], stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True)

while True:
    # Wait for command from main
    line = p.stdout.readline().strip()
    if line:
        print line
        match = re.search('#(-?[0-9]+\.?[0-9]*)$', line) # if follows the form "#<delta>" where delta is a non-zero float telling you how to adjust
        if match is not None:
            delta = float(match.group(1))
            if delta != 0:
                processOutput(delta, p)
            else:
                print 'Done'
                print int(pos)
                p.kill()
                break
plt.show()
