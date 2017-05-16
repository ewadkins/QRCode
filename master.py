import numpy as np
from scipy import ndimage
from scipy import misc
import base64
from subprocess import Popen, PIPE
import sys
import re
import glob
import matplotlib.pyplot as plt

pos = 80

def getImageFromPath(path):
    return ndimage.imread(path, flatten=True).astype(np.uint8)

def getImageFromData(data, width, height):
    buf = base64.b64decode(data)
    return np.reshape(np.frombuffer(buf, dtype=np.uint8), (height, width))

def getDataFromImage(image):
    buf = np.getbuffer(np.array(image, dtype=np.uint8))
    return base64.b64encode(buf)
    
# The following code creates a subprocess, whose stdin and stdout this process is able to write to and read from. 
# Then, we listen for commands from the subprocess that tell us how to adjust the camera. After adjusting it, we then send the subprocess image data, and the process repeats. Once the subprocess responds with a 0, it is in focus.
#
#def processOutput(delta, p):
#    global pos
#    pos += delta
#    ### Get next image somehow (from the camera, but for testing I grab it from a file)
#    path = 'focusImages/focus' + str(int(pos)) + '.png'
#    width, height, data = getDataFromPath(path)
#    print path
#    ###
#    # Send main image data
#    response = str(width) + ' ' + str(height) + ' ' + data
#    p.stdin.write(response + '\n')
#    ###
#    ### Unimportant
#    plt.imshow(getImageFromData(data, width, height), cmap="gray")
#    plt.pause(0.00001)
#    ###
#    
#p = Popen(['ipython', 'main.py', '--', 'autofocus'], stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True)
#
#while True:
#    # Wait for command from main
#    line = p.stdout.readline().strip()
#    if line:
#        print line
#        match = re.search('^OUTPUT (.+)$', line)
#        if match is not None:
#            parts = match.group(1).split(' ')
#            delta = float(parts[0])
#            state = parts[1] if len(parts) > 0 else None
#            if delta != 0:
#                processOutput(delta, p)
#            else:
#                print 'Done'
#                print int(pos)
#                p.kill()
#                break
#plt.show()

#### Get the image, width, and height somehow - Matlab script is responsible for this
# image should be a 2d array of pixel values from 0 to 255
image = getImageFromPath('focusImages/focus' + str(int(pos)) + '.png')
width = len(image[0])
height = len(image)
####

def getImageInfo(pos):
    image = getImageFromPath('focusImages/focus' + str(int(pos)) + '.png')
    print 'focusImages/focus' + str(int(pos)) + '.png'
    width = len(image[0])
    height = len(image)
    return image, width, height
    
image = None
data = None
state = ''

delta = None
while state is not None:
    image, width, height = getImageInfo(pos)
    data = getDataFromImage(image)
    args = ['ipython', '--', 'main.py', 'autofocus', 'data', str(width), str(height), data]
    if state:
        args.append(state)
    p = Popen(args, stdout=PIPE, universal_newlines=True)
    output = p.communicate()[0].strip()
    print output
    match = re.search('OUTPUT (.+)$', output)
    if match is not None:
        parts = match.group(1).split(' ')
        delta = float(parts[0])
        state = parts[1] if len(parts) > 1 else None
        pos += delta
        print delta, state
    else:
        raise 'Output formatted incorrectly'
