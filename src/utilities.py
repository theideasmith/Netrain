import imageio
import os
from Constants import *
import cv2
import numpy as np
def rgb2gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) # color.rgb2gray(rgb)
                                                            
def ifnone(dicty, key, default):
  return dicty[key] if key in dicty else default


def reader_imageio(videofile):                       
  vid = imageio.get_reader(videofile)         
  metadata = vid.get_meta_data()              
  print metadata
  nframes = metadata['nframes']               
  current = 0                                 
  for frame in vid:
    yield frame
  print 'Print video is closing'
  vid.close()
  raise StopIteration()

def reader_cv2(vidloc):
  vid = cv2.VideoCapture(vidloc)
  while vid.isOpened():
    r, f = vid.read()
    if r: yield f
  print "iterating stopped"
  raise StopIteration()

def framereader(vidloc, mode=VIDREAD_MODE, grayscale=False):
  def loadread():
    reader = None
    if mode =='imageio':
      reader = reader_imageio(vidloc) 
    elif mode =='cv' or mode == 'cv2':
      reader = reader_cv2(vidloc)
    return reader
  reader = loadread()
  if reader:
    while True:
      try:
        frame = reader.next()
        if grayscale: frame = rgb2gray(frame) 
        frame = frame.astype(np.float64)      
        yield frame                           
      except StopIteration:
        reader = loadread()
        continue
  
def prepare_training_image(image): 
  f = image.astype(np.float64)     
  z  = (f-np.mean(f))/np.std(f)    
  return z                         

def wrap(start,n,end):    
    s = min([start, end]) 
    e = max([start, end]) 
    
    ret = np.array(n)
    ret[s<=n and n<=e][:] = n
    ret[n>e][:] = e
    ret[n<s] = s
    return ret
