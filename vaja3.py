#!/usr/bin/python3

import sys, getopt
import numpy as np
import cv2
from matplotlib import pyplot as plt
def watershad(image,outputfile):
    print ("something!")
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print(inputfile)
   img = cv2.imread(inputfile)
   #setting a image into a grayscale image
   gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   kernel = np.ones((2,2),np.uint8)
   print(kernel)
   # defining the gradient function
   #It is the difference between dilation and erosion of an image.
   gradient = cv2.morphologyEx(gray,cv2.MORPH_GRADIENT,kernel)
   #remove all the places that have a small
   ret, thresh = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

   # noise removal
   kernel = np.ones((2,2),np.uint8)
   show_images([gradient,thresh],2,["gradient","threshold"])

  # cv2.imshow('Gradient', gradient)
  # cv2.imshow('Return', thresh)
   #numpy_vstack = np.vstack((gradient, thresh))
   #plt.imshow('Gradient and treshold values', numpy_vstack)
  # cv2.waitKey()
   watershad(ret,outputfile)
   print ('Input file is "', inputfile)
   print ('Output file is "', outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])
