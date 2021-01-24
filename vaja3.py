#!/usr/bin/python3

import sys, getopt
import numpy as np
import cv2
from matplotlib import pyplot as plt
def watershad(inputfile,outputfile):

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

   img = cv2.imread(inputfile)
   #setting a image into the grayscale
   gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

   # defining the gradient function

   # over the image and structuring element

   #It is the difference between dilation and erosion of an image.

   gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
   #remove all the places that have a small
   ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

   # noise removal
   kernel = np.ones((2,2),np.uint8)


   cv2.imshow('Gradient', gradient)


   watershad(inputfile,outputfile)
   print ('Input file is "', inputfile)
   print ('Output file is "', outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])
