# simple script that will build mass trend-lines from RGA spectrum data.

import glob
import csv
import time
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

script, path_to_RGA_data = argv # call the script with the path to the file containing the RGA csv files.

list_of_RGA_specta_paths = glob.glob(path_to_RGA_data + '/*.csv')# This returns a list of the the paths for all
# the RGA data. It is not sorted.

# we need to iterate through each of the files and extract the timestamp, mass value, and partial pressure
# then save those things into an array.

datatemp = [] # temporary list to hold the data before we convert it to an array
for i in range(len(list_of_RGA_specta_paths)):
    with open(list_of_RGA_specta_paths[i]) as csvtemp: # syntax for reading a csv file
        readertemp = csv.reader(csvtemp, delimiter = ',')
        for row in islice(readertemp, 137, None): #skips all the garbage at the start of the file
            datatemp.append(row) # add the row of the csv file to the list.

data = np.array(datatemp) # create an array to hold all the entries of the RGA csv files in the provided directory.
data_pts = data.shape # we grab the dimensions of the array here

for i in range(data_pts[0]): # converting the timestamp string into UNIX time
    data[i,0] = float(time.mktime(time.strptime(data[i,0],'%Y/%m/%d %H:%M:%S.%f')))

mass_x_list = [] # temporary list to hold the array elements for our desired mass

for i in range(data_pts[0]):
    if float(data[i,1]) == 3.0:
        mass_x_list.append(data[i,0:3]) # create a list of the data points for our desired mass.

mass_x = np.array(mass_x_list) # convert it back to an array
mass_x = mass_x.astype(float)
mass_x = mass_x[mass_x[:,0].argsort()] # sort it from earliest to latest using the UNIX time column
mass_x[:,0] = (mass_x[:,0]-mass_x[0,0])/60 # convert time to minutes since start of data

plt.plot(mass_x[:,0],mass_x[:,2])
axes = plt.gca()
axes.set_ylim([0,1.e-7])
plt.xlabel('Minutes Since Start of RGA Data')
plt.ylabel("Partial Pressure (Torr)")
plt.show()
