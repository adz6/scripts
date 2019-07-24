import glob
import csv
import time
from itertools import islice
import numpy as np

class data():

    def __init__(self, path_to_data_dir, mass_val):

        list_of_RGA_specta_paths = glob.glob(path_to_data_dir + '/*.csv') # get data path list
        datatemp = [] # temporary list to hold the data before we convert it to an array
        for i in range(len(list_of_RGA_specta_paths)):
            with open(list_of_RGA_specta_paths[i]) as csvtemp: # syntax for reading a csv file
                readertemp = csv.reader(csvtemp, delimiter = ',')
                for row in islice(readertemp, 137, None): # skips all the garbage at the start of the file
                    datatemp.append(row) # add the row of the csv file to the list.

        data = np.array(datatemp) # create an array to hold all the entries of the RGA csv files in the provided directory.
        data_pts = data.shape # we grab the dimensions of the array here

        for i in range(data_pts[0]): # converting the timestamp string into UNIX time
            data[i,0] = float(time.mktime(time.strptime(data[i,0],'%Y/%m/%d %H:%M:%S.%f')))

        mass_x_list = [] # temporary list to hold the array elements for our desired mass

        for i in range(data_pts[0]):
            if float(data[i,1]) == mass_val:
                mass_x_list.append(data[i,0:3]) # create a list of the data points for our desired mass.

        mass_x = np.array(mass_x_list) # convert it back to an array
        mass_x = mass_x.astype(float)
        mass_x = mass_x[mass_x[:,0].argsort()] # sort it from earliest to latest using the UNIX time column
        mass_x[:,0] = (mass_x[:,0]-mass_x[0,0])/60 # convert time to minutes since start of data

        self.massdata = mass_x #ndarray object that contains partial pressure and time data.
