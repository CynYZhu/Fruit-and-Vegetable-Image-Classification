import time
import numpy as np

import random
import os
from PIL import Image

random.seed(281)


def read_data(data_directory="data/", label_filter=[]):
    """
    Reads the data from the files, creates labels by file name

    Parameters:
        data_dir (string): The name of a directory in which image data is located
        label_filters (list or array): Data labels to load, should match sub-folder names
                                     - default load all subdirectories

    Output:
        Tupel of lists (images, labels) where the former (images) is a list of matrix representations of the
        scaled image data and the latter (labels) is a list of strings of the corresponding label of each
        entry in the former (images) based on the sub-folder the image was loaded from.
    """
    images = []
    labels = []
    loaded = {}

    filter_labels = (len(label_filter) > 0)  # if we didn't pass in anything in the filter read everything

    # Crawl the data directory files and sub-directories...

    for dir_name, sub_dir_list, file_list in os.walk(data_directory):

        # Ignore anything in the actual directory itself (all images are in the sub-directories).
        if dir_name == data_directory:
            continue

        # The label will match the sub-folder name (root data folder ignored above.)
        label = os.path.basename(dir_name)

        if not filter_labels and not label in label_filter:
            label_filter.append(label)

        if label in label_filter:  # don't process directory if not in filter

            print(f'Processing label: {label}')

            for file_name in file_list:

                # Add label to loaded file dictinoary if it's not there already
                # (To keep track of how may files of this label have been loaded.)
                if label not in loaded:
                    loaded[label] = 0

                # Start actual loading process.
                file_path = os.path.join(dir_name, file_name)

                # open in with structure to avoid memory leaks
                with Image.open(file_path) as f:
                    # copy impage into np array
                    image = np.array(f)

                    # append to output
                    images.append(image)

                    # set label to be the index of the label string in the label_filter list
                    label_index = label_filter.index(label)
                    labels.append(label_index)

                loaded[label] += 1

                # Print a summary of what's been loaded by label.
    for label in loaded:
        print('%s: %s' % (label, loaded[label]))
    print('Total: %d' % sum(list(loaded.values())))

    return (images, labels, labels_used)
