from PIL import Image
import glob
import numpy as np
import random


def load_filenames (data_dir):
    """
    Load filenames of each class from a directory
    :param data_dir: path to the directory where files are
    :return: [[class0_file0, class0_file1, ...][class1_file0, ...]...[classN_file0, ...]]
    """
    filenames = []
    for i, class_path in enumerate(glob.glob(data_dir + '/*')):
        print(i, class_path)
        curr_images = []
        for file_path in glob.glob(class_path + '/*'):
            curr_images.append(file_path)

        filenames.append(curr_images)

    return filenames
    
    
def img2arr(filename, img_size):
#    img = Image.open(filename).convert('L')
    img = Image.open(filename).convert('RGB')    
    img = img.resize((img_size, img_size))
    img = np.array(img)
    img = img.astype('float32') / 255

    return img

def get_random_batch(filenames, batch_size, image_size):
    """
    Randomly select 'batch_size' of images per class from training set(flatten), encode labels, and make images into np array
    :param batch_size:
    :param filenames:
    :return: batch_x(np array) and batch_y(np array of onehot encoded labels)
    """
    # randomly select 'batch_size' of data from each class
    batch_names = []
    for label in range(len(filenames)):
        # print(label)
        randomly_selected = random.sample(filenames[label], batch_size)
        for filename in randomly_selected:
            # print(filename)
            batch_names.append(filename)

    # read in image files
    batch_images = []
    for filename in batch_names:
        # open, read, and resize image files
        img = img2arr(filename, image_size)
        batch_images.append(img)

    # reshape 'batch_images'
    batch_images = np.array(batch_images)

    return batch_images
    
    
    
def get_batch(filenames, batch_size, i, image_size):
    """
    Sequentially select 'batch_size' of images per class from training set(flatten), encode labels, and make images into np array
    :param batch_size:
    :param filenames:
    :return: batch_x(np array) 
    """
    # randomly select 'batch_size' of data from each class
    batch_names = []
    for label in range(len(filenames)):
        batch_names.append(filenames[label][i*batch_size:(i+1)*batch_size])

    batch_names = [name for label_list in batch_names for name in label_list]
    # read in image files
    batch_images = []
    for filename in batch_names:
        # open, read, and resize image files
        img = img2arr(filename, image_size)
        batch_images.append(img)

    # reshape 'batch_images'
    batch_images = np.array(batch_images)

    return batch_images