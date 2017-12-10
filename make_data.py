import tensorflow as tf
import numpy as np
import os
import sys
import random
import csv
import shutil
import ntpath
from PIL import Image, ImageEnhance, ImageOps
from six.moves import cPickle as pickle

# Assumes that the images are in folder medical_data in the same directory.

DATA_PATH = 'medical_data/'
NUM_CHANNELS = 3
IMAGE_SIZE = 224
PIXEL_DEPTH = 255.0
# For each disease


DISEASE_MAP = {'Cardiomegaly': 1, 'Emphysema': 2, 'Effusion': 3, 
	'No Finding': 0, 'Hernia': 5, 'Infiltration': 6, 'Mass': 7, 
	'Nodule': 8, 'Atelectasis': 9, 'Pneumothorax': 10, 
	'Pleural_Thickening': 11, 'Pneumonia': 12, 'Fibrosis': 13, 
	'Edema': 14, 'Consolidation': 4}

def scale_pixel_values(dataset):
  return (dataset - PIXEL_DEPTH / 2.0) / PIXEL_DEPTH

def save_pickle_file(pickle_file, save_dict):
  try:
    f = open(DATA_PATH + pickle_file, 'wb')
    pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

  print("Datasets saved to file", DATA_PATH + pickle_file)


def read_image_from_file(file_path):
	img = Image.open(file_path)
	img = img.resize((IMAGE_SIZE,IMAGE_SIZE), Image.ANTIALIAS) #downsample image
	pixel_values = np.array(img.getdata())
	pixel_values = np.reshape(pixel_values, (IMAGE_SIZE ** 2, -1))
	pixel_values = pixel_values[:, 0]
	#print("#### Pixel Shape:", pixel_values.shape)
	return np.reshape(pixel_values, [IMAGE_SIZE, IMAGE_SIZE, 1])

def make_dataset_arrays(num_samples, NUM_DISEASES):
  data = np.ndarray((num_samples * NUM_DISEASES, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
  labels = np.ndarray((num_samples * NUM_DISEASES, NUM_DISEASES), dtype=np.int32)
  return data, labels

# This creates the double labling dataset, the TRAIN_SIZE, VAL_SIZE, TEST_SIZE 
# are the number of samples for a single class 
def create_quad_dataset(TRAIN_SIZE, VAL_SIZE, TEST_SIZE):

	DISEASE_QUAD_MAP = {}
	NUM_DISEASES = 15 * 7
	SINGLE_DATASET_SIZE = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
	TOTAL_DATASET_SIZE = SINGLE_DATASET_SIZE * NUM_DISEASES 
	train_data, train_labels = make_dataset_arrays(TRAIN_SIZE, NUM_DISEASES)
	val_data, val_labels = make_dataset_arrays(VAL_SIZE, NUM_DISEASES)
	test_data, test_labels = make_dataset_arrays(TEST_SIZE,  NUM_DISEASES)

	num_train, num_val , num_test = 0, 0, 0
	all_Diseases = list(DISEASE_MAP.keys())
	dataCounts = {}
	diseaseIdentifier = 0

	disease_id = 0

	for i in range(len(all_Diseases)):
		for j in range(i+1, len(all_Diseases)):
			disease_1 = all_Diseases[i]
			disease_2 = all_Diseases[j]
			if (disease_1 > disease_2):
				disease_1, disease_2 = disease_2, disease_1
			dataCounts[ (disease_1, disease_2)] = 0
			DISEASE_QUAD_MAP[(disease_1, disease_2)] = disease_id
			disease_id += 1

	print(DISEASE_QUAD_MAP)
	with open('Data_Entry_2017.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if(len(row['Finding Labels'].split("|")) != 2): 
				continue;

			diseases = row['Finding Labels'].split("|")

			disease_1 = diseases[0]
			disease_2 = diseases[1]
			if (disease_1 > disease_2):
				disease_1, disease_2 = disease_2, disease_1

			dCount = dataCounts[(disease_1, disease_2 )]
			if ( dCount  == SINGLE_DATASET_SIZE):
				continue;

			disease_num = DISEASE_QUAD_MAP[(disease_1, disease_2)]
			image_file = os.path.join(DATA_PATH, row['Image Index'])
			image_data = read_image_from_file( image_file)

			if (dCount  < TRAIN_SIZE):
				train_data[num_train, :, :, :] = image_data
				train_labels[num_train, :] = 0
				train_labels[num_train, disease_num] = 1
				num_train += 1

			if (dCount  >= TRAIN_SIZE and dCount  < TRAIN_SIZE + VAL_SIZE):
				val_data[num_val, :, :, :] = image_data
				val_labels[num_val, :] = 0
				val_labels[num_val, disease_num] = 1
				num_val += 1
			
			if (dCount >= TRAIN_SIZE + VAL_SIZE):
				test_data[num_test, :, :, :] = image_data
				test_labels[num_test, :] = 0
				test_labels[num_test, disease_num] = 1
				num_test += 1
				

			dataCounts[(disease_1, disease_2)] += 1
			if(num_train + num_val + num_test == TOTAL_DATASET_SIZE ):
				break 

	train_data = scale_pixel_values(train_data)
	val_data = scale_pixel_values(val_data)
	test_data = scale_pixel_values(test_data)
	
	print('Disease Distribution', dataCounts)
	print('Medical dataset size:', train_data.shape)
	print('Mean:', np.mean(train_data))
	print('Standard deviation:', np.std(train_data))
	print('') 	

	pickle_file = 'medical_quad_data.pickle'
	save = {
		'train_data': train_data,
		'train_labels': train_labels,
		'val_data': val_data,
		'val_labels': val_labels,
		'test_data' : test_data,
		'test_labels' : test_labels
	}
	
	save_pickle_file(pickle_file, save)
	return (train_data, train_labels, val_data, val_labels, test_data, test_labels)


# Return Datasets with Just One label, the TRAIN, VAL, TEST sizes are for
# a single class
def create_simple_dataset(TRAIN_SIZE, VAL_SIZE, TEST_SIZE):
	
	NUM_DISEASES = 15
	SINGLE_DATASET_SIZE = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
	TOTAL_DATASET_SIZE = SINGLE_DATASET_SIZE * NUM_DISEASES 
	train_data, train_labels = make_dataset_arrays(TRAIN_SIZE, NUM_DISEASES)
	val_data, val_labels = make_dataset_arrays(VAL_SIZE, NUM_DISEASES)
	test_data, test_labels = make_dataset_arrays(TEST_SIZE, NUM_DISEASES)

	num_train, num_val , num_test = 0, 0, 0
	dataCounts = {}
	for disease in DISEASE_MAP.keys():
		dataCounts[disease] = 0


	with open('Data_Entry_2017.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if(len(row['Finding Labels'].split("|")) != 1): 
				continue;

			disease = row['Finding Labels']
			disease_num = DISEASE_MAP[disease]

			if (dataCounts[disease] == SINGLE_DATASET_SIZE):
				continue;

			image_file = os.path.join(DATA_PATH, row['Image Index'])
			image_data = read_image_from_file( image_file)

			if (dataCounts[disease] < TRAIN_SIZE):
				train_data[num_train, :, :, :] = image_data
				train_labels[num_train, :] = 0
				train_labels[num_train, disease_num] = 1
				num_train += 1

			if (dataCounts[disease] >= TRAIN_SIZE and dataCounts[disease] < TRAIN_SIZE + VAL_SIZE):
				val_data[num_val, :, :, :] = image_data
				val_labels[num_val, :] = 0
				val_labels[num_val, disease_num] = 1
				num_val += 1
			
			if (dataCounts[disease] >= TRAIN_SIZE + VAL_SIZE):
				test_data[num_test, :, :, :] = image_data
				test_labels[num_test, :] = 0
				test_labels[num_test, disease_num] = 1
				num_test += 1
				

			dataCounts[disease] += 1
			if(num_train + num_val + num_test == TOTAL_DATASET_SIZE ):
				break 

	train_data = scale_pixel_values(train_data)
	val_data = scale_pixel_values(val_data)
	test_data = scale_pixel_values(test_data)
	
	print('Disease Distribution', dataCounts)
	print('Medical dataset size:', train_data.shape)
	print('Mean:', np.mean(train_data))
	print('Standard deviation:', np.std(train_data))
	print('') 	

	pickle_file = 'medical_simple_data.pickle'
	save = {
		'train_data': train_data,
		'train_labels': train_labels,
		'val_data': val_data,
		'val_labels': val_labels,
		'test_data' : test_data,
		'test_labels' : test_labels
	}
	
	save_pickle_file(pickle_file, save)
	return (train_data, train_labels, val_data, val_labels, test_data, test_labels)

# Return Datasets with Just One Disease
def create_k_hot_dataset(SINGLE_DATASET_SIZE):
	
	NUM_DISEASES = 15
	TOTAL_DATASET_SIZE = SINGLE_DATASET_SIZE * NUM_DISEASES 
	data, labels = make_dataset_arrays(SINGLE_DATASET_SIZE, NUM_DISEASES)

	num_images= 0
	dataCounts = {}
	for disease in DISEASE_MAP.keys():
		dataCounts[disease] = 0


	with open('Data_Entry_2017.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			
			image_file = os.path.join(DATA_PATH, row['Image Index'])
			image_data = read_image_from_file( image_file)

			if (num_images % 100 == 0):
				print("Current Data set size:", num_images)

			if (num_images < TOTAL_DATASET_SIZE):
				data[num_images, :, :, :] = image_data
				labels[num_images, :] = 0
				for disease in row['Finding Labels'].split("|"):
					disease_num = DISEASE_MAP[disease]
					dataCounts[disease] += 1
					labels[num_images, disease_num] = 1
				num_images += 1

				
			if(num_images == TOTAL_DATASET_SIZE ):
				break 

	data = scale_pixel_values(data)
	
	
	print('Disease Distribution', dataCounts)
	print('Medical dataset size:', data.shape)
	print('Mean:', np.mean(data))
	print('Standard deviation:', np.std(data))
	print('') 	

	pickle_file = 'medical_k_hot_data.pickle'
	save = {
		'train_data': data,
		'train_labels': labels,
		'val_data': data,
		'val_labels': labels,
		'test_data' : data,
		'test_labels' : labels
	}
	
	save_pickle_file(pickle_file, save)
	return (data, labels)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation, :, :, :]
  shuffled_labels = labels[permutation, :]
  return shuffled_dataset, shuffled_labels

def data_augmentation(dataset, labels):
  graph = tf.Graph()
  with graph.as_default():
    tf_img = tf.placeholder(tf.float32, shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    flipped_image = tf.image.random_flip_left_right(tf_img)

    brightened_image = tf.image.random_brightness(tf_img, max_delta=50)
    brightened_image = tf.clip_by_value(brightened_image, 0.0, PIXEL_DEPTH)

    contrasted_image = tf.image.random_contrast(tf_img, lower=0.5, upper=1.5)
    contrasted_image = tf.clip_by_value(brightened_image, 0.0, PIXEL_DEPTH)

  '''Supplement dataset with flipped, rotated, etc images'''
  n = len(dataset)
  new_data, new_labels = make_dataset_arrays(num_rows=n*4)
  num_new = 0

  with tf.Session(graph=graph) as session:
    for i in range(len(dataset)):
      img = np.reshape(dataset[i,:,:,:], (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
      img = np.asarray(img)
      img = img.astype(np.float32)
      label = labels[i,:]
      for _ in range(3):
        r = random.uniform(0,1)
        new_img = session.run(flipped_image, feed_dict={tf_img : img})
        if r < 0.5:
          new_img = session.run(brightened_image, feed_dict={tf_img : new_img})
          new_img = session.run(contrasted_image, feed_dict={tf_img : new_img})
        else:
          new_img = session.run(contrasted_image, feed_dict={tf_img : new_img})
          new_img = session.run(brightened_image, feed_dict={tf_img : new_img})
        new_data[num_new,:,:,:] = new_img
        new_labels[num_new,:] = label
        num_new += 1

  assert num_new == n*3
  new_data[num_new:,:,:,:] = dataset
  new_labels[num_new:,:] = labels
  new_data, new_labels = randomize(new_data, new_labels)
  return new_data, new_labels

def augment_training_set(PICKLE_FILE):
  print("\nAugmenting training data...")
  with open(PICKLE_FILE, 'rb') as f:
    save = pickle.load(f, encoding='latin1')
    train_X = save['train_data']
    train_Y = save['train_labels']

  train_RGB = (train_X * PIXEL_DEPTH) + PIXEL_DEPTH / 2.0 
  new_train, new_labels = data_augmentation(train_RGB, train_Y)
  new_train = scale_pixel_values(new_train)

  save['train_data'] = new_train
  save['train_labels'] = new_labels
  save_pickle_file('augmented_medical_data.pickle', save)

create_k_hot_dataset(2000)

