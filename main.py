import os
import sys
import glob
import config
import numpy as np
import maxflow as mf
from skimage import io, color, draw
from detectree import pixel_features
from sklearn import ensemble
from termcolor import cprint
from tree_template import TreeTemplate

def done():
	cprint('Done.', 'green')

def bprint(msg):
	cprint('[*] %s' % msg, 'blue')

def err(msg):
	cprint('\n%s\n' % msg, 'red')
	sys.exit()

def get_distance_between_points(p1, p2):
	return np.sqrt(((p2['x'] - p1['x'])**2) + ((p2['y'] - p1['y'])**2))

# Calculate the correlation coefficient between the specified image + its probabilities predicted
# 	by the classifier, and the tree template image + the template disk with suitable dimensions 
def get_correlation(cropped_img, probabilities, template, template_disk):
	return np.sum(np.square(cropped_img - template)) + np.sum(np.square(probabilities - template_disk))

def detect_trees(full_map, map_pred_refined, map_prob, templates):
	label_1 = map_prob[:, :, 1]
	candidates = []

	for template in templates:

		# Get starting and ending point for the image slice relative to its center pixel
		# a, b fit x-a and x+b such that the image slice has the same shape as the template
		a = int(template.size / 2)
		if a % 2 != 0:
			b = a + 1
		else:
			b = a

		for y in range(full_map.shape[0]):
			for x in range(full_map.shape[1]):
				# Choose only slices for which the center pixel is labeled as 1
				if map_pred_refined[y, x] == 1:

					img_slice = full_map[y-a:y+b, x-a:x+b]
					probabilities_slice = label_1[y-a:y+b, x-a:x+b]
					
					if(img_slice.shape != (template.size, template.size, 3)):
						continue

					coeff = get_correlation(img_slice, probabilities_slice, template.rgb, template.p)
					candidates.append({'x': x, 'y': y, 'r': template.size/2, 'coeff': coeff})

	# Sort by coefficients in decreasing order
	def get_coeff(x):
		return x['coeff']
	candidates = sorted(candidates, key=get_coeff, reverse=True)

	# Apply minimum threshold to the similiarity coefficient
	candidates = [x for x in candidates if x['coeff'] > config.SIMILARITY_THRESHOLD]

	trees = []
	while len(candidates) != 0:
		top_candidate = candidates[0]
		trees.append(top_candidate)

		candidates.remove(top_candidate)

		# Remove all overlaps from the candidates list
		candidates = [candidate for candidate in candidates if (top_candidate['r'] + candidate['r'] - get_distance_between_points(top_candidate, candidate)) / min(top_candidate['r'], candidate['r']) < config.OVERLAP_THRESHOLD]

	return trees

# Solve the max-flow-min-cut problem to perform image segmentation
def refine(map_prob):
	structure = np.array([[0, 0, 0],
						  [0, 0, 1],
						  [1, 1, 1]])

	label_0 = map_prob[:, :, 0]
	label_1 = map_prob[:, :, 1]

	# Create the graph
	g = mf.GraphInt()
	node_ids = g.add_grid_nodes((map_prob.shape[0], map_prob.shape[1]))

	g.add_grid_edges(node_ids, weights=100, structure=structure, symmetric=True)
	cap_0 = (1e4 * np.log(label_1)).astype(int) # Sink -> i capacities
	cap_1 = (1e4 * np.log(1 - label_1)).astype(int) # i -> Source capacities
	g.add_grid_tedges(node_ids, cap_1, cap_0)

	g.maxflow()

	new_labels = g.get_grid_segments(node_ids).astype(int)
	return new_labels


def main():

	bprint('Loading the training set...')
	if config.TRAINING_DIR.endswith('/'):
		x_dir = config.TRAINING_DIR + 'x/'
		y_dir = config.TRAINING_DIR + 'y/'
	else:
		x_dir = config.TRAINING_DIR + '/x/'
		y_dir = config.TRAINING_DIR + '/y/'

	x_files = [os.path.basename(path) for path in glob.glob(x_dir + '*.jpg')]
	y_files = [os.path.basename(path) for path in glob.glob(y_dir + '*.jpg')]

	if len(x_files) == 0:
		err('%s cannot be empty' % x_dir)

	featureBuilder = pixel_features.PixelFeaturesBuilder()
	clf = ensemble.AdaBoostClassifier()

	x = np.ndarray( (len(x_files), config.TILE_SIZE**2, 27) )
	y = np.ndarray( (len(y_files), config.TILE_SIZE**2) )
	for i, file in enumerate(x_files):
		# Read x and its corresponding y (labeled) image
		x_image = io.imread(x_dir + file) / 255.0
		try:
			y_image = color.rgb2gray(io.imread(y_dir + file) / 255.0)
		except FileNotFoundError:
			err('ERROR: There is no corresponding labeled image for %s%s' % (x_dir, file))

		y_image[y_image >= 0.5] = 1
		y_image[y_image < 0.5] = 0

		x_features = featureBuilder.build_features_from_arr(x_image)
		x_features = x_features.reshape((config.TILE_SIZE**2, 27))
		y_labels = y_image.reshape((config.TILE_SIZE**2,))

		x[i, :, :] = x_features
		y[i, :] = y_labels
	done()
	
	x = x.reshape( (len(x_files) * config.TILE_SIZE**2, 27) )
	y = y.reshape( (len(x_files) * config.TILE_SIZE**2,) )

	bprint('Training the classifier...')

	# Train the classifier
	clf.fit(x, y)

	done()

	bprint('Classifying the full scale image...')

	# Load the full scale map
	map_image = io.imread(config.FULL_MAP) / 255.0
	map_features = featureBuilder.build_features_from_arr(map_image).reshape(( map_image.shape[0] * map_image.shape[1], 27 ))

	# Classify the full scale image
	map_prob = clf.predict_proba(map_features).reshape((map_image.shape[0], map_image.shape[1], 2))
	map_pred = clf.predict(map_features).reshape((map_image.shape[0], map_image.shape[1]))

	done()

	bprint('Refining the predictions...')
	map_pred_refined = refine(map_prob)
	done()
	io.imsave('trees_mask.jpg', map_pred_refined)

	bprint('Detecting trees...')

	# Load the templates
	template_dirs = os.listdir(config.TEMPLATES_DIR)
	templates = [TreeTemplate(config.TEMPLATES_DIR + path, int(path.split('x')[-1])) for path in template_dirs if 'x' in path]

	trees = detect_trees(map_image, map_pred_refined, map_prob, templates)

	''' GENERATE THE RESULTS '''

	# Draw the tree outlines on the map
	map_with_trees = map_image.copy()
	for tree in trees:
		row, col = draw.circle_perimeter(tree['y'], tree['x'], int(tree['r']))
		map_with_trees[row, col, :] = [1, 1, 1]
	io.imsave('trees_map.jpg', map_with_trees)

	''' THIS IS DEBUGGING '''

	done()

if __name__ == '__main__':
	main()