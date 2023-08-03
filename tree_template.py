import glob
import numpy as np
from skimage import io, draw

class TreeTemplate():

	def __init__(self, path, size):
		self.size = size
		self.rgb = self.load_templates(path)
		self.p = self.get_probabilities_template(size)

	# Load all tree templates from the specified directory and calculate the mean to get one, final template
	def load_templates(self, dir):

		if not dir.endswith('/'):
			dir = dir + '/*.png'
		else:
			dir = dir + '*.png'

		files = glob.glob(dir)

		images = []
		for file in files:
			img = io.imread(file) / 255.0
			images.append(img)

		# Return the mean along 0 axis
		return np.mean(images, axis=0)

	# Prepare the probabilities template, which is a white disk with specific dimensions
	def get_probabilities_template(self, size):
		template = np.zeros((size, size), dtype=np.uint8)
		rr, cc = draw.disk((int(size/2), int(size/2)), int(size/2)+1, shape=(size, size))
		template[rr, cc] = 1

		return template