import requests
from io import BytesIO
from PIL import Image
import math

# Google Maps API Key
API_KEY = ''

TILE_SIZE = 512
NW_COORD = [50.295702, 18.661792]
SE_COORD = [50.293735, 18.666952]

ZOOM = 19

LOGO_CUTOFF = 32

# Convert degrees into pixels
def project(lat_deg, lon_deg, zoom):
	lat_rad = lat_deg * (math.pi / 180.0)
	lon_rad = lon_deg * (math.pi / 180.0)

	x = ( TILE_SIZE / (math.pi*2) ) * pow(2, zoom) * (lon_rad + math.pi)
	y = ( TILE_SIZE / (math.pi*2) ) * pow(2, zoom) * (math.pi - math.log( math.tan( (math.pi/4) + (lat_rad/2) ) ))

	return y, x
	
# Convert pixels into degrees
def unproject(y, x, zoom):
	F = (TILE_SIZE/(2*math.pi)) * pow(2, zoom)

	lon_rad = (x/F) - math.pi
	lat_rad = 2 * math.atan(math.exp(math.pi - (y/F))) - (math.pi/2)

	lon_deg = (lon_rad * 180) / math.pi
	lat_deg = (lat_rad * 180) / math.pi

	return (lat_deg, lon_deg)


def get_maps_images(nw_coord, se_coord, zoom):

	nw_coord_projected = project(*nw_coord, zoom)
	se_coord_projected = project(*se_coord, zoom)

	# Calculate the map size and get the number of tiles
	map_width =  se_coord_projected[1] - nw_coord_projected[1]
	map_height = se_coord_projected[0] - nw_coord_projected[0]
	x_tiles = int(map_width / TILE_SIZE)
	y_tiles = int(map_height / TILE_SIZE)
	print(x_tiles, y_tiles)

	# Create a blank full scale image which will later consist of tiles stitches together
	out = Image.new('RGB', (x_tiles * TILE_SIZE, y_tiles * TILE_SIZE))

	for y in range(y_tiles):
		for x in range(x_tiles):
			filename = '%s_%s.jpg' % (x, y)
			print('Downloading tile [%s, %s] as %s' % (x, y, filename))
			print()

			# Add TILE_SIZE to current coordinates
			curr_center_coord = unproject(nw_coord_projected[0]+y*TILE_SIZE*2, nw_coord_projected[1]+x*TILE_SIZE*2, zoom)
			image_url = 'https://maps.googleapis.com/maps/api/staticmap?center=%s,%s&zoom=%s&size=512x512&maptype=satellite&key=%s' % (*curr_center_coord, zoom, API_KEY)

			# To get rid of the logos at the bottom we download one more image and cover the bottom bar with it
			bottom_center_coord = unproject(nw_coord_projected[0]+y*TILE_SIZE*2 + TILE_SIZE*2-LOGO_CUTOFF*2, nw_coord_projected[1]+x*TILE_SIZE*2, zoom)
			logo_image_url = 'https://maps.googleapis.com/maps/api/staticmap?center=%s,%s&zoom=%s&size=512x512&maptype=satellite&key=%s' % (*bottom_center_coord, zoom, API_KEY)

			# Download both images
			response = requests.get(image_url)
			img = Image.open(BytesIO(response.content)).convert('RGB')
			response = requests.get(logo_image_url)
			logo_img = Image.open(BytesIO(response.content)).convert('RGB')

			# Merge them together to get rid of the logos
			img.paste(logo_img, (0, TILE_SIZE-LOGO_CUTOFF))
			img.save('images/' + filename)

			# Stitch the tiles
			out.paste(img, (x*TILE_SIZE, y*TILE_SIZE))
	
	out.save('full.jpg')


if __name__ == '__main__':
	get_maps_images(NW_COORD, SE_COORD, ZOOM)
