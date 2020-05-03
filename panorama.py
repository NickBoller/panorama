import argparse
import numpy as np
import math
from PIL import Image, ImageDraw
import random

	
"""
RANSAC:
-  n >= 8 b/c 2d projective transformation 


"""
def panorama():
	print("panorama")

def ransac(img1_path, img2_path, visualize, n, k, t, d):
	# Get the corners array with the xy coords, harris response value, and the x and y gradient windows
	print("Computing corners for image 1...")
	img1_keypoints = harris_corner_detector(img1_path, args.windowsize, args.k, args.threshold, args.sigmasq, args.keypoints, False)
	print("\nComputing corners for image 2...")
	img2_keypoints = harris_corner_detector(img2_path, args.windowsize, args.k, args.threshold, args.sigmasq, args.keypoints, False)

	# Raw/Candidate match list img1 => img2
	match_dict = {}

	# Loop through keypoints and find correspondences
	print("Finding Correspondences")
	for i in range(len(img1_keypoints)):
		img1 = img1_keypoints[i]
		for j in range(len(img2_keypoints)):
			img2 = img2_keypoints[j]
			L2 = 0.0
			# Extract the x and y gradients windows for each image, default size: 5x5
			G1x = img1[3]
			G1y = img1[4]

			G2x = img2[3]
			G2y = img2[4]

			for x in range(len(G1x)):
				for y in range(len(G1y)):
					# Compute the L2 loss
					L2 += (G1x[x][y]-G2x[x][y])**2 + (G1y[x][y]-G2y[x][y])**2
			# Map the xy coords from img1 to the xy coords from img2 along with the L2 loss
			if (img1[0], img1[1]) not in match_dict:
				match_dict[(img1[0], img1[1])] = (img2[0], img2[1], L2)
			elif L2 < match_dict[(img1[0], img1[1])][2]:
				match_dict[(img1[0], img1[1])] = (img2[0], img2[1], L2)

	# Sort the match dictionary by L2 loss
	sorted_match_list = sorted(match_dict.items(), key=lambda item: item[1][2])

	# Clean up sorted_match_list and take only the best half 
	match_list = []
	for toks in sorted_match_list:
		match_list.append(list(toks[0]) + list(toks[1]))

	half_length = int(len(match_list)/2)
	match_list = match_list[:half_length]

	# Show the corrspondences 
	if visualize:
		im1 = Image.open(img1_path)
		rgb1 = im1.convert('RGB')

		im2 = Image.open(img2_path)
		rgb2 = im2.convert('RGB')

		r = lambda: random.randint(0,255)

		for i in match_list[:10]:
			r1 = r()
			g1 = r()
			b1 = r()

			draw1 = ImageDraw.Draw(rgb1)
			draw1.ellipse((i[0]-5, i[1]-5, i[0]+5, i[1]+5), outline=(r1,g1,b1))

			draw2 = ImageDraw.Draw(rgb2)
			draw2.ellipse((i[2]-5, i[3]-5, i[2]+5, i[3]+5), outline=(r1,g1,b1))

		rgb1.show() 
		rgb2.show()




def harris_corner_detector(input_filename, window_size, k, threshold, sigmasq, max_corners, visualize):
	original = Image.open(input_filename)

	# Convert the image to greyscale to make it easier to work with
	im = original.convert('L')
	width, height = im.size

	# Seperable Sobel Filter to compute gradients
	sobel_x = np.array([-1,1])

	# Pad the image
	print("Padding Image...")
	sobel_padding = int(len(sobel_x)/2)
	gaussian_padding = 3
	padded_pixels, padded_pixels_width, padded_pixels_height = pad_image(im, sobel_padding+gaussian_padding)

	# Blur the image with a gaussian blur filter
	print("Applying Blur...")
	blurred_image = gaussian_blur(padded_pixels, padded_pixels_width, padded_pixels_height, sigmasq)
	blur_height, blur_width = blurred_image.shape

	# Compute Intensity Gradient components using 2 seperable Sobel filters
	print("Computing Gradients...")
	x_grad, y_grad = compute_gradients(blurred_image, sobel_x)

	# Resize x_grad to be the same shape as y_grad
	h,_ = x_grad.shape
	x_grad = x_grad[:h-2, :]

	# Compute the values for the M matrix
	Gxx = x_grad**2
	Gxy = x_grad*y_grad
	Gyy = y_grad**2

	# Loop through and find the corners
	print("Finding Corners...")

	# Set the size of the offset for the window
	offset = int(window_size/2) if int(window_size/2) > 0 else 1
	maximum = float("-inf")
	corners = []

	for y in range(offset, height-offset):
		for x in range(offset, width-offset):
			# Calculate the sum of gradients for each window
			sum_Gxx = np.sum(Gxx[y-offset:y+offset+1, x-offset:x+offset+1])
			sum_Gyy = np.sum(Gyy[y-offset:y+offset+1, x-offset:x+offset+1])
			sum_Gxy = np.sum(Gxy[y-offset:y+offset+1, x-offset:x+offset+1])

			# Calculate the determinant: ad-bc
			d = (sum_Gxx*sum_Gyy)-(sum_Gxy**2)

			# Calculate the trace
			trace = sum_Gxx + sum_Gyy

			# Calculate the Harris response and filter based on a threshold
			R = d - k*(trace**2)
			if R > threshold:
				# Add the x and y gradients around each point to be used in correspondence matching in RANSAC
				corners.append((x,y,R, x_grad[y-offset:y+offset+1, x-offset:x+offset+1], 
					y_grad[y-offset:y+offset+1, x-offset:x+offset+1]))

			maximum = max(maximum, R)

	print("MAX: "+ str(maximum))
	print("Number of corners: " + str(len(corners)))

	# Sort corners by their Harris Response Score
	corners.sort(key=lambda x: x[2], reverse=True)

	# Filter out redundant corners
	if len(corners) > 1:
		filtered_corners = []
		delete_counter = 0

		i = j = 0
		while i < len(corners)-1:
			j = i + 1
			while j < len(corners):
				# Distance formula of edges
				distance = math.sqrt(math.pow(corners[i][0]-corners[j][0], 2) + math.pow(corners[i][1]-corners[j][1], 2))

				# Remove corners that are within 10 pixels of eachother and reset j
				if distance <= 5:
					corners.remove(corners[j])
					j -= 1
				j += 1
			i += 1

	print("Number of corners after filtering out redundancies: " + str(len(corners)))

	# Take the best corners
	corners = corners[:max_corners] 

	# Draw the corners
	if visualize:
		for c in corners:
			draw = ImageDraw.Draw(original)
			draw.ellipse((c[0]-offset, c[1]-offset, c[0]+offset, c[1]+offset), fill = 'red')

		original.show()

	return corners


def compute_gradients(img, filter_X):
	filter_Y = filter_X.T
	height, width = img.shape
	offset = len(filter_X)

	# First apply the x filter
	x_gradient_intensity = []
	for i in range(height):
		x_gradient_intensity.append([])
		for j in range(width-offset):
			value = np.dot(filter_X, img[i][j:j+offset])
			x_gradient_intensity[i].append(value)

	# Now apply the y filter
	y_gradient_intensity = []

	for i in range(height-offset):
		y_gradient_intensity.append([])
		for j in range(width-offset):
			value = np.dot(filter_Y, img[i:i+offset, j])
			y_gradient_intensity[i].append(value)

	return np.array(x_gradient_intensity), np.array(y_gradient_intensity)


def gaussian_blur(img, width, height, sigma_squared):
	im = np.array(img)

	# Create the Guassian Filter Size
	sigma_sq = float(sigma_squared)
	filter_size = 6*int(math.sqrt(sigma_sq)) if 6*int(math.sqrt(sigma_sq)) % 2 == 1 else 6*int(math.sqrt(sigma_sq)) + 1
	k = int((filter_size-1)/2)

	# Compute the seperable guassian filter
	gaussian_filter = []
	total = 0
	for i in range(filter_size):
		# Compute the x Gaussian Values using the formula
		exponent = -1*((math.pow((i-(k+1)), 2))/2*sigma_sq)
		gaussian_value = 1/(math.sqrt(2*math.pi*sigma_sq))*pow(math.e, exponent)
		gaussian_filter.append(gaussian_value)

	# Apply Gaussian Smoothing to the padded image to filter out noise using 2 seperable filters

	# First Apply Smoothing to original image with the x filter
	blurred_image = []
	for i in range(height):
		blurred_image.append([])
		for j in range(width-filter_size+1):
			total +=1
			value = np.dot(gaussian_filter, im[i][j:j+filter_size])
			blurred_image[i].append(value)

	gaussian_blur = []	

	# Apply Smotthing to the x_filtered image with the y filter
	for index in range(len(blurred_image)-filter_size+1):
		gaussian_blur.append([])
		for j in range(len(blurred_image[0])):
			y_slice = []

			for z in range(index, index + len(gaussian_filter)):
				y_slice.append(blurred_image[z][j])

			value = np.dot(gaussian_filter, y_slice)
			gaussian_blur[index].append(value)

	return np.array(gaussian_blur)




# Pad an image by a certain amount of pixels on all four sides
def pad_image(im, num_padding):
	pixels = np.asarray(im)
	width, height = im.size

	padded_pixels = np.resize((pixels), (height + 2*num_padding, width + 2*num_padding))

	padded_pixels_height, padded_pixels_width = padded_pixels.shape

	# Add padding to the image
	top_border = pixels[0]
	bottom_border = pixels[len(pixels)-1]
	left_border = [row[0] for row in pixels]
	right_border = [row[len(pixels[0])-1] for row in pixels]

	for i in range(padded_pixels_height):
		for j in range(padded_pixels_width):
			if i < num_padding:
				# Top Middle
				if j >= num_padding and j < padded_pixels_width - num_padding:
					padded_pixels[i][j] = top_border[j-num_padding]
				elif j < num_padding: 
					# Top Left Corner
					padded_pixels[i][j] = top_border[0]
				else:
					# Top Right Corner
					padded_pixels[i][j] = top_border[len(top_border)-1]
			elif i > height - 1 + num_padding:
				# Bottom Middle
				if j >= num_padding and j < padded_pixels_width - num_padding:
					padded_pixels[i][j] = bottom_border[j-num_padding]
				elif j < num_padding: 
					# Bottom Left Corner
					padded_pixels[i][j] = bottom_border[0]
				else:
					# Bottom Right Corner
					padded_pixels[i][j] = bottom_border[len(bottom_border)-1]
			elif j >= padded_pixels_width - num_padding:
				# Right Border
				padded_pixels[i][j] = right_border[i-num_padding]
			elif j < num_padding:
				# Left Border
				padded_pixels[i][j] = left_border[i-num_padding]
			else:
				padded_pixels[i][j] = pixels[i-num_padding][j-num_padding]

	return padded_pixels, padded_pixels_width, padded_pixels_height


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='code')
    parser.add_argument('--img', type=str, default="chess_board.png",
                        help='path to image')

    parser.add_argument('--img2', type=str, default="chess_board.png",
                        help='path to image2 if creating a panaorama')

    parser.add_argument('--windowsize', type=int, default=5,
                        help='the size of the window')

    parser.add_argument('--k', type=float, default=0.05,
                        help='the cofficient k typically between 0.04 and 0.06')

    parser.add_argument('--threshold', type=float, default=20_000_000,
                        help='the threshold for the Harris response')

    parser.add_argument('--sigmasq', type=float, default=1.0,
                        help='the value of sigma squared for the gaussian blur')

    parser.add_argument('--keypoints', type=int, default=500,
                        help='the maximum number of corners the algorithm will display')

    parser.add_argument('--run_panorama', type=bool, default=False, 
    					help='True if you want to create a panaorama or do not speficy if you just want to run harris corner detector')

    parser.add_argument('--visualize', type=bool, default=False, 
    					help='True if you want to show correspondences between two images or corners of a single image')
    args = parser.parse_args()

    if args.run_panorama:
    	ransac(args.img, args.img2, args.visualize)
    else:
    	harris_corner_detector(args.img, args.windowsize, args.k, args.threshold, args.sigmasq, args.keypoints, args.visualize)

