import numpy as np
import random
import math 
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

item = int(random.random() * len(train_X))

if len(sys.argv) > 1:
    item = int(sys.argv[1])

label = train_y[item]
source_points = train_X[item]

print("item " + str(item) + " - label is " + str(label))

zero_min = np.zeros((28, 28))
one_max = np.ones((28, 28))

def gaussian_noise(X, sigma):
    ''' adds a gaussian noise limited to 0 and 1 inclusive'''
    X_nonzero_indexes = np.nonzero(X)
    noise = np.random.normal(0, sigma, X.shape)
    copy = X.copy()
    copy[X_nonzero_indexes] = np.minimum(np.maximum(X[X_nonzero_indexes] + 
        noise[X_nonzero_indexes], zero_min[X_nonzero_indexes]), one_max[X_nonzero_indexes])
    return copy

def convert_to_rgb(gray_image, color_map):
    '''Convert gray image to RGB using the given color map'''
    s_m = pyplot.cm.ScalarMappable(cmap = color_map)
    img_shape = gray_image.shape
    flattened = gray_image.flatten()
    colors = s_m.to_rgba(flattened)
    result = np.zeros(flattened.shape + (3,))

    for i in range(len(flattened)):
        if flattened[i] > 0:
            result[i] = colors[i][:-1]
    return result.reshape(img_shape + (3,))

def create_colored_grid(image, depth, color_map):
    grid = np.zeros(image.shape + (depth, 3,))
    for z in range(depth):
        instance_points = gaussian_noise(image, 0.2)
        rgb_points = convert_to_rgb(instance_points, color_map)
        grid[:, :, z] = rgb_points
    return grid

color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu']

def transform_instance(instance, color_map, x_rot, y_rot, z_rot):

    print("Using color_map: " + color_map)

    grid = create_colored_grid(instance / 255.0, 28, color_map)

    result = rotate(grid, z_rot, y_rot, x_rot)

    return result

def print_grid(grid):

    grid_shape = grid.shape

    flattened = grid.reshape(((grid_shape[0] * grid_shape[1] * grid_shape[2]), 3))
    voxel_grid_array = np.zeros(len(flattened))

    for i in range(len(flattened)):
        temp = flattened[i]
        if temp[0] > 0 or temp[1] > 0 or temp[2] > 0:
            voxel_grid_array[i] = 1

    voxel_grid = voxel_grid_array.reshape((grid_shape[0], grid_shape[1], grid_shape[2]))

    fig = pyplot.figure()
    ax = fig.add_subplot(projection='3d')
    ax.azim = 15
    ax.dist = 8
    ax.elev = 75
    ax.voxels(voxel_grid, facecolors=grid)

    pyplot.show(block=False)

    return fig

def rotate(grid, z_ang, y_ang, x_ang):

    grid_shape = grid.shape

    result = np.zeros(grid_shape)

    x_lim = grid_shape[0]
    y_lim = grid_shape[1]
    z_lim = grid_shape[2]

    for i in range(x_lim):
        for j in range(y_lim):
            for k in range(z_lim):

                X = i
                Y = j
                Z = k

                if z_ang != 0:
                    x = i - x_lim / 2
                    y = j - y_lim / 2
                    X = int(math.floor(x*math.cos(z_ang) - y*math.sin(z_ang) + x_lim / 2))
                    Y = int(math.floor(x*math.sin(z_ang) + y*math.cos(z_ang) + y_lim / 2)) 

                if y_ang != 0:
                    x = X - x_lim / 2
                    z = Z - z_lim / 2
                    X = int(math.floor(x*math.cos(y_ang) - z*math.sin(y_ang) + x_lim / 2))
                    Z = int(math.floor(x*math.sin(y_ang) + z*math.cos(y_ang) + z_lim / 2)) 

                if x_ang != 0:
                    y = Y - y_lim / 2
                    z = Z - z_lim / 2
                    Y = int(math.floor(y*math.cos(x_ang) - z*math.sin(x_ang) + y_lim / 2))
                    Z = int(math.floor(y*math.sin(x_ang) + z*math.cos(x_ang) + z_lim / 2)) 

                if X >= 0 and Y >= 0 and Z >= 0 and X < x_lim and Y < y_lim and Z < z_lim:
                    result[X, Y, Z] = grid[i, j, k]

    return result

color_map = None
if len(sys.argv) > 2:
    test = sys.argv[2]
    if test in color_maps:
        color_map = test
    else:
        print(test + " was not recognized as a valid color scheme")

if not color_map:
    color_map = random.choice(color_maps)

x_rot = 0 * math.pi / 180.0
y_rot = 0 * math.pi / 180.0
z_rot = 0 * math.pi / 180.0

my_cube = transform_instance(source_points, color_map, x_rot, y_rot, z_rot)

fig = print_grid(my_cube)

name = input("Enter the name of image (empty to no save) : ").strip()

if name:
    file_path = "/tmp/" + name + '.png'
    fig.savefig(file_path, dpi=fig.dpi)
    print("Saved image at " + file_path)
