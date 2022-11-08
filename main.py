import cv2
import numpy as np
from matplotlib import pyplot as plt
import statistics

params1 = {'image': 'test.png', 'start_x': 90, 'start_y': 190}
params2 = {'image': '2020.png', 'start_x': 420, 'start_y': 335}
params3 = {'image': '2021.png', 'start_x': 560, 'start_y': 160}
params4 = {'image': '2022.png', 'start_x': 630, 'start_y': 177}
params5 = {'image': 'test2.png', 'start_x': 100, 'start_y': 950}

params = params3

distance_type = 'mahalanobis'
connection_type = "four_neighbourhood"

training_set_box_radius = 6

# mahalanobis distance params
tolerance = 5

# uniform distance params
threshold_rgb = 35

# update reference params
update_ref = False
update_frequency = 200

# plot params
preview = False
rt_plot = False
plot_frequency = 200

check_bin = []
river_in = []
river_out = []
check_bin_old = []

threshold_r = 0
threshold_g = 0
threshold_b = 0

ref_r = 0
ref_g = 0
ref_b = 0

mahalanobis_box_plot = []


def plot_rt_scan():
    # this funciton is needed for debugging purposes
    for position in check_bin:
        # this function marks with a red marker every pixel in the check bin, updated in real time
        plt.plot(position[1], position[0], marker='.', color="red", markersize=1)
    for position in river_out:
        # this function marks with a violet marker every pixel in the river bank, updated in real time
        plt.plot(position[1], position[0], marker='.', color="violet", markersize=1)
    for position in mahalanobis_box_plot:
        # this function marks with a green marker all the pixels beloning to the mahalanobis box used to calculate the
        # standard deviation of each channel
        plt.plot(position[1], position[0], marker='.', color="green", markersize=1)
    plt.plot(params['start_y'], params['start_x'], marker='v', color="blue")
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(.1)


class RiverPoint:
    generation_n = 0
    position = (0, 0)
    father_direction_trend = (0, 0)
    children_directions = []

    def __init__(self, generation_n, point_position, father_direction_trend=(0, 0)):
        self.generation_n = generation_n
        self.position = point_position
        self.father_direction_trend = father_direction_trend

    def add_direction_child(self, children_position):
        direction = self.position - children_position
        self.children_directions.append(direction)

    def get_movement_trend(self):
        trend = self.father_direction_trend * 2
        for d in self.children_directions:
            trend = trend + d
        trend = trend / (len(self.children_directions) + 2)
        return trend


def calc_distance(start_x, start_y):
    global distance_type
    global tolerance

    # the variable training_set_box_radius determines the size of the box used to learn the characteristics
    # of the colour of the river surface. Its width and height are hence calculated and used to determine
    # the position of its pixels.
    training_set = []
    x_width = list(range(start_x - training_set_box_radius, start_x + training_set_box_radius + 1))
    y_width = list(range(start_y - training_set_box_radius, start_y + training_set_box_radius + 1))
    for x in x_width:
        for y in y_width:
            p = (x, y)
            if 0 <= x < image_original.shape[0] and 0 <= y < image_original.shape[1]:
                training_set.append(p)

    # so in this case the training set is a box centered in the starting point
    # its side has length equal to stdev_box_size * 2 + 1

    if distance_type == 'uniform':
        # this distance gives the same threshold to each channel, the radius is the distance wrt the center
        set_uniform_thresholds(training_set, threshold_rgb)

    if distance_type == 'mahalanobis':
        # this is my implementation of the mahalanobis distance

        set_mahalanobis_thresholds(training_set, tolerance)

        # this is for debug and visualization of the box used for standard deviation
        for p in training_set:
            if \
                    p[0] == start_x - training_set_box_radius or \
                            p[0] == start_x + training_set_box_radius or \
                            p[1] == start_y - training_set_box_radius or \
                            p[1] == start_y + training_set_box_radius:
                mahalanobis_box_plot.append(p)

        return


def set_uniform_thresholds(training_set, thr):
    global ref_r, ref_g, ref_b
    global threshold_r, threshold_g, threshold_b

    # the three channels are here separated to apply independent calculations of the mean
    r_channel = []
    g_channel = []
    b_channel = []
    for p in training_set:
        r_channel.append(image_original[p[0], p[1], 0])
        g_channel.append(image_original[p[0], p[1], 1])
        b_channel.append(image_original[p[0], p[1], 2])

    # calculate the mean for each vector, this will be our reference colour for the distance calculation
    ref_r = statistics.mean(r_channel)
    ref_g = statistics.mean(g_channel)
    ref_b = statistics.mean(b_channel)

    # the distance is the same on each channel and defined by the global variable threshold
    threshold_r = thr
    threshold_g = thr
    threshold_b = thr
    return


def set_mahalanobis_thresholds(training_set, tolerance):
    global ref_r, ref_g, ref_b
    global threshold_r, threshold_g, threshold_b

    # the three channels are here separated to apply independent calculations of the standard deviations
    r_channel = []
    g_channel = []
    b_channel = []
    for p in training_set:
        r_channel.append(image_original[p[0], p[1], 0])
        g_channel.append(image_original[p[0], p[1], 1])
        b_channel.append(image_original[p[0], p[1], 2])

    # calculate the mean for each vector, this will be our reference colour for the distance calculation
    ref_r = statistics.mean(r_channel)
    ref_g = statistics.mean(g_channel)
    ref_b = statistics.mean(b_channel)

    # calculate the standard deviation and the thresholds for each channel, given by sigma * 3.
    # sometimes an empirical correction factor is necessary to be applied to the thresholds
    # in order to get some tolerance on the computed thresholds

    r_stdev = statistics.stdev(r_channel)
    g_stdev = statistics.stdev(g_channel)
    b_stdev = statistics.stdev(b_channel)

    threshold_r = r_stdev * 3 + tolerance
    threshold_g = g_stdev * 3 + tolerance
    threshold_b = b_stdev * 3 + tolerance

    return


def update_reference():
    global threshold_rgb, tolerance
    # This function is used to update the reference colour for the distance estimation

    if distance_type == 'uniform':
        set_uniform_thresholds(check_bin, threshold_rgb)

    if distance_type == 'mahalanobis':
        set_mahalanobis_thresholds(check_bin, tolerance)

    return


def barycentre_calc():
    # the next steps are needed in order to calculate the barycentre of the distribution represented by the check bin.
    sum_x = 0
    sum_y = 0
    for p in check_bin:
        sum_x = sum_x + p[0]
        sum_y = sum_y + p[1]
    barycentre = (sum_x / len(check_bin), sum_y / len(check_bin))
    reference = check_bin[0]
    reference_distance = abs(check_bin[0][0] - barycentre[0]) + abs(check_bin[0][1] - barycentre[1])

    # From the geometrical barycentre the closest pixel belonging to the check bin will be selected to be the new
    # reference point.
    # Since we are scanning a river this could be useful in order to get its center as a reference,
    # since the river banks colour is usually quite different from the one in the center,
    # because of the diminishing depth level.

    for p in check_bin:
        distance = abs(p[0] - barycentre[0]) + abs(p[1] - barycentre[1])
        if distance < reference_distance:
            reference = p
            reference_distance = distance

    return reference[0], reference[1]


def expand_check_bin():
    # a check bin is a set of pixels that need to be checked in order to determine if each given pixel belongs to
    # the river or not. The criteria are the calculated distances (classical or mahalanobis) from the reference colour.
    global check_bin
    global check_bin_old

    # here a new check bin is created. This is a cumulative iteration so this set is built expanding every pixel in the
    # set, with no repetitions. An expansion is a neighbourhood of a pixel.
    new_check_bin = []
    for position in check_bin:
        # this function builds up the 4-n or 8-n of every pixel in the set, then adds the neighbourhood to the check bin
        expansion = build_expansion(position, new_check_bin)
        new_check_bin = new_check_bin + expansion

    # here we store the old check bin before overwriting it with the new one. We will need this information in order to
    # prevent to process the same pixel twice.
    check_bin_old = check_bin
    check_bin = new_check_bin


def build_expansion(position, new_checkbin):
    global connection_type
    # extracting the single coordinates from the position
    x = position[0]
    y = position[1]
    # building the new expansion on the neighbourhood of the pixel detected by the given position
    expansion = []
    combs = []

    if connection_type == "four_neighbourhood":
        combs = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    if connection_type == "eight_neighbourhood":
        combs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    for p in combs:
        # connection type contains all the possible position combinations for the pixel surrounding the given one
        # just summing those to the coordinates pf the pixels gives the entire neighbourhood
        # depending on the neighbourhood kind this could be (up to) 4 points or 8 points
        # in the first case the resulting river will be an 8-connected blob, otherwise il will be 4-connected
        new_pos = verify_position((x + p[0], y + p[1]), new_checkbin)
        if new_pos is not None:
            expansion.append(new_pos)
    return expansion


def verify_position(position_to_check, new_checkbin):
    # this function checks if the just created neighbour pixel coordinate couple si valid or not.
    # invalid conditions are for instance the pixel being outside the image or belonging to the expansion of an
    # adjacent pixel.
    if \
            position_to_check not in check_bin and \
            position_to_check not in check_bin_old and \
            position_to_check not in new_checkbin:
        if 0 <= position_to_check[0] < image_original.shape[0] and 0 <= position_to_check[1] < image_original.shape[1]:
            return position_to_check
    return None


def check(bin_to_check=None):
    global ref_r, ref_g, ref_b
    global check_bin_old
    global check_bin
    global river_in
    global river_out
    # this function checks every pixel in the check bin (passed has as a global variable) and determines is it belongs
    # to the river or not.
    if bin_to_check is not None:
        my_bin = bin_to_check
    else:
        my_bin = check_bin

    for pixel in my_bin:
        # each channel of the reference colour is combined in a reference pixel colour (R,G,B)
        ref_pixel_colour = [ref_r, ref_g, ref_b]
        # a new pixel from the check bin is extracted and the distance for each channel is calculated
        extracted_pixel_colour = [int(x) for x in image_original[pixel]]

        dis_r = abs(ref_pixel_colour[0] - extracted_pixel_colour[0])
        dis_g = abs(ref_pixel_colour[1] - extracted_pixel_colour[1])
        dis_b = abs(ref_pixel_colour[2] - extracted_pixel_colour[2])

        # check if the distances are all below the threshold calculated with
        # the distance method selected (classic or mahalanobis)
        if dis_r < threshold_r and dis_g < threshold_g and dis_b < threshold_b:
            # the next calculations are needed to convert the distance in a 0-255 scale for graphic purposes
            avg_distance = statistics.mean([dis_r, dis_g, dis_b])
            avg_threshold = statistics.mean([threshold_r, threshold_g, threshold_b])
            val = 255 * (avg_distance/avg_threshold)
            # the value shifts from red to green the more the distance is close to the reference point
            image_river[pixel] = (val, 255 - val, 0)

            # this is a variable that contains all the points that belongs to the river
            river_in.append(pixel)

        else:
            # those pixels are the ones belonging to the river banks, coloured here in blue
            river_out.append(pixel)
            image_river[pixel] = (0, 0, 255)

    # the last step is removing all the pixels that do not belong to the river from the check bin,
    # for the next expansion
    check_bin = [x for x in my_bin if x not in river_out]
    return


# Initializing the algorithm
# Importing the selected image for the processing
image_original = cv2.imread(params['image'], cv2.IMREAD_COLOR)
image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
image_river = np.copy(image_original)

# Selecting a reference point that will be the starting point for the algorithm
# and the reference center for the colour search
ref = (params['start_x'], params['start_y'])

# This option is used in order to show a preview of the river highlighting the starting point,
# to be sure is has been selected correctly
if preview:
    plt.figure()
    plt.imshow(image_river)
    plt.plot(params['start_y'], params['start_x'], marker='v', color="red")
    plt.show()

# this structure contains all the point that will be checked to determine
# if they belong to the river or not
check_bin.append(ref)

# calculating the distances in terms og colour. This gives different output
# depending on the algorithm chosen (radial distance or mahalanobis distance)
calc_distance(params['start_x'], params['start_y'])
print("current reference colour: ({}, {}, {})".format(ref_r, ref_g, ref_b))

# if this option is active you can visualize the evolution of the scan in real time
if rt_plot:
    plt.ion()
    fig = plt.figure()
    plt.imshow(image_original)

# this is the main cicle who determines the gradual expansion fo the check bin
# (the set of the point that will be checked)
n = 0
while len(check_bin) > 0:
    # for each iteration the check bin is expanded and then checked
    # the every time a pixel is analyzed it is removed from the check bin, so we know
    # the elaboration is over when the check bin is empty
    expand_check_bin()
    check()
    # this print is useful to keep track of the ongoing iteration
    print("iteration {}".format(n))
    # if the rt plot in enabled this will show an update every 200 iterations
    # showing too many iterations will slow down the algorithm
    if rt_plot is True and n % plot_frequency == 0:
        plot_rt_scan()
    # this is essential to update the reference colour used to determine if a point
    # belongs to the river or not. This was needed in order to compensate colour variations
    # given by variations in the depth of the river
    if update_ref and n % update_frequency == 0 and n > 0:
        # this update is done at a given frequency. It can also be disabled in order to do the computations faster.
        update_reference()
        print("current reference colour: ({}, {}, {})".format(ref_r, ref_g, ref_b))
    n += 1

# this is a debuggin plot, which is needed in order to check if the procedure was correct
# and the visualization of the starting point
# plot_rt_scan()

# if this plot option was enabled it can be ended now
if rt_plot:
    plot_rt_scan()
    plt.ioff()

# showing the final result collecting all the pixels belonging to the river set
print("plotting river image...")
plt.figure()
plt.imshow(image_river)
plt.show()
