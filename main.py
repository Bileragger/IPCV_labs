import cv2
import numpy as np
from matplotlib import pyplot as plt

image_grey = cv2.imread('2020.png', cv2.COLOR_BGR2GRAY)
img2020 = cv2.imread('2020.png', cv2.COLOR_BGR2RGB)
img2021 = cv2.imread('2021.png', cv2.COLOR_BGR2RGB)
img2022 = cv2.imread('2022.png', cv2.COLOR_BGR2RGB)

threshold = 128
#image_grey[image_grey < threshold] = 0
#image_grey[image_grey > threshold] = 255

plt.figure(1)
plt.imshow(image_grey)

R = img2020[:,:,0]
G = img2020[:,:,1]
B = img2020[:,:,2]

print(image_grey.shape)

hist_R, bins_r = np.histogram(R.flatten(), 256, [0,256])
hist_G, bins_g = np.histogram(G.flatten(), 256, [0,256])
hist_B, bins_b = np.histogram(B.flatten(), 256, [0,256])

figure, axis = plt.subplots(3, 1)

axis[0].plot(hist_R, color='r')
axis[1].plot(hist_G, color='g')
axis[2].plot(hist_B, color='b')

fig = plt.figure(4, figsize=(4, 8))

fig.add_subplot(3, 1, 1)
plt.imshow(img2020)
fig.add_subplot(3, 1, 2)
plt.imshow(img2021)
fig.add_subplot(3, 1, 3)
plt.imshow(img2022)

plt.show()

