from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

image_path = 'Z:/IDRiD_32.png'


from skimage.io import imread
image = imread(image_path, as_gray=True)

binary_image = image > 0

skeleton = skeletonize(binary_image)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Оригинальное изображение')
ax[0].axis('off')

ax[1].imshow(skeleton, cmap='gray')
ax[1].set_title('Скелетизированное изображение')
ax[1].axis('off')

plt.tight_layout()
plt.show()
