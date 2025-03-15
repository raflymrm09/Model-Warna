import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan gambar
def show_images(images, titles, cmap=None):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Load gambar (gantilah 'image.jpg' dengan path gambar yang sesuai)
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Citra Negatif
negative_image = 255 - image

# 2. Transformasi Log
c = 255 / np.log(1 + np.max(image))
log_image = c * np.log(1 + image.astype(np.float32))
log_image = np.uint8(log_image)

# 3. Transformasi Power Law (Gamma Correction)
gamma = 2.2
power_image = np.array(255 * (image / 255) ** gamma, dtype='uint8')

# 4. Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# 5. Histogram Normalization
norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

# 6. Konversi RGB ke HSI (Opsional)
def rgb_to_hsi(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    r, g, b = cv2.split(image)
    intensity = (r + g + b) / 3
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = 1 - (3 / (r + g + b + 1e-6)) * min_rgb
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6
    theta = np.arccos(numerator / denominator)
    hue = np.where(b <= g, theta, 2 * np.pi - theta) / (2 * np.pi)
    hsi_image = cv2.merge((hue, saturation, intensity))
    return hsi_image

hsi_image = rgb_to_hsi(cv2.imread('image.jpg'))

# Menampilkan hasil
show_images([image, negative_image, log_image, power_image, equalized_image, norm_image],
            ['Original', 'Negative', 'Log Transform', 'Power Law', 'Histogram Equalization', 'Histogram Normalization'],
            cmap='gray')
