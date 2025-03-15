import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan gambar dalam satu figure
def show_all_transformations(original, negative, log_transformed, power_transformed, hist_equalized, hist_normalized, hsi):
    titles = ['Original', 'Negative', 'Log Transform', 'Power Law (Gamma)', 'Histogram Equalization', 'Histogram Normalization', 'RGB to HSI']
    images = [original, negative, log_transformed, power_transformed, hist_equalized, hist_normalized, hsi]
    
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Load Gambar (Gantilah path sesuai lokasi file kamu)
image_path = "/content/download.jpg"  # Sesuaikan path ini
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. Citra Negatif
negative_image = 255 - image

# 2. Transformasi Log
c = 255 / np.log(1 + np.max(image))
log_image = c * np.log(1 + image.astype(np.float32))
log_image = np.uint8(log_image)

# 3. Transformasi Power Law (Gamma)
gamma = 2.0  # Bisa diubah sesuai kebutuhan
c = 255 / (np.max(image) ** gamma)
power_image = c * (image.astype(np.float32) ** gamma)
power_image = np.uint8(power_image)

# 4. Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# 5. Histogram Normalization
normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

# 6. Konversi RGB ke HSI
def rgb_to_hsi(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255

    R, G, B = cv2.split(img)

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(num / (den + 1e-6))

    H = np.where(B > G, 2 * np.pi - theta, theta) / (2 * np.pi) * 255
    S = 1 - 3 * np.minimum(R, np.minimum(G, B)) / (R + G + B + 1e-6)
    I = (R + G + B) / 3

    HSI = cv2.merge([H, S * 255, I * 255])
    return np.uint8(HSI)

hsi_image = rgb_to_hsi(image_path)

# Tampilkan Semua Gambar
show_all_transformations(image, negative_image, log_image, power_image, equalized_image, normalized_image, hsi_image)

