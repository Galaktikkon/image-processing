import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.fftpack import dct, idct


IMAGE_PATH = "data/lena.jpg"


def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


def add_gaussian_noise(image, mean=0, std=75):
    noisy = image.astype(np.float32) + np.random.normal(mean, std, image.shape)
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def compute_dct(image):
    return dct(image, norm="ortho")


def getDCTImage(dct_image):
    return np.log(np.abs(dct_image) + 1)


def quantize_dct(dct_image, q=40):
    return np.round(dct_image / q)


def dequantize_dct(quantized_dct, q=40):
    return quantized_dct * q


def reconstruct_image(quantized_dct):
    dequantized = dequantize_dct(quantized_dct)
    return idct(dequantized, norm="ortho")


def display_images(images, titles, cmap="gray", max_cols=7, gap=(0.4, 0.6)):
    n = len(images)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    fig.subplots_adjust(wspace=gap[0], hspace=gap[1])
    plt.show()


def main():
    image = load_image(IMAGE_PATH)

    dct_image = compute_dct(image)
    quantized_dct = quantize_dct(dct_image)
    reconstructed_image = reconstruct_image(quantized_dct)
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

    noisy_image = add_gaussian_noise(image, mean=0, std=20)

    noisy_dct = compute_dct(noisy_image)
    quantized_noisy_dct = quantize_dct(noisy_dct)
    reconstructed_noisy_image = reconstruct_image(quantized_noisy_dct)
    reconstructed_noisy_image = np.clip(reconstructed_noisy_image, 0, 255).astype(
        np.uint8
    )

    display_images(
        [
            image,
            getDCTImage(dct_image),
            getDCTImage(quantized_dct),
            reconstructed_image,
            noisy_image,
            getDCTImage(noisy_dct),
            getDCTImage(quantized_noisy_dct),
            reconstructed_noisy_image,
        ],
        [
            "Original Image",
            "DCT of Original",
            "Quantized DCT",
            "Reconstructed DCT Image",
            "Noisy Image",
            "DCT of Noisy",
            "Quantized DCT (Noisy)",
            "Reconstructed DCT (Noisy)",
        ],
        max_cols=4,
    )


if __name__ == "__main__":
    main()
