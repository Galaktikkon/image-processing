import matplotlib.pyplot as plt
import numpy as np
import cv2

IMAGE_PATH = "data/lena.jpg"


def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


def convolve2d(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def correlate2d(image, kernel):
    kernel_flipped = np.flip(kernel)
    return cv2.filter2D(image, -1, kernel_flipped)


def visualize_convolution_correlation(image, kernels):
    n = len(kernels)
    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    for i, (kname, kernel) in enumerate(kernels):
        conv_result = convolve2d(image, kernel)
        corr_result = correlate2d(image, kernel)
        axs[i, 0].imshow(image, cmap="gray")
        axs[i, 0].set_title("Original")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(conv_result, cmap="gray")
        axs[i, 1].set_title(f"Convolution\n({kname})")
        axs[i, 1].axis("off")
        axs[i, 2].imshow(corr_result, cmap="gray")
        axs[i, 2].set_title(f"Correlation\n({kname})")
        axs[i, 2].axis("off")
    plt.tight_layout()
    plt.show()


def visualize_template_matching(image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    th, tw = template.shape
    bottom_right = (top_left[0] + tw, top_left[1] + th)

    matched_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(matched_img, top_left, bottom_right, (0, 0, 255), 2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Template")
    plt.imshow(template, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Result (Heatmap)")
    plt.imshow(result, cmap="hot")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("Best Match")
    plt.imshow(matched_img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    image = load_image(IMAGE_PATH)

    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

    conv_result = convolve2d(image, kernel)
    corr_result = correlate2d(image, kernel)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Original")
    axs[0].axis("off")
    axs[1].imshow(conv_result, cmap="gray")
    axs[1].set_title("Convolution")
    axs[1].axis("off")
    axs[2].imshow(corr_result, cmap="gray")
    axs[2].set_title("Correlation")
    axs[2].axis("off")
    plt.tight_layout()
    plt.show()

    h, w = image.shape

    kernels = [
        (
            "Edge (Laplacian)",
            np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
        ),
        ("Sharpen", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)),
        ("Gaussian Blur", cv2.getGaussianKernel(3, 1) @ cv2.getGaussianKernel(3, 1).T),
        ("Emboss", np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)),
    ]

    visualize_convolution_correlation(image, kernels)

    h, w = image.shape
    th, tw = 60, 60
    center_y, center_x = h // 2, w // 2
    template = image[
        center_y - th // 2 : center_y + th // 2, center_x - tw // 2 : center_x + tw // 2
    ]

    visualize_template_matching(image, template)


if __name__ == "__main__":
    main()
