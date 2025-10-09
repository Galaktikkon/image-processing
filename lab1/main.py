import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity


def create_emoticons(emoticon: str):
    img = np.zeros((64, 64), dtype=np.uint8)

    cv2.circle(img, (32, 32), 30, 255, 2)

    cv2.circle(img, (22, 24), 3, 255, -1)
    cv2.circle(img, (42, 24), 3, 255, -1)

    if emoticon == "smile":
        cv2.ellipse(img, (32, 40), (12, 5), 0, 0, 180, 255, 2)
    elif emoticon == "sad":
        cv2.ellipse(img, (32, 45), (12, 5), 0, 180, 360, 255, 2)
    elif emoticon == "surprise":
        cv2.circle(img, (32, 40), 5, 255, 2)
    elif emoticon == "wink":
        cv2.line(img, (18, 24), (26, 24), 255, 2)
        cv2.circle(img, (42, 24), 3, 255, -1)
        cv2.ellipse(img, (32, 40), (12, 5), 0, 0, 180, 255, 2)
    elif emoticon == "neutral":
        cv2.line(img, (20, 45), (44, 45), 255, 2)
    else:
        raise ValueError("Unknown emoticon type")
    return img


def distort_image(img: np.ndarray, noise_level: float = 10, deform_shift: float = 2):
    noisy = img.astype(np.float32)
    noisy += np.random.normal(0, noise_level, img.shape)
    noisy = np.clip(noisy, 0, 225)

    rows, cols = img.shape

    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])

    dst_points = src_points + np.random.randint(
        -deform_shift, deform_shift + 1, src_points.shape
    ).astype(np.float32)

    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    distorted = cv2.warpAffine(
        noisy, affine_matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT
    )

    return distorted.astype(np.uint8)


def extract_fft_descriptor(img: np.ndarray, keep_fraction: float = 0.25):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    log_mag = np.log(mag + 1)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    r_keep, c_keep = int(crow * keep_fraction), int(ccol * keep_fraction)
    log_mag[crow - r_keep : crow + r_keep, ccol - c_keep : ccol + c_keep] = 0
    descriptor = log_mag.flatten()

    return descriptor


def affine_transform(img: np.ndarray, angle: float, scale: float, tx: float, ty: float):
    rows, cols = img.shape
    center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    transformed = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return transformed


def get_database():
    emoticons = ["smile", "sad", "surprise", "wink", "neutral"]
    database = []
    labels = []

    for emoticon in emoticons:
        base_img = create_emoticons(emoticon)
        for _ in range(20):
            distorted_img = distort_image(base_img)
            descriptor = extract_fft_descriptor(distorted_img)
            database.append(descriptor)
            labels.append(emoticon)

    return np.array(database), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Emoticon Classifier")
    parser.add_argument(
        "--test_emoticon",
        type=str,
        default="smile",
        choices=["smile", "sad", "surprise", "wink", "neutral"],
        help="Type of emoticon to test",
    )
    args = parser.parse_args()

    database, labels = get_database()
    knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    knn.fit(database, labels)

    test_img = create_emoticons(args.test_emoticon)
    test_img = distort_image(test_img)
    test_img = affine_transform(
        test_img,
        angle=np.random.uniform(-15, 15),
        scale=np.random.uniform(0.9, 1.1),
        tx=np.random.randint(-3, 4),
        ty=np.random.randint(-3, 4),
    )
    test_descriptor = extract_fft_descriptor(test_img).reshape(1, -1)

    predicted_label = knn.predict(test_descriptor)[0]
    print(f"Predicted Emoticon: {predicted_label}")

    plt.subplot(1, 2, 1)
    plt.title("Test Image")
    plt.imshow(test_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Predicted: {predicted_label}")
    plt.imshow(create_emoticons(predicted_label), cmap="gray")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
