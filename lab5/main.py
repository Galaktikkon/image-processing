import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA


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
    else:
        raise ValueError("Unknown emoticon type")
    return img


def affine_transform(img: np.ndarray, angle: float, scale: float, tx: float, ty: float):
    rows, cols = img.shape
    center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    transformed = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return transformed


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


def plot_images(original: np.ndarray, distorted: np.ndarray, restored: np.ndarray):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Distorted Image")
    plt.imshow(distorted, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Transformed Image")
    plt.imshow(restored, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def get_images():
    emoticons = ["smile", "sad", "surprise"]
    images = [create_emoticons(e) for e in emoticons]
    return images


def get_random_emoticon_image():
    emoticons = [
        "smile",
        "sad",
        "surprise",
    ]
    choice = np.random.choice(emoticons)
    image = create_emoticons(choice)
    return distort_image(
        affine_transform(image, angle=15, scale=1.1, tx=5, ty=-3),
        noise_level=15,
        deform_shift=3,
    )


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


def extract_haar_descriptor(img: np.ndarray, wavelet: str = "haar", levels: int = 2):
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=levels)
    arr, _ = pywt.coeffs_to_array(coeffs)
    return arr.flatten()


def get_descriptor_PCA(descriptor: np.ndarray, n_components: int = 20):
    if descriptor.ndim == 1:
        descriptor = descriptor.reshape(1, -1)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(descriptor)
    return reduced.flatten(), pca


def get_database(variations: int = 10):
    labels = ["smile", "sad", "surprise"]
    for emoticon in labels:
        class_images = []
        for i in range(variations):
            image = create_emoticons(emoticon)
            distorted_image = distort_image(
                affine_transform(
                    image,
                    angle=np.random.uniform(-20, 20),
                    scale=np.random.uniform(0.9, 1.1),
                    tx=np.random.uniform(-5, 5),
                    ty=np.random.uniform(-5, 5),
                ),
                noise_level=15,
                deform_shift=3,
            )
            class_images.append((distorted_image, emoticon))
    return class_images


def get_KNN_models(fft_descriptors, dwt_descriptors, concat_descriptors, labels):
    knn_fft = KNeighborsClassifier(n_neighbors=1).fit(fft_descriptors, labels)
    knn_dwt = KNeighborsClassifier(n_neighbors=1).fit(dwt_descriptors, labels)
    knn_concat = KNeighborsClassifier(n_neighbors=1).fit(concat_descriptors, labels)
    return knn_fft, knn_dwt, knn_concat


def normalize_vector(vec):
    v = np.asarray(vec)
    return (v - v.min()) / (v.max() - v.min() + 1e-8)


class_images = get_database(variations=40)
images, labels = zip(*class_images)
labels = list(labels)

fft_descriptors = [normalize_vector(extract_fft_descriptor(img)) for img in images]
dwt_descriptors = [normalize_vector(extract_haar_descriptor(img)) for img in images]

fft_norm = normalize_vector(np.array(fft_descriptors))
dwt_norm = normalize_vector(np.array(dwt_descriptors))

concat_norm = np.concatenate([fft_norm, dwt_norm], axis=1)

n_components = 5
fft_pca, fft_pca_model = get_descriptor_PCA(
    np.array(fft_descriptors), n_components=n_components
)
dwt_pca, dwt_pca_model = get_descriptor_PCA(
    np.array(dwt_descriptors), n_components=n_components
)
concat_pca, concat_pca_model = get_descriptor_PCA(
    concat_norm, n_components=n_components
)


knn_fft, knn_dwt, knn_concat = get_KNN_models(
    fft_pca.reshape(len(labels), -1),
    dwt_pca.reshape(len(labels), -1),
    concat_pca.reshape(len(labels), -1),
    np.arange(len(labels)),
)


def plot_results(query_img, result_fft, result_dwt, result_concat):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("Query Image")
    plt.imshow(query_img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title(f"FFT Result: {result_fft}")
    plt.imshow(create_emoticons(result_fft), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title(f"DWT Result: {result_dwt}")
    plt.imshow(create_emoticons(result_dwt), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title(f"Concat Result: {result_concat}")
    plt.imshow(create_emoticons(result_concat), cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_descriptors(fft, fft_pca, dwt, dwt_pca, concat, concat_pca):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.title("FFT Descriptor")
    plt.plot(fft)
    plt.subplot(3, 2, 2)
    plt.title("FFT PCA Descriptor")
    plt.plot(fft_pca)
    plt.subplot(3, 2, 3)
    plt.title("DWT Descriptor")
    plt.plot(dwt)
    plt.subplot(3, 2, 4)
    plt.title("DWT PCA Descriptor")
    plt.plot(dwt_pca)
    plt.subplot(3, 2, 5)
    plt.title("Concat Descriptor")
    plt.plot(concat)
    plt.subplot(3, 2, 6)
    plt.title("Concat PCA Descriptor")
    plt.plot(concat_pca)
    plt.tight_layout()
    plt.show()


query_img = get_random_emoticon_image()

query_fft = normalize_vector(extract_fft_descriptor(query_img))
query_dwt = normalize_vector(extract_haar_descriptor(query_img))

query_fft_pca = fft_pca_model.transform(query_fft.reshape(1, -1))
query_dwt_pca = dwt_pca_model.transform(query_dwt.reshape(1, -1))

query_fft_norm = normalize_vector(query_fft.reshape(1, -1))
query_dwt_norm = normalize_vector(query_dwt.reshape(1, -1))

query_concat_norm = np.concatenate([query_fft_norm, query_dwt_norm], axis=1)
query_concat_pca = concat_pca_model.transform(query_concat_norm)

result_fft = labels[knn_fft.predict(query_fft_pca)[0]]
result_dwt = labels[knn_dwt.predict(query_dwt_pca)[0]]
result_concat = labels[knn_concat.predict(query_concat_pca)[0]]

plot_results(query_img, result_fft, result_dwt, result_concat)
plot_descriptors(
    query_fft,
    query_fft_pca.flatten(),
    query_dwt,
    query_dwt_pca.flatten(),
    query_concat_norm.flatten(),
    query_concat_pca.flatten(),
)
