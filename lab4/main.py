import cv2
import numpy as np

import pywt

import matplotlib.pyplot as plt


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
    features = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            for arr in coeff:
                features.append(arr.flatten())
        else:
            features.append(coeff.flatten())
    return np.concatenate(features)


def extract_hybrid_descriptor(
    img: np.ndarray,
    keep_fraction: float = 0.25,
    wavelet: str = "haar",
    levels: int = 2,
    fusion: str = "concat",
):
    fft_desc = extract_fft_descriptor(img, keep_fraction)
    haar_desc = extract_haar_descriptor(img, wavelet, levels)
    if fusion == "concat":
        return np.concatenate([fft_desc, haar_desc])
    elif fusion == "sum":
        min_len = min(len(fft_desc), len(haar_desc))
        return fft_desc[:min_len] + haar_desc[:min_len]
    elif fusion == "mean":
        min_len = min(len(fft_desc), len(haar_desc))
        return (fft_desc[:min_len] + haar_desc[:min_len]) / 2
    else:
        raise ValueError(f"Unknown fusion method: {fusion}")


def get_database():
    emoticons = ["smile", "sad", "surprise", "wink", "neutral"]

    db_fft = []
    db_haar = []
    db_hybrid_concat = []
    db_hybrid_sum = []
    db_hybrid_mean = []
    labels = []

    for emoticon in emoticons:
        img = create_emoticons(emoticon)
        db_fft.append(extract_fft_descriptor(img))
        db_haar.append(extract_haar_descriptor(img))
        db_hybrid_concat.append(extract_hybrid_descriptor(img, fusion="concat"))
        db_hybrid_sum.append(extract_hybrid_descriptor(img, fusion="sum"))
        db_hybrid_mean.append(extract_hybrid_descriptor(img, fusion="mean"))
        labels.append(emoticon)

    return {
        "fft": np.array(db_fft),
        "haar": np.array(db_haar),
        "hybrid_concat": np.array(db_hybrid_concat),
        "hybrid_sum": np.array(db_hybrid_sum),
        "hybrid_mean": np.array(db_hybrid_mean),
        "labels": np.array(labels),
    }


def retrieve(query_desc, db_descs, labels):
    dists = np.linalg.norm(db_descs - query_desc, axis=1)
    idx = np.argmin(dists)
    return labels[idx], dists[idx]


def compare_retrieval():
    db = get_database()
    labels = db["labels"]
    emoticons = ["smile", "sad", "surprise", "wink", "neutral"]

    results = {
        "fft": [],
        "haar": [],
        "hybrid_concat": [],
        "hybrid_sum": [],
        "hybrid_mean": [],
    }
    dists = {
        "fft": [],
        "haar": [],
        "hybrid_concat": [],
        "hybrid_sum": [],
        "hybrid_mean": [],
    }

    for emoticon in emoticons:
        img = create_emoticons(emoticon)
        fft_q = extract_fft_descriptor(img)
        haar_q = extract_haar_descriptor(img)
        hybrid_concat_q = extract_hybrid_descriptor(img, fusion="concat")
        hybrid_sum_q = extract_hybrid_descriptor(img, fusion="sum")
        hybrid_mean_q = extract_hybrid_descriptor(img, fusion="mean")

        res_fft, dist_fft = retrieve(fft_q, db["fft"], labels)
        res_haar, dist_haar = retrieve(haar_q, db["haar"], labels)
        res_hc, dist_hc = retrieve(hybrid_concat_q, db["hybrid_concat"], labels)
        res_hs, dist_hs = retrieve(hybrid_sum_q, db["hybrid_sum"], labels)
        res_hm, dist_hm = retrieve(hybrid_mean_q, db["hybrid_mean"], labels)

        results["fft"].append(res_fft)
        results["haar"].append(res_haar)
        results["hybrid_concat"].append(res_hc)
        results["hybrid_sum"].append(res_hs)
        results["hybrid_mean"].append(res_hm)

        dists["fft"].append(dist_fft)
        dists["haar"].append(dist_haar)
        dists["hybrid_concat"].append(dist_hc)
        dists["hybrid_sum"].append(dist_hs)
        dists["hybrid_mean"].append(dist_hm)

        print(f"Query: {emoticon}")
        print(f"  FFT: {res_fft} (dist={dist_fft:.2f})")
        print(f"  Haar: {res_haar} (dist={dist_haar:.2f})")
        print(f"  Hybrid-concat: {res_hc} (dist={dist_hc:.2f})")
        print(f"  Hybrid-sum: {res_hs} (dist={dist_hs:.2f})")
        print(f"  Hybrid-mean: {res_hm} (dist={dist_hm:.2f})")
        print()

    methods = list(results.keys())
    correct = [
        sum([results[m][i] == emoticons[i] for i in range(len(emoticons))])
        for m in methods
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(
        methods, correct, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
    )
    plt.ylim(0, len(emoticons))
    plt.ylabel("Liczba poprawnych rozpoznań")
    plt.title("Porównanie skuteczności retrieval (FFT, Haar, Hybrid)")
    plt.show()


if __name__ == "__main__":
    compare_retrieval()
