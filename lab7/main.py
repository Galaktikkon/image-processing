import cv2
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt


def create_emoticons(emoticon: str):
    img = np.zeros((64, 64), dtype=np.uint8)

    cv2.circle(img, (32, 32), 30, 255, 2)

    cv2.circle(img, (22, 24), 3, 255, -1)
    cv2.circle(img, (42, 24), 3, 255, -1)

    match emoticon:
        case "smile":
            cv2.ellipse(img, (32, 40), (12, 5), 0, 0, 180, 255, 2)
        case "sad":
            cv2.ellipse(img, (32, 45), (12, 5), 0, 180, 360, 255, 2)
        case "surprise":
            cv2.circle(img, (32, 40), 5, 255, 2)
        case "angry":
            cv2.line(img, (15, 15), (25, 20), 255, 2)
            cv2.line(img, (49, 15), (39, 20), 255, 2)
            cv2.ellipse(img, (32, 50), (12, 5), 0, 180, 360, 255, 2)
        case "neutral":
            cv2.line(img, (20, 45), (44, 45), 255, 2)
        case "wink":
            cv2.line(img, (18, 24), (26, 24), 255, 2)
            cv2.circle(img, (42, 24), 3, 255, -1)
            cv2.ellipse(img, (32, 40), (12, 5), 0, 0, 180, 255, 2)
        case "confused":
            cv2.circle(img, (22, 24), 3, 255, -1)
            cv2.line(img, (39, 22), (45, 26), 255, 2)
            cv2.ellipse(img, (32, 45), (12, 5), 0, 180, 360, 255, 2)
        case "laugh":
            cv2.circle(img, (22, 24), 3, 255, -1)
            cv2.circle(img, (42, 24), 3, 255, -1)
            cv2.ellipse(img, (32, 38), (12, 7), 0, 0, 180, 255, 2)
            cv2.ellipse(img, (32, 45), (12, 5), 0, 180, 360, 255, 2)
        case "cry":
            cv2.circle(img, (22, 24), 3, 255, -1)
            cv2.circle(img, (42, 24), 3, 255, -1)
            cv2.ellipse(img, (32, 45), (12, 5), 0, 180, 360, 255, 2)
            cv2.line(img, (25, 30), (20, 40), 255, 2)
            cv2.line(img, (39, 30), (44, 40), 255, 2)
        case "kiss":
            cv2.circle(img, (22, 24), 3, 255, -1)
            cv2.circle(img, (42, 24), 3, 255, -1)
            cv2.ellipse(img, (32, 45), (8, 5), 0, 0, 360, 255, 2)
            cv2.circle(img, (32, 52), 3, 255, -1)
        case "grin":
            cv2.circle(img, (22, 24), 3, 255, -1)
            cv2.circle(img, (42, 24), 3, 255, -1)
            cv2.ellipse(img, (32, 40), (15, 7), 0, 0, 180, 255, 2)
            cv2.line(img, (20, 50), (44, 50), 255, 2)
        case "sleepy":
            cv2.line(img, (18, 24), (26, 24), 255, 2)
            cv2.line(img, (38, 24), (46, 24), 255, 2)
            cv2.ellipse(img, (32, 45), (12, 5), 0, 180, 360, 255, 2)
        case _:
            pass

    return img


def affine_transform(img: np.ndarray, angle: float, scale: float, tx: float, ty: float):
    rows, cols = img.shape
    center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    transformed = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return transformed


def generate_dataset(emoticons: list[str], samples_per_emoticon: int):
    data = []
    labels = []

    for idx, emoticon in enumerate(emoticons):
        for _ in range(samples_per_emoticon):
            img = create_emoticons(emoticon)

            angle = np.random.uniform(-15, 15)
            scale = np.random.uniform(0.9, 1.1)
            tx = np.random.uniform(-5, 5)
            ty = np.random.uniform(-5, 5)

            img_transformed = affine_transform(img, angle, scale, tx, ty)

            data.append(img_transformed)
            labels.append(idx)

    return np.array(data), np.array(labels)


emoticons = [
    "smile",
    "sad",
    "surprise",
    "angry",
    "neutral",
    "wink",
    "confused",
    "laugh",
    "cry",
    "kiss",
    "grin",
    "sleepy",
]

data, labels = generate_dataset(emoticons, 1000)

np.random.shuffle(data)

x_test, x_train = data[:300], data[300:]

from tensorflow import keras
from tensorflow.keras import layers

input_img = keras.Input(shape=(64, 64, 1))

x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 32x32x16
x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 16x16x8
x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
encoded = layers.MaxPooling2D((2, 2), padding="same")(x)  # 8x8x8

x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
x = layers.UpSampling2D((2, 2))(x)  # 16x16x8
x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)  # 32x32x8
x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)  # 64x64x16
decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)  # 64x64x1

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))
x_test = np.reshape(x_test, (len(x_test), 64, 64, 1))

history = autoencoder.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test),
)

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss Over Epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()

from sklearn.manifold import TSNE

encoded_imgs = keras.Model(input_img, encoded).predict(x_train)

X_embedded = TSNE(
    n_components=2, learning_rate="auto", init="random", perplexity=3
).fit_transform(encoded_imgs.reshape((encoded_imgs.shape[0], -1)))

plt.figure(figsize=(10, 5))

for i in range(X_embedded.shape[0]):
    plt.scatter(
        X_embedded[i, 0],
        X_embedded[i, 1],
        c=plt.cm.get_cmap("tab10")(labels[i] / 10),
        label=emoticons[labels[i]] if i == 0 else "",
    )
plt.legend()

encoder = keras.Model(input_img, encoded)
encoded_imgs = encoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 8))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape((8, 8 * 8)).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
