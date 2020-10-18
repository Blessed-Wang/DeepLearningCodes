import numpy as np
import matplotlib.pyplot as plt


def convert_sequence_to_image(sequence):
    img = np.full([len(sequence), 256, 256], fill_value=255)
    for i in range(len(sequence)):
        img[i, sequence[i], np.arange(256)] = 0
    return img


if __name__ == '__main__':
    # img = np.full(shape=[10, 256, 256], fill_value=255)
    seq = np.random.randint(low=0, high=255, size=[100, 256])
    img = convert_sequence_to_image(seq)

    plt.imshow(255 - img[0], cmap='gray')
    plt.show()