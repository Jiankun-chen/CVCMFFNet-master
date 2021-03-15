import os
import random
import numpy as np
import imageio
import glob
from tqdm import tqdm
from tqdm import trange

class GetData:
    def __init__(self, data_dir):
        file_list = []
        images1_list_real = []
        images1_list_imag = []
        images2_list_real = []
        images2_list_imag = []
        ang_list = []
        labels_list = []

        self.source_list = []

        examples = 0
        print("loading images")
        label_dir = os.path.join(data_dir, "Labels")
        image1_real_dir = os.path.join(data_dir, "Images", "real1")
        image1_imag_dir = os.path.join(data_dir, "Images", "imag1")
        image2_real_dir = os.path.join(data_dir, "Images", "real2")
        image2_imag_dir = os.path.join(data_dir, "Images", "imag2")
        ang_dir = os.path.join(data_dir, "Images", "ang")

        for label_root, dir, files in os.walk(label_dir):
            for file in files:
                if not file.endswith((".png", ".jpg", ".gif", "tif")):
                    continue
                try:
                    folder = os.path.relpath(label_root, label_dir)
                    image1_root_real = os.path.join(image1_real_dir, folder)
                    image1_root_imag = os.path.join(image1_imag_dir, folder)
                    image2_root_real = os.path.join(image2_real_dir, folder)
                    image2_root_imag = os.path.join(image2_imag_dir, folder)
                    ang_root = os.path.join(ang_dir, folder)

                    image1_real = imageio.imread(os.path.join(image1_root_real, file))
                    image1_imag = imageio.imread(os.path.join(image1_root_imag, file))
                    image2_real = imageio.imread(os.path.join(image2_root_real, file))
                    image2_imag = imageio.imread(os.path.join(image2_root_imag, file))
                    ang = imageio.imread(os.path.join(ang_root, file))

                    # image = np.array(Image.fromarray(image).resize((256, 256)))
                    # image = scipy.misc.imresize(image, 0.5)
                    label = imageio.imread(os.path.join(label_root, file))
                    # label = np.array(Image.fromarray(label).resize((256, 256)))
                    # label = scipy.misc.imresize(label, 0.5)

                    image1_1_real = np.stack((image1_real[..., 0], image1_real[..., 0], image1_real[..., 0]), axis=2)
                    image1_1_imag = np.stack((image1_imag[..., 0], image1_imag[..., 0], image1_imag[..., 0]), axis=2)
                    image1_2_real = np.stack((image2_real[..., 0], image2_real[..., 0], image2_real[..., 0]), axis=2)
                    image1_2_imag = np.stack((image2_imag[..., 0], image2_imag[..., 0], image2_imag[..., 0]), axis=2)
                    ang = np.stack((ang[..., 0], ang[..., 0], ang[..., 0]), axis=2)

                    images1_list_real.append(image1_1_real)
                    images1_list_imag.append(image1_1_imag)
                    images2_list_real.append(image1_2_real)
                    images2_list_imag.append(image1_2_imag)
                    ang_list.append(ang)

                    labels_list.append((label[..., 0]).astype(np.int64))

                    examples = examples + 1
                except Exception as e:
                    print(e)
        print("finished loading images")
        self.examples = examples
        print("Number of examples found:", examples)
        self.images1_real = np.array(images1_list_real)
        self.images1_imag = np.array(images1_list_imag)
        self.images2_real = np.array(images2_list_real)
        self.images2_imag = np.array(images2_list_imag)
        self.ang = np.array(ang_list)
        self.labels = np.array(labels_list)

    def next_batch(self, batch_size):

        if len(self.source_list) < batch_size:
            new_source = list(range(self.examples))
            random.shuffle(new_source)
            self.source_list.extend(new_source)

        examples_idx = self.source_list[:batch_size]
        del self.source_list[:batch_size]

        return self.images1_real[examples_idx, ...], self.images1_imag[examples_idx, ...], self.images2_real[examples_idx, ...], self.images2_imag[examples_idx, ...], self.ang[examples_idx, ...], self.labels[examples_idx, ...]
