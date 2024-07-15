import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def read_img(path, greyscale=True):
    img = Image.open(path)
    if greyscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    return np.array(img).astype(np.float)


def save_img(img, path):
    img = img - img.min()
    img = img / img.max()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
    print(path, "is saved!")


def find_maxima(scale_space, k_xy=5, k_s=1):
    """
    Extract the peak x,y locations from scale space

    Input
      scale_space: Scale space of size HxWxS
      k: neighborhood in x and y
      ks: neighborhood in scale

    Output
      list of (x,y) tuples; x<W and y<H
    """
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None]

    H, W, S = scale_space.shape
    maxima = []
    for i in range(H):
        for j in range(W):
            for s in range(S):
                # extracts a local neighborhood of max size
                # (2k_xy+1, 2k_xy+1, 2k_s+1)
                neighbors = scale_space[max(0, i - k_xy):min(i + k_xy + 1, H),
                                        max(0, j - k_xy):min(j + k_xy + 1, W),
                                        max(0, s - k_s):min(s + k_s + 1, S)]
                mid_pixel = scale_space[i, j, s]
                num_neighbors = np.prod(neighbors.shape) - 1
                # if mid_pixel > all the neighbors; append maxima
                if np.sum(mid_pixel < neighbors) == num_neighbors:
                    maxima.append((i, j, s))
    return maxima


def visualize_scale_space(scale_space, min_sigma, k, file_path=None):
    """
    Visualizes the scale space

    Input
      scale_space: scale space of size HxWxS
      min_sigma: the minimum sigma used
      k: the sigma multiplier
    """
    if len(scale_space.shape) == 2:
        scale_space = scale_space[:, :, None]
    H, W, S = scale_space.shape

    # number of subplots
    p_h = int(np.floor(np.sqrt(S)))
    p_w = int(np.ceil(S / p_h))
    for i in range(S):
        plt.subplot(p_h, p_w, i + 1)
        plt.axis('off')
        plt.title('{:.1f}:{:.1f}'.format(min_sigma * k**i,
                                         min_sigma * k**(i + 1)))
        plt.imshow(scale_space[:, :, i])

    # plot or save to fig
    if file_path:
        plt.savefig(file_path, dpi=300)
    else:
        plt.show()


def visualize_maxima(image, maxima, min_sigma, k, file_path=None):
    """
    Visualizes the maxima on a given image

    Input
      image: image of size HxW
      maxima: list of (x,y) tuples; x<W, y<H
      file_path: path to save image. if None, display to screen
    Output-   None
    """
    H, W = image.shape
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for maximum in maxima:
        y, x, s = maximum
        assert x < W and y < H and x >= 0 and y >= 0
        radius = np.sqrt(2) * min_sigma * (k**s)
        # radius = 1
        circ = plt.Circle((x, y), radius, color='r', fill=False)
        ax.add_patch(circ)

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
