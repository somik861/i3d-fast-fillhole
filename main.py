from skimage import io
from argparse import ArgumentParser
import numpy as np
from skimage.morphology import reconstruction, dilation, erosion, cube


def fillhole(img: np.ndarray) -> np.ndarray:
    img = dilation(img, footprint=cube(15))
    img = erosion(img, footprint=cube(15))

    marker = img.copy()
    marker[1:-1, 1:-1, 1:-1] = img.max()
    rv = reconstruction(marker, img,
                        footprint=np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]), method='erosion')

    
    return rv


def fillhole2(img: np.ndarray) -> np.ndarray:
    marker = img.copy()
    marker[1:-1, 1:-1, 1:-1] = img.min()

    return reconstruction(marker, img,
                          footprint=np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                              [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                                              [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]), method='dilation')


if __name__ == '__main__':
    parser = ArgumentParser('fillhole')
    parser.add_argument('--input', type=str, metavar='IMG', required=True)
    parser.add_argument('--output', type=str, metavar='IMG', required=True)

    args = parser.parse_args()

    img = io.imread(args.input)

    io.imsave(args.output, fillhole2(img),
              compression='zlib', check_contrast=False)
