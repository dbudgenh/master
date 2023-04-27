from skimage import io
from matplotlib import pyplot as plt
import numpy as np

def show_image(image_array,title) -> None:
    """Show an image in matplotlib

    Args:
        image (np_array): numpy array of the iamge
    """
    plt.title(title)
    io.imshow(image_array)
    plt.show()

def show_tensor_as_image(tensor,title) -> None:
    plt.title(title)
    io.imshow(np.array(tensor.permute(1,2,0)))
    plt.show()
