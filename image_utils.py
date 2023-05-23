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
    print(tensor)
    plt.title(title)
    io.imshow(np.array(tensor.permute(1,2,0)))
    plt.show()

def show_image_from_path(path,title) -> None:
    image_array = io.imread(path)
    print(image_array)
    plt.title(title)
    io.imshow(image_array)
    plt.show()

def array_from_image_path(path):
    return io.imread(path)/255.0

def show_multiple_images(images,titles):
    fig, axes = plt.subplots(1, len(images))
    for i in range(len(images)):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        axes[i].set_title(titles[i])
    plt.subplots_adjust(wspace=0.05)
    plt.show()