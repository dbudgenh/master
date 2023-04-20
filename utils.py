from skimage import io
from matplotlib import pyplot as plt

def show_image(image_array,title) -> None:
    """Show an image in matplotlib

    Args:
        image (np_array): numpy array of the iamge
    """
    plt.title(title)
    io.imshow(image_array)
    plt.show()

