from skimage import io
from matplotlib import pyplot as plt
import numpy as np

def show_image_from_array(image_array,title='',show=True) -> None:
    """Show an image in matplotlib

    Args:
        image (np_array): numpy array of the iamge
    """


    #plt.figure(figsize=(16, 12))  # You can adjust the figsize as needed
    fig = plt.figure()
    plt.imshow(image_array)  # Display the image
    plt.title(title, fontsize=18)  # Set the title with larger font size
    plt.axis('off')  # Optional: Turn off axis ticks for better presentation
    if show:
        plt.show()
    return fig

def show_image_from_tensor(tensor,title='',show=True) -> None:
    return show_image_from_array(np.array(tensor.permute(1,2,0)),title,show=show)

def show_image_from_path(path,title) -> None:
    image_array = io.imread(path)
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

def create_multiple_images(images,titles):
    fig, axes = plt.subplots(1, len(images))
    for i in range(len(images)):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        axes[i].set_title(titles[i])
    plt.subplots_adjust(wspace=0.05)
    result = plt.gcf()
    plt.close()
    return result