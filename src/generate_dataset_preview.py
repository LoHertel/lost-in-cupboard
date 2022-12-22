import matplotlib.pyplot as plt
import PIL


def square_img(im: PIL.Image) -> PIL.Image:
    """
    Square image by retaining shorter dimension and 
    crop on longer dimension centered.
    
    :param PIL.Image im: A image object
    """

    # get x and y dimensions of image size in pixels
    img_size: tuple[int, int] = im.size
    x, y = img_size

    # calculate offset for squared crop
    size: int = min(x, y)
    x_offset: int = round((x - size) / 2)
    y_offset: int = round((y - size) / 2)

    # crop and return image
    return im.crop(box=(x_offset, y_offset, x-x_offset, y-y_offset))


def generate_image(data_folder: str, image_folder: str, n_rows: int = 3, n_cols: int = 10) -> None:
    """
    Generates image with a number of squared thumbnails of the dataset.
    
    :param int n_rows: Number of rows in preview
    :param int n_cols: Number of columns in preview
    """

    # prevent numbers smaller than one
    n_rows = max(1, n_rows)
    n_cols = max(1, n_cols)
    n: int = n_rows * n_cols

    plt.figure(figsize=(n_cols, n_rows))

    for i in range(n):
        im = PIL.Image.open(f'{data_folder}/{i:04d}.jpg')
        im = square_img(im)
        ax = plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(im)
        ax.spines['bottom'].set_color('grey')
        ax.spines['top'].set_color('grey') 
        ax.spines['right'].set_color('grey')
        ax.spines['left'].set_color('grey')

    # remove the x and y ticks
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

    # set the spacing between subplots
    plt.subplots_adjust(
        wspace=0.1,
        hspace=0.1
    )

    # save image
    plt.savefig(f'{image_folder}/preview.png', bbox_inches='tight', transparent=True)


if __name__ == '__main__':

    data_folder = '../data/images'
    image_folder = '../images'

    generate_image(data_folder, image_folder)