import matplotlib.pyplot as plt
import pandas as pd
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


def plot_thumbnails(image_paths: list[str], n_cols: int = 10) -> None:
    """
    Plots squared thumbnails in a grid.
    
    :param list[str] image_paths: Paths to the images. All images are plotted.
    :param int n_cols: Number of columns in preview (optional)
    """

    # prevent numbers smaller than one
    n_cols = max(1, n_cols)
    n: int = len(image_paths)
    n_rows = n // n_cols

    fig = plt.figure(figsize=(n_cols*2, n_rows*2))
    fig.patch.set_alpha(0.0)

    for i in range(n):
        im = PIL.Image.open(image_paths[i])
        #im = square_img(im)
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
    plt.show()


def plot_thumbnails_for_label(df: pd.DataFrame, label: str, n: int = 30, n_cols: int = 10) -> None:
    """
    Plots squared thumbnails in a grid.
    
    :param pd.DataFrame df: Dataframe with columns "label" and "filename"
    :param str label: Label, for which images should be plotted
    :param int n_cols: Number of images in preview (optional)
    :param int n_cols: Number of columns in preview (optional)
    """

    image_paths: list[str] = df.loc[df['label'] == label, 'filename'].iloc[:n].tolist()

    plot_thumbnails(
        image_paths=image_paths, 
        n_cols=n_cols
    )
    