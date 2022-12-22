import json
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import PIL

from tensorflow.keras import Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img


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


def predict_test_set(model: Model, image_size: tuple[int], labels: list[str], path: str = 'data/test.csv', evaluate: bool = False) -> pd.DataFrame:
    """Use model to predict on test data set.
    
    :param keras.Model model: A trained keras model object.
    :param tuple[int] image_size: Tuple with height and width of images, the data was trained on.
    :param list[str] labels: List with labels in the order, the model was trained on.
    :param str path: Path to the DataFrame with the test data.
    :param bool evaluate: (optional) If True model will only predict on data, for which a label exists in the test data. 
                          Default is False and prediction will be performed on the full test dataset. 

    Returns a DataFrame with the prediction for each test image.
    """

    # load csv with test data
    df_test = pd.read_csv(path, dtype={'Id': str})
    df_test['filename'] = 'data/images/' + df_test['Id'] + '.jpg'

    # load only labeled test data in evaluation mode
    # but for submission all test observations need to be predicted
    if evaluate and 'label' in df_test.columns:
        df_test = df_test[~df_test['label'].isna()]
    
    # create data generator
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_dataframe(
        df_test,
        x_col='filename',
        class_mode='input',
        target_size=image_size,
        batch_size=32,
        shuffle=False
    )

    # predict
    y_pred = model.predict(test_generator)

    classes = np.array(labels)
    predictions = classes[y_pred.argmax(axis=1)]

    df_test['pred'] = predictions

    if evaluate:
        df_test['correct'] = df_test['pred'] == df_test['label']
        print(f"Prediction accuracy: {df_test['correct'].mean()*100:.2f}%.")

    return df_test


def print_false_preds(df_test: pd.DataFrame) -> None:
    """Plots the first twelve falsely predicted images with predicted and actual label.
    
    :param pd.DataFrame df_test: A DataFrame with predictions.
    """

    plt.figure(figsize=(10,7))

    for idx, row in df_test[~df_test['correct']].sample(frac=1).reset_index(drop=True)[:12].iterrows():
        im = PIL.Image.open(row['filename'])
        plt.subplot(3, 4, idx+1)
        plt.imshow(im)
        plt.axis('off')
        plt.title(f"{row['Id']} - {row['pred']}/{row['label']}")
    plt.show()


def prepare_kaggle_submission(df_test: pd.DataFrame) -> None:
    """Saves a submission.csv file for Kaggle from the Dataframe with predictions.
    
    :param pd.DataFrame df_test: A DataFrame with predictions.
    """

    if df_test.shape[0] != 3808:
        print(f"Warning: Dataset with 3808 rows expected, but got {df_test.shape[0]} rows instead.")

    df_submission = df_test.drop(['label'], axis='columns').rename(columns={'pred': 'label'})
    df_submission[['Id', 'label']].to_csv('submission.csv', index=False)


def track_experiment(keras_history: History, architecture: str, version: int, hyper_params: dict, labels: list[str], start_epoch: int=1) -> pd.DataFrame:
    """Saves training performance in a central csv file.
    
    :param keras.callbacks.History keras_history: A keras history object.
    :param str architecture: Name of the used model architecture for labeling the training.
    :param int version: A number for the training.
    :param dict hyper_params: Used hyper parameters for the training.
    :param list[str] labels: List with labels in the order, the model was trained for.
    :param int start_epoch: Epoch, when the history was started. Default is 1, could be changed when training was resumed.

    Returns a DataFrame with all input data.
    """

    df_history = pd.DataFrame(keras_history.history)
    df_history['architecture'] = architecture
    df_history['epoch'] = list(range(start_epoch, len(keras_history.history['loss']) + start_epoch))
    df_history['version'] = version
    df_history['params'] = json.dumps(hyper_params)
    df_history['labels'] = json.dumps(labels)
    df_history['timestamp'] = datetime.now()

    save_history(df_history)

    return df_history


def save_history(df: pd.DataFrame, path: str='models/train_history.csv') -> None:
    """Appends training history to a central csv file.
    
    :param pd.DataFrame df: A DataFrame, which will be appended to the central csv file.
    :param str path: Location of the central csv file. (optional)
    """

    with open(path, mode='a', newline='') as f:
        df.to_csv(f, mode='a', header=f.tell()==0, index=False)


def load_history(path: str='models/train_history.csv') -> pd.DataFrame:
    """Loads all tracked training histories from a central csv file.
    
    :param str path: Location of the central csv file. (optional)

    Returns a DataFrame with all training histories which were tracked.
    """

    return pd.read_csv(path)


def load_history_version(version: int) -> pd.DataFrame:
    """Loads the tracked training history for a specific training session from a central csv file.
    
    :param int version: Number of the training session.

    Returns a DataFrame with the full training history for the specified session.
    """

    df = load_history()

    # filter for specific version in history
    df = df[df['version'] == version]

    return df


def get_params_for_version(version: int) -> dict:
    """Gets the used training parameters for a specific training session.
    
    :param int version: Number of the training session.

    Returns a dictionary with the used training parameters for the specified session.
    """

    df = load_history_version(version=version)
    params = df['params'].iloc[0]
    return json.loads(params)


def get_labels_for_version(version: int) -> list[str]:
    """Gets the used labels for a specific training session.
    
    :param int version: Number of the training session.

    Returns a list with the used labels for the specified training session.
    """

    df = load_history_version(version=version)
    labels = df['labels'].iloc[0]
    return json.loads(labels)


def get_latest_version() -> int:
    """Returns highest version number from the central csv file with tracked training histories.

    Returns a integer with the number. Returns  if none was found.0
    """

    try: 
        return load_history()['version'].max()

    except Exception as err:
        return 0


def plot_accuracy_for_training_history(df: pd.DataFrame) -> None:
    """Plots training and validation accuracy per epoch.
    
    :param pd.DataFrame df: History of the training cycle.
    """

    #df = df.reset_index()

    fig = plt.figure(figsize=(10, 3))
    ax = plt.axes()

    plt.plot(df['epoch'], df['accuracy'], label='training')
    plt.plot(df['epoch'], df['val_accuracy'], label='validation')

    plt.xlabel('epochs')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    plt.ylabel('accuracy')

    plt.title(f"Training: {df['version'].iloc[0]:d} - Architecture: {df['architecture'].iloc[0]:s}")

    plt.legend()
    
    plt.show()


def plot_accuracy_for_version(version: int) -> None:
    """Plots training and validation accuracy per epoch.
    
    :param int version: Version number of the training cycle.
    """

    df = load_history_version(version=version)

    plot_accuracy_for_training_history(df)


def plot_multiple_versions(versions: list[int], titles: list[str] = None, y_lim: tuple[float] = None) -> None:
    """Plots training and validation accuracy for multiple versions.
    
    :param list[int] versions: List of version numbers for training cycles.
    """

    df = load_history()

    fig, ax = plt.subplots(1, len(versions), sharex='col', sharey='row', figsize=(10, 5))

    if titles is None or len(versions) != len(titles):
        titles = [None] * len(versions)

    for idx, (version, title) in enumerate(zip(versions, titles)):

        df_v = df[df['version'] == version]

        ax[idx].plot(df_v['epoch'], df_v['accuracy'], label=('training'))
        ax[idx].plot(df_v['epoch'], df_v['val_accuracy'], label=('validation'))
        ax[idx].set_title(f"Training: {version:d} {chr(10) + str(title) if title is not None else ''}")

        if y_lim is not None:
            ax[idx].set_ylim(y_lim)

    plt.legend()
    plt.tight_layout()
    plt.show()