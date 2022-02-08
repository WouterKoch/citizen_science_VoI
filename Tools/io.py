import csv
import hashlib
import os

import numpy as np
from PIL import Image as pil_image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_img(path, grayscale=False, target_size=None, squaring_method="crop"):
    """Loads an image into PIL format.

      # Arguments
          path: Path to image file
          grayscale: Boolean, whether to load the image as grayscale.
          target_size: Either `None` (default to original size)
              or tuple of ints `(img_height, img_width)`.

      # Returns
          A PIL Image instance.

      # Raises
          ImportError: if PIL is not available.
      """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    try:
        img = pil_image.open(path)
    except IOError:
        print("IOError: Failed to read", path)
        return None
    except ZeroDivisionError:
        print("ZeroDivisionError", path)
        return None
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        try:
            img = square_pil_image(img, target_size, squaring_method)
        except OSError:
            print("OSError: Failed to read", path)
            return None
            # img.show()

    return img


def square_pil_image(img, target_size, squaring_method="crop"):
    """
    Squares a PIL image
    :param img:
    :param target_size:
    :param squaring_method: in ["crop", "pad", "random_crop"]
    :return: a squared and resized PIL image
    """
    hw_tuple = (target_size[1], target_size[0])
    if isinstance(squaring_method, (list, tuple)):
        squaring_method, squaring_parameter = squaring_method

    if img.size != hw_tuple:

        if squaring_method == "pad":
            # pad
            new_size = (max(img.size),) * 2
            new_im = pil_image.new("RGB", new_size)  # luckily, this is already black!
            new_im.paste(img, (int((new_size[0] - img.size[0]) / 2),
                               int((new_size[1] - img.size[1]) / 2)))

            if hw_tuple[0] > -1:
                img = new_im.resize(hw_tuple)
            else:
                img = new_im
        # img.show()
        elif squaring_method == "crop":
            min_size = min(img.size)
            img = img.crop(((img.size[0] - min_size) / 2, (img.size[1] - min_size) / 2, (img.size[0] + min_size) / 2,
                            (img.size[1] + min_size) / 2))
            img = img.resize(hw_tuple)
        elif squaring_method == "random_crop":
            min_size = min(img.size)
            x_start = (img.size[0] - min_size) / 2  # np.random.choice(range())
            y_start = (img.size[1] - min_size) / 2  # np.random.choice(range())

            if x_start > 0:
                x_start = np.random.choice(list(range(x_start)))

            if y_start > 0:
                y_start = np.random.choice(list(range(y_start)))

            img = img.crop((x_start, y_start, x_start + min_size, y_start + min_size))

            img = img.resize(hw_tuple)
        elif squaring_method == "center_crop":
            crop_size = max(img.size) * squaring_parameter
            x_start = (img.size[0] - crop_size) / 2
            y_start = (img.size[1] - crop_size) / 2

            img = img.crop((x_start, y_start, x_start + crop_size, y_start + crop_size))

            img = img.resize(hw_tuple)

            # img.save("/tmp/center_crop.jpg")


        elif squaring_method == "none":
            pass
    return img


def create_directory(*path):
    if len(path) > 1:
        path = os.path.join(path[0], *path[1:])
    else:
        path = path[0]
    if not os.path.isdir(path):
        os.makedirs(path)

    return path



def change_path_end(path, postfix, extension):
    if "." not in extension:
        extension = "." + extension
    return os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + postfix + extension)


def write_table(filename, table, headers=None):
    f = open(filename, "w", encoding="utf-8")
    writer = csv.writer(f)
    if headers is not None:
        writer.writerow(headers)
    for row in table:
        writer.writerow(row if not isinstance(row, dict) else [row[h] for h in headers])


def read_table(filename, has_headers=True, return_as_dict=False, index_by=None,
               check_for_split_format: bool = True):
    """

    :param filename:
    :param has_headers:
    :param return_as_dict:
    :param index_by:
    :param check_for_split_format: if True checks for filenames with format basename.{split_index}.ext, where basename and ext from
    filename
    :return:
    """

    if not os.path.exists(filename) and check_for_split_format:
        filename = split_template(filename)
    is_split_template = "{split_index}" in filename

    if not is_split_template:
        table, headers = read_table_basic(filename, has_headers, return_as_dict, index_by)
    else:
        split_index = 0
        table = []
        headers = None
        while os.path.exists(filename.format(split_index=split_index)):
            table_, headers = read_table_basic(filename.format(split_index=split_index), has_headers, return_as_dict, index_by)
            print("|table|", len(table_))
            table.extend(table_)
            split_index += 1
    return table, headers


def read_table_basic(filename, has_headers=True, return_as_dict=False, index_by=None):
    f = open(filename, "r", encoding="utf-8")
    reader = csv.reader(f)
    if index_by is None:
        table = []
    else:
        table = {}
    headers = None
    i = 0
    for row in reader:
        if i == 0 and has_headers:
            headers = row
        else:
            if has_headers and return_as_dict:
                dict_row = {}
                for i, header in enumerate(headers):
                    dict_row[header] = row[i]
                row = dict_row

            if index_by is None:
                table.append(row)
            else:
                table[row[index_by]] = row

        i += 1
    return table, headers


def encode_list(l):
    return [item.encode() for item in l]


def append_to_table(filename, table):
    f = open(filename, "a")
    writer = csv.writer(f)
    for row in table:
        writer.writerow(row)


def file_hash(path):
    """
    Computes a hash from a file.
    :param path:
    :return:
    """
    block_size = 65536
    hasher = hashlib.sha224()
    with open(path, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()


def split_template(path) -> str:
    """
    Returns the splitted file template for a path consisting of basename.{split_index}.ext
    """
    basename, extension = os.path.splitext(path)
    return basename + ".{split_index}" + extension


def numpy_load(path, key, check_for_split_format: bool = True) -> np.ndarray:
    """
    Load implementation of numpy.load that supports split files
    """

    if not os.path.exists(path) and check_for_split_format:
        path = split_template(path)
    is_split_template = "{split_index}" in path

    if not is_split_template:
        print(np.load(path).keys())
        array = np.load(path)[key]
    else:
        split_index = 0
        array = []
        while os.path.exists(path.format(split_index=split_index)):
            array_ = np.load(path.format(split_index=split_index))[key]
            print(array_.shape)
            array.append(array_)
            split_index += 1

        array = np.vstack(array)
        print(array.shape)
    return array
