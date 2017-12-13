import itertools
from os import makedirs
from os.path import expanduser, join

# Default folders
dir_root = join(expanduser('~'), 'imagenet')
dir_originals = join(dir_root, 'original')
dir_resized = join(dir_root, 'resized')
dir_tfrecord = join(dir_root, 'tfrecords')
dir_metrics = join(dir_root, 'metrics')
dir_checkpoints = join(dir_root, 'checkpoints')


def maybe_create_folder(folder):
    try:
        makedirs(folder)
    except:
        pass


def progressive_filename_generator(pattern='file_{}.ext'):
    for i in itertools.count():
        yield pattern.format(i)
