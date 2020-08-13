import tensorflow as tf
import os
import shutil


class Logger(object):
    def __init__(self, log_dir, suffix=None):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir, filename_suffix=suffix)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)


class ModelLogger(object):
    def __init__(self, log_dir, save_func):
        self.log_dir = log_dir
        self.save_func = save_func

    def save(self, model, epoch, isGenerator):
        if isGenerator:
            new_path = os.path.join(self.log_dir, "model_%05d.pth" % epoch)
        else:
            new_path = os.path.join(self.log_dir, "disc_%05d.pth" % epoch)
        self.save_func(model, new_path)

    def copy_file(self, source):
        shutil.copy(source, self.log_dir)
