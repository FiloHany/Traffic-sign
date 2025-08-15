import matplotlib.pyplot as plt
from .datetime_utils import DateTimeFormatter

class PerformancePlotter:
    """Handles plotting of model performance metrics."""

    @staticmethod
    def plot(history, figure_directory=None, ylim_pad=(0, 0)):
        xlabel = 'Epoch'
        legends = ['Training', 'Validation']

        plt.figure(figsize=(20, 5))

        # Accuracy Plot
        y1, y2 = history.history['accuracy'], history.history['val_accuracy']
        PerformancePlotter._plot_subplot(121, y1, y2, 'Model Accuracy', xlabel, 'Accuracy', legends, ylim_pad[0])

        # Loss Plot
        y1, y2 = history.history['loss'], history.history['val_loss']
        PerformancePlotter._plot_subplot(122, y1, y2, 'Model Loss', xlabel, 'Loss', legends, ylim_pad[1])

        if figure_directory:
            plt.savefig(f"{figure_directory}/history")

        plt.show()

    @staticmethod
    def _plot_subplot(position, y1, y2, title, xlabel, ylabel, legends, pad):
        min_y, max_y = min(min(y1), min(y2)) - pad, max(max(y1), max(y2)) + pad
        plt.subplot(position)
        plt.plot(y1)
        plt.plot(y2)
        plt.title(f"{title}\n{DateTimeFormatter.format(1)}", fontsize=17)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.ylim(min_y, max_y)
        plt.legend(legends, loc='upper left')
        plt.grid()