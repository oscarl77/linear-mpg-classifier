import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(labels, predicted):
    cm = confusion_matrix(labels, predicted)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()