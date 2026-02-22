import matplotlib.pyplot as plt
import io
from PIL import Image

def plot_spectrogram_to_numpy(ground_truth, prediction):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    im1 = ax[0].imshow(ground_truth, aspect="auto", origin="lower", interpolation='none')
    ax[0].set_title("Ground Truth Mel")
    fig.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(prediction, aspect="auto", origin="lower", interpolation='none')
    ax[1].set_title("Predicted Mel")
    fig.colorbar(im2, ax=ax[1])

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf)
    return image
