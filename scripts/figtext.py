"""Save figures as PDFs whose text is part of the graphic.

The figure is rendered once at 600 dpi and embedded losslessly in the PDF.
Labels look the same in print, but the PDF carries no text layer. Rotated
axis labels, mathematical glyphs from private Unicode areas and white labels
on dark backgrounds otherwise reach text extractors as scattered characters.
Rendering the whole figure keeps every label exactly where Matplotlib put it.
"""
import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def save_flat_pdf(fig, path, dpi=600):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = plt.imread(buf)
    h, w = img.shape[:2]
    f2 = Figure(figsize=(w / dpi, h / dpi))
    FigureCanvasAgg(f2)
    ax = f2.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="none", aspect="auto")
    ax.set_axis_off()
    f2.savefig(path, dpi=dpi, metadata={"CreationDate": None, "ModDate": None})
