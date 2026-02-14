import os
# 防御性编程 避免在某些环境下报 KMP 重复库错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from typing import List, Optional
import matplotlib.pyplot as plt
from IPython import display

def use_svg_display() -> None:
    # 启用更清晰的SVG显示 并设置默认图像尺寸
    try:
        from IPython import display  # noqa
        plt.rcParams["figure.figsize"] = (3.5, 2.5)
    except Exception:
        pass

def set_axes(ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend: Optional[List[str]] = None) -> None:
    ax.set_xlabel(xlabel) if xlabel is not None else None
    ax.set_ylabel(ylabel) if ylabel is not None else None
    ax.set_xscale(xscale) if xscale is not None else None
    ax.set_yscale(yscale) if yscale is not None else None
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None
    if legend:
        ax.legend(legend)
    ax.grid(True)

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale="linear", yscale="linear",
                 fmts=("-", "m--", "g-.", "r:"), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x_, y_, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_, y_, fmt)
        self.config_axes()
        plt.pause(0.01)
