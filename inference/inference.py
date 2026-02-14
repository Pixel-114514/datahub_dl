import torch
import matplotlib.pyplot as plt
from data.dataloader import load_image

def get_mnist_labels(labels):
    """返回MNIST数据集的文本标签"""
    text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows * num_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy().squeeze(), cmap='gray')
        else:
            ax.imshow(img, cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()

def predict_batch(net, test_loader, device, n=6):
    """从验证集中随机抽取一个batch进行预测并展示前n张"""
    net.eval()
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        break
    
    with torch.no_grad():
        y_hat = net(X)
        preds = get_mnist_labels(y_hat.argmax(axis=1))
        trues = get_mnist_labels(y)
    
    titles = [f"True: {t}\nPred: {p}" for t, p in zip(trues, preds)]
    show_images(X[0:n].cpu().reshape((n, 28, 28)), 1, n, titles=titles[0:n])

def predict_single_image(net, img_path, device):
    """加载本地单张图片进行预测"""
    try:
        # 使用 dataloader.py 中封装好的加载函数
        X = load_image(img_path).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    net.eval()
    with torch.no_grad():
        y_hat = net(X)
        pred_label = get_mnist_labels(y_hat.argmax(axis=1))[0]
    
    print(f'Prediction for {img_path}: {pred_label}')
    show_images(X.cpu().reshape((1, 28, 28)), 1, 1, titles=[f'Predict: {pred_label}'])
