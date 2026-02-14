260213：跑通了代码，读明白了所有代码，并分析了文件与文件的联系与作用
260214：
- 增加了画图模块在./trainer/animator.py中，并将其在base.py中调用，实现了在训练过程中实时观测精度与loss的变化。
- 增加了推理模块，编写在./inference/inference.py中，并在main函数将其调用。在训练完成时，会自动从从验证集中抽取一个batch进行推理。除此之外，还可以利用在checkpoints中保存的最优模型数据来导入数据集进行推理，本地的测试集保存在./test_images中

可以调用以下命令来使用保存的最优模型进行推理：
```powershell
python main.py --predict test_images/your_image.png --checkpoint checkpoints/your_model/best.pth --config configs/your_model.yaml
```
本地有提供0，3，7，8的图片进行推理。
但是推理效果并不好。