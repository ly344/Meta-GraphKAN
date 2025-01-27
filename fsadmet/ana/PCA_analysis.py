import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
class TSNEModule:
    def __init__(self, output_dir='/root/codes/MolFeSCue-master-2/tsne_visualizations', perplexity=30, n_iter=1000, random_state=42):
        """
        初始化 t-SNE 模块。
        
        :param output_dir: 保存 t-SNE 可视化结果的目录，默认为当前目录。
        :param perplexity: t-SNE 的困惑度参数，默认为 30。
        :param n_iter: t-SNE 迭代次数，默认为 1000。
        :param random_state: 随机状态种子，确保结果可复现，默认为 42。
        """
        self.output_dir = output_dir
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state

    def visualize(self, features, labels, epoch, title=None, filename=None, show_plot=True):
        """
        对特征进行 t-SNE 降维并绘制可视化图。
        
        :param features: 特征数据，形状为 (n_samples, n_features) 的 NumPy 数组。
        :param labels: 标签数据，长度为 n_samples 的 NumPy 数组或列表。
        :param epoch: 当前训练的 epoch 数。
        :param title: 图像标题，默认为 None。
        :param filename: 保存图像的文件名，默认为 None（不保存）。
        :param show_plot: 是否显示图像，默认为 True。
        """
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 使用 t-SNE 进行降维
        tsne = TSNE(n_components=2, perplexity=self.perplexity, n_iter=self.n_iter, random_state=self.random_state)
        features_2d = tsne.fit_transform(features)

        # 定义颜色映射
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colormap = plt.cm.get_cmap('tab10')  # 使用 tab10 颜色映射
        colors = [colormap(i / num_classes) for i in range(num_classes)]

        # 将标签映射到颜色
        label_to_color = {label: color for label, color in zip(unique_labels, colors)}
        point_colors = [label_to_color[label] for label in labels]

        # 绘制 t-SNE 结果
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=point_colors, cmap=None, s=15, alpha=0.7)  # 设置标记大小和透明度
        plt.title(title or f't-SNE visualization of the graph embeddings (Epoch {epoch})')
        plt.grid(True)  # 添加网格线

        # 创建图例
        legend_labels = {f'Class {label}': color for label, color in label_to_color.items()}
        handles = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='', markersize=10) for color in legend_labels.values()]
        plt.legend(handles, legend_labels.keys(), loc='upper right', fontsize=10)  # 右上角标签
            # 清除刻度标签
        plt.xticks([])  # 清除x轴刻度标签
        plt.yticks([])  # 清除y轴刻度标签
        if epoch:
            filepath = os.path.join(self.output_dir, str(epoch))
            plt.savefig(filepath)
            print(f'filepath: ', filepath)
            print(f"t-SNE visualization saved to {filepath}")

        if show_plot:
            plt.draw()
            plt.pause(3)  # 暂停一段时间以显示图像，然后继续执行
        
        plt.close()  # 关闭图形以释放内存

class PCAModule:
    def __init__(self, output_dir='/root/codes/MolFeSCue-master-2/pca_visualizations', n_components=2, random_state=42):
        """
        初始化 PCA 模块。
        
        :param output_dir: 保存 PCA 可视化结果的目录，默认为当前目录。
        :param n_components: PCA 的主成分数量，默认为 2。
        :param random_state: 随机状态种子，确保结果可复现，默认为 42。
        """
        self.output_dir = output_dir
        self.n_components = n_components
        self.random_state = random_state

    def visualize(self, features, labels, epoch, title=None, filename=None, show_plot=True):
        """
        对特征进行 PCA 降维并绘制可视化图。
        
        :param features: 特征数据，形状为 (n_samples, n_features) 的 NumPy 数组。
        :param labels: 标签数据，长度为 n_samples 的 NumPy 数组或列表。
        :param epoch: 当前训练的 epoch 数。
        :param title: 图像标题，默认为 None。
        :param filename: 保存图像的文件名，默认为 None（不保存）。
        :param show_plot: 是否显示图像，默认为 True。
        :param fixed_axis_limits: 固定的轴范围，形如 (xmin, xmax, ymin, ymax) 的元组，默认为 None（自动调整）。
        """
        # fixed_axis_limits = [-80,100,-80,70]
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 使用 PCA 进行降维
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        features_2d = pca.fit_transform(features)

        # 定义颜色映射
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        colormap = plt.cm.get_cmap('tab10')  # 使用 tab10 颜色映射
        colors = [colormap(i / num_classes) for i in range(num_classes)]

        # 将标签映射到颜色
        label_to_color = {label: color for label, color in zip(unique_labels, colors)}
        point_colors = [label_to_color[label] for label in labels]

        # 绘制 PCA 结果
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=point_colors, cmap=None, s=5, alpha=0.7)  # 设置标记大小和透明度
        plt.title(title or f'PCA visualization of the graph embeddings (Epoch {epoch})')
        plt.grid(True)  # 添加网格线

        # # 如果提供了固定的轴范围，则应用这些限制
        # if fixed_axis_limits is not None:
        #     xmin, xmax, ymin, ymax = fixed_axis_limits
        #     plt.xlim(xmin, xmax)
        #     plt.ylim(ymin, ymax)

        # 创建图例
        legend_labels = {f'Class {label}': color for label, color in label_to_color.items()}
        handles = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='', markersize=10) for color in legend_labels.values()]
        plt.legend(handles, legend_labels.keys(), loc='upper right', fontsize=10)  # 右上角标签

        if filename:
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, format='png')
            print(f"PCA visualization saved to {filepath}")

        if show_plot:
            plt.draw()
            plt.pause(2)  # 暂停一段时间以显示图像，然后继续执行
        
        plt.close()  # 关闭图形以释放内存
# 示例用法
if __name__ == "__main__":
    # 假设我们有一些特征和标签
    example_features = np.random.rand(100, 50)  # 100个样本，每个样本有50个特征
    example_labels = np.random.randint(0, 2, size=100)  # 二分类问题的标签

    # 创建 t-SNE 模块实例
    tsne_module = TSNEModule(output_dir='./tsne_visualizations', perplexity=30, n_iter=1000, random_state=42)

    # 调用 visualize 方法进行可视化
    tsne_module.visualize(example_features, example_labels, epoch=1, title="Example t-SNE Visualization", filename="example_tsne.png")