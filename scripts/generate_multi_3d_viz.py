import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_sparse_data(num_layers, num_neurons, seed, num_peaks=2):
    """
    生成稀疏激活数据 - 少数几个高值，其他都是低值
    """
    np.random.seed(seed)
    z_data = np.zeros((num_layers, num_neurons))

    # 设置低基础值，错落有致
    for i in range(num_layers):
        for j in range(num_neurons):
            # 增加低值范围，让柱子有一定高度
            base_low = np.random.uniform(100, 400)
            # 添加变化让高度错落有致
            variation = np.random.uniform(0.7, 1.3)
            z_data[i, j] = base_low * variation

    # 随机设置少数几个高峰（更稀疏）
    for _ in range(num_peaks):
        i = np.random.randint(0, num_layers)
        j = np.random.randint(0, num_neurons)
        z_data[i, j] = np.random.uniform(1500, 2500)

    return z_data

def create_3d_subplot(ax, z_data, color='#1f77b4', label_text=None):
    """
    在给定的子图上创建3D柱状图
    """
    num_layers, num_neurons = z_data.shape

    x_pos = []
    y_pos = []
    z_pos = []
    dx = []
    dy = []
    dz = []
    colors_list = []

    for i in range(num_layers):
        for j in range(num_neurons):
            x_pos.append(i)
            y_pos.append(j)
            z_pos.append(0)
            dx.append(0.6)
            dy.append(0.6)
            dz.append(z_data[i, j])
            colors_list.append(color)

    # 绘制3D柱状图
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz,
             color=colors_list,
             shade=True,
             alpha=0.8,
             edgecolor='black',
             linewidth=0.3)

    # 设置视角
    ax.view_init(elev=20, azim=135)

    # 设置网格
    ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.5)

    # 设置背景
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_alpha(0.05)
    ax.yaxis.pane.set_alpha(0.05)
    ax.zaxis.pane.set_alpha(0.05)

    # 设置轴范围
    ax.set_xlim(0, num_layers)
    ax.set_ylim(0, num_neurons)
    ax.set_zlim(0, np.max(z_data) * 1.2 if np.max(z_data) > 0 else 100)

    # 简化刻度标签
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xticks([0, num_layers//2, num_layers])
    ax.set_yticks([0, num_neurons//2, num_neurons])

    # 移除轴标签（保持简洁）
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # 如果提供了标签文本，添加到图下方
    if label_text:
        ax.text2D(0.5, -0.1, label_text,
                  transform=ax.transAxes,
                  fontsize=10,
                  ha='center',
                  family='sans-serif')

def generate_multi_panel_visualization(num_panels=4, num_layers=12, num_neurons=10):
    """
    生成多个3D可视化面板
    """
    # 创建图形，横向排列
    fig = plt.figure(figsize=(16, 4), dpi=150)

    # 定义非常淡的天蓝色颜色方案
    colors = ['#C5E8F3', '#fdae6b', '#74c476', '#f07470']  # 非常淡的天蓝、浅橙、浅绿、浅红

    # 定义每个子图的标签（可选）
    labels = ['2533\n1415', '2533\n1415', '2533\n1415', '2533\n1415']

    for i in range(num_panels):
        ax = fig.add_subplot(1, num_panels, i+1, projection='3d')

        # 生成更稀疏的数据，减少峰值数量
        num_peaks = np.random.randint(1, 3)  # 1-2个高峰，更稀疏
        z_data = generate_sparse_data(num_layers, num_neurons, seed=42+i, num_peaks=num_peaks)

        # 使用天蓝色
        color = colors[0]  # 全部使用天蓝色，或者改为 colors[i % len(colors)] 使用不同颜色

        # 创建子图
        create_3d_subplot(ax, z_data, color=color, label_text=labels[i] if i < len(labels) else None)

    # 调整子图间距
    plt.tight_layout()

    # 保存图表
    output_file = 'multi_panel_3d_viz.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"多面板图表已保存到: {output_file}")

    plt.close()

def generate_single_style_visualization(color='#C5E8F3', label_text='2533\n1415'):
    """
    生成单个类似参考图的可视化
    """
    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # 生成更稀疏的激活数据
    num_layers = 12
    num_neurons = 10
    z_data = generate_sparse_data(num_layers, num_neurons, seed=42, num_peaks=2)

    # 创建图表
    create_3d_subplot(ax, z_data, color=color, label_text=label_text)

    plt.tight_layout()

    # 保存图表
    output_file = 'single_3d_viz.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"单个图表已保存到: {output_file}")

    plt.close()

if __name__ == "__main__":
    print("生成类似参考图的3D可视化...")

    # 生成4个面板的组合图
    print("\n1. 生成多面板组合图（4个子图）...")
    generate_multi_panel_visualization(num_panels=4, num_layers=16, num_neurons=12)

    # 生成单个图表
    print("\n2. 生成单个图表...")
    generate_single_style_visualization(color='#1f77b4', label_text='2533\n1415')

    print("\n完成！")
    print("提示：可以修改颜色参数来改变配色方案")
    print("例如：'#1f77b4'(蓝), '#ff7f0e'(橙), '#2ca02c'(绿), '#d62728'(红)")
