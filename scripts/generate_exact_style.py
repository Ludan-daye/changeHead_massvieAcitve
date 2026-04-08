import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_sparse_activation_data(num_layers, num_neurons, seed, num_peaks=3):
    """
    生成稀疏激活数据 - 完全模仿参考图的模式
    """
    np.random.seed(seed)
    z_data = np.zeros((num_layers, num_neurons))

    # 大部分值都非常接近0，几乎看不见
    for i in range(num_layers):
        for j in range(num_neurons):
            z_data[i, j] = np.random.uniform(0, 20)

    # 随机设置几个高峰
    for _ in range(num_peaks):
        i = np.random.randint(0, num_layers)
        j = np.random.randint(0, num_neurons)
        z_data[i, j] = np.random.uniform(1900, 2100)

    return z_data

def create_subplot_exact_style(ax, z_data, label_text=None):
    """
    创建和参考图完全一样风格的3D柱状图
    """
    num_layers, num_neurons = z_data.shape

    x_pos = []
    y_pos = []
    z_pos = []
    dx = []
    dy = []
    dz = []

    for i in range(num_layers):
        for j in range(num_neurons):
            x_pos.append(i)
            y_pos.append(j)
            z_pos.append(0)
            dx.append(0.85)
            dy.append(0.85)
            dz.append(z_data[i, j])

    # 使用标准蓝色
    color = '#1f77b4'

    # 绘制3D柱状图
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz,
             color=color,
             shade=True,
             alpha=0.95,
             edgecolor='#0d47a1',
             linewidth=0.15)

    # 设置视角 - 完全匹配参考图
    ax.view_init(elev=15, azim=125)

    # 设置网格
    ax.grid(True, linestyle='-', alpha=0.4, linewidth=0.5)

    # 设置背景
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_alpha(0.02)
    ax.yaxis.pane.set_alpha(0.02)
    ax.zaxis.pane.set_alpha(0.02)

    # 设置轴范围
    ax.set_xlim(0, num_layers)
    ax.set_ylim(0, num_neurons)
    max_z = np.max(z_data) if np.max(z_data) > 0 else 2000
    ax.set_zlim(0, max_z * 1.1)

    # 设置Z轴刻度为 0, 1k, 2k 格式
    z_max = max_z * 1.1
    ax.set_zticks([0, z_max/2, z_max])
    ax.set_zticklabels(['0', f'{int(z_max/2000)}k', f'{int(z_max/1000)}k'])

    # 简化X、Y轴刻度
    ax.set_xticks([0, num_layers//2, num_layers])
    ax.set_yticks([0, num_neurons//2, num_neurons])

    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=9)

    # 移除轴标签
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # 添加底部标签
    if label_text:
        ax.text2D(0.5, -0.05, label_text,
                  transform=ax.transAxes,
                  fontsize=11,
                  ha='center',
                  va='top',
                  family='serif')

def generate_four_panel_exact():
    """
    生成4个面板，完全模仿参考图
    """
    # 创建图形
    fig = plt.figure(figsize=(18, 4.5), dpi=150)

    # 定义标签
    labels = ['2533\n1415', '2533\n1415', '2533\n1415', '2533\n1415']

    # 参数设置 - 完全匹配参考图
    num_layers = 18
    num_neurons = 14

    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')

        # 生成数据，完全模仿参考图的峰值分布
        # 第1个图：多个峰值，第2个图：中等峰值，第3个图：少量峰值，第4个图：中等峰值
        num_peaks = [5, 4, 2, 4][i]
        z_data = generate_sparse_activation_data(num_layers, num_neurons, seed=100+i*10, num_peaks=num_peaks)

        # 创建子图
        create_subplot_exact_style(ax, z_data, label_text=labels[i])

    # 调整子图间距
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.15)

    # 保存图表
    output_file = 'exact_style_4panel.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"完全复刻版图表已保存到: {output_file}")

    plt.close()

if __name__ == "__main__":
    print("生成和参考图完全一样的4面板3D可视化...")
    generate_four_panel_exact()
    print("完成！")
