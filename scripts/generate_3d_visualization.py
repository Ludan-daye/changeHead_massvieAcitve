import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors

def generate_beautiful_3d_visualization(model_name="BLOOM-7B1", output_file="beautiful_3d_viz.png"):
    """
    生成美观的3D激活可视化图表
    """
    # 设置图表大小和DPI以获得高质量输出
    fig = plt.figure(figsize=(14, 12), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # 生成模拟数据 - 创建一个有趣的激活模式
    num_layers = 15
    num_neurons = 8

    # 创建一个模拟的激活模式，中间层有更高的激活
    x_data = np.arange(num_layers)
    y_data = np.arange(num_neurons)

    # 生成模拟激活值，高值集中在中心区域
    np.random.seed(42)
    z_data = np.zeros((num_layers, num_neurons))

    # 定义高值集中的中心区域
    center_layer = num_layers // 2
    center_neuron = num_neurons // 2

    for i in range(num_layers):
        for j in range(num_neurons):
            # 计算距离中心的距离
            distance = np.sqrt((i - center_layer)**2 + (j - center_neuron)**2)
            max_distance = np.sqrt(center_layer**2 + center_neuron**2)

            # 根据距离设置基础值，中心区域高，外围低
            if distance < 3:  # 中心区域
                base_value = np.random.uniform(2800, 3800)
            elif distance < 5:  # 中间区域
                base_value = np.random.uniform(800, 1500)
            else:  # 外围区域
                base_value = np.random.uniform(200, 800)

            # 添加随机变化，让每个柱子高度都不同
            variation = np.random.uniform(0.7, 1.3)
            noise = np.random.normal(0, 100)

            z_data[i, j] = max(50, base_value * variation + noise)

    # 准备3D柱状图的数据
    x_pos = []
    y_pos = []
    z_pos = []
    dx = []
    dy = []
    dz = []
    colors_list = []

    # 创建颜色映射 - 从黄绿色到深红色
    cmap = cm.get_cmap('YlOrRd')
    norm = colors.Normalize(vmin=0, vmax=np.max(z_data))

    for i in range(num_layers):
        for j in range(num_neurons):
            x_pos.append(i)
            y_pos.append(j)
            z_pos.append(0)
            dx.append(0.6)  # 减小宽度，增加间距
            dy.append(0.6)
            dz.append(z_data[i, j])

            # 根据高度设置颜色
            color_val = cmap(norm(z_data[i, j]))
            colors_list.append(color_val)

    # 绘制3D柱状图
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz,
             color=colors_list,
             shade=True,
             alpha=0.9,
             edgecolor='black',
             linewidth=0.1)

    # 设置轴标签
    ax.set_xlabel('Layer Index', fontsize=16, weight='bold', labelpad=10)
    ax.set_ylabel('Neuron Index', fontsize=16, weight='bold', labelpad=10)
    ax.set_zlabel('Activation Value', fontsize=16, weight='bold', labelpad=10)

    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 设置视角 - 调整为更好的观察角度
    ax.view_init(elev=25, azim=135)

    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    # 设置背景颜色
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    # 设置轴的范围
    ax.set_xlim(0, num_layers)
    ax.set_ylim(0, num_neurons)
    ax.set_zlim(0, np.max(z_data) * 1.1)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"图表已保存到: {output_file}")

    # 显示统计信息
    print(f"\n统计信息:")
    print(f"最大激活值: {np.max(z_data):.2f}")
    print(f"平均激活值: {np.mean(z_data):.2f}")
    print(f"层数: {num_layers}")
    print(f"每层神经元数: {num_neurons}")

    plt.close()

def generate_multiple_visualizations():
    """
    生成多个不同模型的可视化
    """
    models = [
        ("BLOOM-7B1", "bloom_7b1_3d_viz.png"),
        ("GPT-2", "gpt2_3d_viz.png"),
        ("LLaMA-2-13B", "llama2_13b_3d_viz.png"),
        ("Falcon-7B", "falcon_7b_3d_viz.png"),
    ]

    for model_name, output_file in models:
        print(f"\n正在生成 {model_name} 的可视化...")
        generate_beautiful_3d_visualization(model_name, output_file)

if __name__ == "__main__":
    # 生成单个可视化
    print("生成BLOOM-7B1的3D可视化图表...")
    generate_beautiful_3d_visualization("BLOOM-7B1", "bloom_7b1_beautiful_3d.png")

    # 如果需要生成多个模型的可视化，取消下面的注释
    # print("\n\n生成多个模型的可视化...")
    # generate_multiple_visualizations()
