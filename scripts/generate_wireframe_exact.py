import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 24,
    'figure.titlesize': 28
})
matplotlib.rcParams['text.usetex'] = False

def plot_3d_feat_sub(ax, tokens, zdata, title=None):
    """
    完全复刻原始仓库的3D绘图风格 - 垂直柱子
    """
    num_tokens = len(tokens)
    num_channels = zdata.shape[0]

    # 生成底部基线的微小凸起
    np.random.seed(hash(str(tokens)) % 2**32)
    base_bumps = np.random.uniform(20, 80, (num_channels, num_tokens))

    # 为每个通道绘制底部水平基线（带微小凸起）
    for i in range(num_channels):
        x = list(range(num_tokens))
        y = [i] * num_tokens
        z = base_bumps[i, :].tolist()
        ax.plot(x, y, z, color="royalblue", linewidth=2.5)

    # 为每个X位置绘制垂直连接线（连接两个通道的基线）
    for j in range(num_tokens):
        x = [j, j]
        y = [0, num_channels - 1]
        z = [base_bumps[0, j], base_bumps[1, j]]
        ax.plot(x, y, z, color="royalblue", linewidth=2.5)

    # 为每个位置绘制垂直线（柱子）
    for j in range(num_tokens):
        for i in range(num_channels):
            if zdata[i, j] > 0:  # 只绘制有数据的柱子
                x = [j, j]
                y = [i, i]
                z = [base_bumps[i, j], zdata[i, j]]
                ax.plot(x, y, z, color="royalblue", linewidth=2.5)

    # 设置X轴刻度（使用数字替代token标签）
    ax.set_xticks(np.linspace(0, num_tokens-1, num_tokens), [str(i+1) for i in range(num_tokens)], rotation=0, fontsize=16)

    # 设置Z轴刻度
    ax.set_zticks([0, 1000, 2000], ["0", "1k", "2k"], fontsize=15)

    # 设置Y轴刻度
    ax.set_yticks([0, 1], [1415, 2533], fontsize=15, fontweight="heavy")

    # 设置标题
    if title:
        ax.set_title(title, fontsize=18, fontweight="bold", y=1.015)

    # 旋转Y轴标签
    plt.setp(ax.get_yticklabels(), ha="left", rotation_mode="anchor")

    # 调整刻度间距
    ax.tick_params(axis='x', which='major', pad=-4)
    ax.tick_params(axis='y', which='major', pad=-5)
    ax.tick_params(axis='z', which='major', pad=-1)

    # 设置轴范围
    ax.set_xlim(-0.5, num_tokens-0.5)
    ax.set_ylim(-0.5, num_channels-0.5)
    ax.set_zlim(0, 2400)

def generate_sample_data(tokens, seed=42):
    """
    生成模拟的激活数据 - 2个通道，稀疏激活
    """
    np.random.seed(seed)
    num_tokens = len(tokens)
    num_channels = 2

    # 大部分值为0（只显示基线）
    zdata = np.zeros((num_channels, num_tokens))

    # 随机设置几个高峰
    for _ in range(3):
        channel = np.random.randint(0, num_channels)
        token_idx = np.random.randint(0, num_tokens)
        zdata[channel, token_idx] = np.random.uniform(800, 2200)

    return zdata

def generate_exact_replica():
    """
    生成和参考图一模一样的3面板图表
    """
    fig = plt.figure(figsize=(14, 6))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.13)

    # 定义3组不同的token序列
    token_sets = [
        ["Summer", "is", "warm", "Winter", "is", "cold", "In"],
        ["Summer", "is", "warm", "In", "Winter", "is", "cold", "in"],
        ["Why", "is", "summer", "warm", "and", "winter", "cold", "?"]
    ]

    # 为每个子图生成数据
    for i, tokens in enumerate(token_sets):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        # 生成稀疏激活数据
        zdata = generate_sample_data(tokens, seed=42+i*10)

        # 绘制子图（不显示子标题）
        plot_3d_feat_sub(ax, tokens, zdata, title=None)

    # 不添加总标题

    # 保存图表
    output_file = 'exact_replica_wireframe.png'
    plt.savefig(output_file, bbox_inches="tight", dpi=200, facecolor='white')
    print(f"完全复刻版图表已保存到: {output_file}")
    plt.close()

def generate_single_panel():
    """
    生成单个面板的图表
    """
    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    tokens = ["Summer", "is", "warm", "Winter", "is", "cold", "In"]
    zdata = generate_sample_data(tokens, seed=42)

    plot_3d_feat_sub(ax, tokens, zdata, title="LLaMA2-7B")

    output_file = 'single_wireframe.png'
    plt.savefig(output_file, bbox_inches="tight", dpi=200, facecolor='white')
    print(f"单面板图表已保存到: {output_file}")
    plt.close()

if __name__ == "__main__":
    print("生成完全复刻原始仓库风格的3D可视化...")

    # 生成3面板图
    print("\n1. 生成3面板组合图...")
    generate_exact_replica()

    # 生成单面板图
    print("\n2. 生成单面板图...")
    generate_single_panel()

    print("\n完成！")
