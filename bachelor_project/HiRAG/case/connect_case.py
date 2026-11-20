import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 示例数据 (Example data)
data = {
    "Category": [
        "Mix",
        "Mix",
        "Mix",
        "CS",
        "CS",
        "CS",
        "Legal",
        "Legal",
        "Legal",
        "Agriculure",
        "Agriculure",
        "Agriculure",
    ],
    "Method": [
        "GraphRAG",
        "LightRAG",
        "HiRAG",
        "GraphRAG",
        "LightRAG",
        "HiRAG",
        "GraphRAG",
        "LightRAG",
        "HiRAG",
        "GraphRAG",
        "LightRAG",
        "HiRAG",
    ],
    "Value": [
        0.0091,
        0.0086,
        0.0692,
        0.0133,
        0.0181,
        0.0305,
        0.0185,
        0.0086,
        0.0236,
        0.0250,
        0.0173,
        0.0350,
    ],
}

# 将数据转换为DataFrame (Convert the data to a DataFrame)
df = pd.DataFrame(data)

# 设置Seaborn样式 (Set the Seaborn style)
sns.set(style="whitegrid")

# 创建图形 (Create the figure)
plt.figure(figsize=(6, 3))

# 绘制柱状图 (Plot the bar chart)
custom_palette = ["#3C3B8B", "#917BBD", "#E5CBE1"]  # 蓝色和橙色 (blue and orange)
sns.barplot(x="Category", y="Value", hue="Method", data=df, palette=custom_palette)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 设置标题和标签 (Set the title and labels)
plt.title("", fontsize=15)
plt.xlabel("Dataset", fontsize=15)
plt.ylabel("Clustering Coefficient", fontsize=15)
plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.18)
plt.legend(title="")

# 显示图形 (Display the figure)
plt.savefig("./connect.png")
