import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. Prepare the data ---

data = [
    ("MobileNetV2",             1.9251752448e10),
    ("RetinaSys" ,              2.2686770432e10),
    ("EfficientNetB0",          2.5250817155e10),
    ("NASNetMobile",            3.6386375168e10),
    ("MobileNet",               3.6401180160e10),
    ("EfficientNetB1",          4.4863292419e10),
    ("EfficientNetB2",          6.4744811203e10),
    ("EfficientNetB3",          1.18910253571e11),
    ("DenseNet121",             1.81468425728e11),
    ("DenseNet169",             2.15113858560e11),
    ("ResNet50V2",              2.23153548800e11),
    ("ResNet50",                2.47310282240e11),
    ("DenseNet201",             2.74733309440e11),
    ("EfficientNetB4",          2.85362507779e11),
    ("InceptionV3",             3.66430995968e11),
    ("ResNet101V2",             4.60844887552e11),
    ("ResNet101",               4.85056212480e11),
    ("Xception",                5.35288550144e11),
    ("EfficientNetB5",          6.65398641155e11),
    ("ResNet152V2",             6.98916857600e11),
    ("ResNet152",               7.22840677888e11),
    ("InceptionResNetV2",       8.42993092096e11),
    ("VGG-16",                  9.90726778368e11),
    ("EfficientNetB6",          1.234433491133e12),
    ("VGG-19",                  1.257123606176e12),
    ("U-Net (DeepLabV3 ResNet50)", 1.6e12),
    ("Swin Transformer (Base-224)", 1.54e13),
    ("Vision Transformer (ViT-B/16)", 1.7e13),
    # NASNetLarge is omitted here since it has '-'
]

# Convert to a DataFrame and drop any None or invalid entries
df = pd.DataFrame(data, columns=["Model", "FLOPs"]).dropna()

# Sort by FLOPs in ascending order so smaller FLOPs are at the top
df.sort_values(by="FLOPs", inplace=True)
df.reset_index(drop=True, inplace=True)

# --- 2. Create a color palette that reflects FLOPs (blue → white → red) ---

# We'll use a 'coolwarm' colormap, which transitions from blue to red with a light center.
norm = plt.Normalize(df["FLOPs"].min(), df["FLOPs"].max())
cmap = plt.cm.coolwarm
colors = [cmap(norm(value)) for value in df["FLOPs"]]


sns.set_theme(style="whitegrid", font_scale=1.2)

fig, ax = plt.subplots(figsize=(10, 8))  # Adjust figure size for high-quality output

bars = ax.barh(df["Model"], df["FLOPs"], color=colors)

# Use a log scale on the x-axis to handle the large range of FLOPs
ax.set_xscale("log")

# Labeling
ax.set_xlabel("FLOPs (Log Scale)", labelpad=10)
ax.set_ylabel("Model", labelpad=10)
ax.set_title("FLOPs of Deep Learning Models", pad=15)

# Invert y-axis so that the smallest FLOPs is at the top
ax.invert_yaxis()

# Create a colorbar to show how FLOPs map to colors
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for older Matplotlib versions
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("FLOPs Scale")

# Make layout tight so labels don’t get cut off
plt.tight_layout()

# Display the plot
plt.show()

# If you want to save this figure for your paper, uncomment the following line:
plt.savefig("plots/flops_models_plot.png", dpi=300, bbox_inches="tight")
