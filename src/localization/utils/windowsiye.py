#%%



import matplotlib.pyplot as plt


# Data: (version, MAE, window_size)
data = [
    ("version_1", 0.5851, 10),
    ("version_2", 0.4199, 20),
    ("version_10", 0.253, 30),
    ("version_4", 0.2625, 40),
    ("version_11", 0.273, 50),
    ("version_12", 0.286, 60),
    ("version_9", 0.298, 70),
    ("version_5", 0.3217, 80),
    ("version_6", 0.4873, 160),
    ("version_7", 1.1327, 320),

]

# Extract window size (x) and MAE (y)
window_sizes = [d[2] for d in data]
mae_values = [d[1] for d in data]

# Sort by window size for better visualization
sorted_data = sorted(zip(window_sizes, mae_values))
window_sizes, mae_values = zip(*sorted_data)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(window_sizes, mae_values, marker='o', linestyle='-', linewidth=2, markersize=8)

plt.xlabel("Window Size", fontsize=12)
plt.ylabel("MAE", fontsize=12)
plt.title("MAE vs Window Size", fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(10, 6))
plt.scatter(window_sizes, mae_values, s=100, c='blue', edgecolors='black')

for i, txt in enumerate(data):
    plt.annotate(txt, (window_sizes[i], mae_values[i]), 
                 textcoords="offset points", xytext=(5,5), fontsize=9)

plt.xlabel("Window Size", fontsize=12)
plt.ylabel("MAE", fontsize=12)
plt.title("MAE vs Window Size (Scatter)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
#%%
plt.figure(figsize=(12, 6))

plt.bar(window_sizes, mae_values, color='steelblue', edgecolor='black')

plt.xlabel("Window Size", fontsize=12)
plt.ylabel("MAE", fontsize=12)
plt.title("MAE by Window Size", fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
# Data
data = {
    'Version': ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v10', 'v11', 'v12', 'v9'],
    'Window Size': [10, 20, 40, 80, 160, 320, 30, 50, 60, 70],
    'MAE': [0.5851, 0.4199, 0.2625, 0.3217, 0.4873, 1.1327, 0.253, 0.273, 0.286, 0.298]
}
time=[0.619125/10 *i for i in data['Window Size']   ]
data['Time (s)'] = time

df = pd.DataFrame(data)
df = df.sort_values('Window Size')  # Sort by window size

# Print table
print(df.to_string(index=False))

# Display as table plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, 
                 cellLoc='center', loc='center',
                 colColours=['lightgreen']*4)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
plt.show()

#%%
2.4765/4

	
version_15
0,729
2
version_16
0,693
4
version_17
0,795
8
version_18
0,985
12