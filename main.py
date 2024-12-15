import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import numpy as np

data = fetch_openml(name='wine', version=1, as_frame=True)

features = list(df.columns)
print("Available features:", features)
selected_features = [features[0], features[2], features[4], features[6], features[7]]
print("Selected features: ", selected_features)

fig, axs  = plt.subplots(1, len(selected_features), figsize = (20,3))

for ax, f in zip(axs, selected_features):
    ax.hist(df[f], bins=5, color='skyblue', edgecolor='black')
    ax.set_xlabel(f)

reference_feature = selected_features[1]
y = df[reference_feature]

fig, axs  = plt.subplots(1, len(selected_features), figsize = (20,3))

for ax, f in zip(axs, features):
  ax.scatter(df[f], y)
  ax.set_xlabel(f)

plt.show()

# Select the features for plotting
reference_feature = selected_features[1]  # Reference feature
comparison_feature = selected_features[2]  # Comparison feature

# Set up the figure with better styling
plt.style.use('ggplot')  # Apply a clean, modern style


# Create the scatter plot
plt.figure(figsize=(10, 7))
plt.scatter(df[reference_feature], df[comparison_feature], 
            alpha=0.7, color='teal', s=100, edgecolors='black')

# Get the x and y values for the trend line
x = df[reference_feature]
y = df[comparison_feature]

# חישוב קו המגמה (שימוש ב-polyfit למציאת המקדמים)
coefficients = np.polyfit(x, y, 2)  # Fit a linear trend line (degree 1)
trend_line = np.poly1d(coefficients)  # Create the trend line equation
plt.plot(x, trend_line(x), color='gray', linewidth=1.5, label=f'Trend Line: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')

# Customize the plot with meaningful labels and title
plt.title(f'Correlation between {reference_feature} & {comparison_feature}', 
          fontsize=16, weight='bold')
plt.xlabel(reference_feature, fontsize=14, weight='bold')
plt.ylabel(comparison_feature, fontsize=14, weight='bold')

# Add grid and improve readability
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot to an image file
plt.savefig('correlation_plot_improved.png', bbox_inches='tight')

# Display the plot
plt.show()
