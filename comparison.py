import matplotlib.pyplot as plt
import numpy as np

# Initialize variables to store accuracy and AUROC values
accuracy_values = []
auroc_values = []
lines = ['Accuracy (Guessing the Petitioner Wins)', 'Accuracy (MLP)', 'Accuracy (LightGBM)',
         'Accuracy (XGBoost)', 'Accuracy (Random Forest)']
lines2 = ['AUROC (Guessing the Petitioner Wins)', 'AUROC (MLP)', 'AUROC (LightGBM)', 'AUROC (XGBoost)', 'AUROC (Random Forest)']

# Specify the file path
file_path = 'output.txt'

# Read accuracy and AUROC values
for entry in lines:
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(entry):
                accuracy_values.append(float(line.split(':')[1].strip()))
                break

# Read AUROC values
for entry in lines2:
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(entry):
                auroc_values.append(float(line.split(':')[1].strip()))
                break
for value in zip(auroc_values, accuracy_values):
    print(value)
    
# Create a NumPy array for the x-axis positions
x = np.arange(len(lines))

# Sort labels, accuracy, and AUROC values by accuracy
sorted_indices = np.argsort(accuracy_values)[::-1]
lines = [lines[i] for i in sorted_indices]
accuracy_values = [accuracy_values[i] for i in sorted_indices]
auroc_values = [auroc_values[i] for i in sorted_indices]

# Create a bar graph for accuracy
fig, ax1 = plt.subplots()
bars1 = ax1.bar(x - 0.2, accuracy_values, width=0.4, label='Accuracy', color='blue')

# Annotate each bar with its exact value
for bar, accuracy in zip(bars1, accuracy_values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{accuracy:.5f}',
             ha='center', va='bottom')

# Create a bar graph for AUROC
ax2 = ax1.twinx()
bars2 = ax2.bar(x + 0.2, auroc_values, width=0.4, label='AUROC', color='orange')

# Annotate each bar with its exact value
for bar, auroc in zip(bars2, auroc_values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{auroc:.5f}',
             ha='center', va='bottom')

# Add labels and title
ax1.set_xlabel('Models')
ax1.set_ylabel('Accuracy', color='blue')
ax2.set_ylabel('AUROC', color='orange')
ax1.set_title('Accuracy and AUROC for Each Model (Sorted by Accuracy)')

# Set x-axis ticks and labels
ax1.set_xticks(x)
ax1.set_xticklabels(lines)

# Show the plot
plt.show()
