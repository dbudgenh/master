import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
CSV_FILE_PATH = './statistics/EfficientNet-V2_S/metrics.csv'

# Read the CSV file using pandas
df = pd.read_csv(CSV_FILE_PATH)

# Specify the column to remove
columns_to_remove = ['train_accuracy_step','train_loss_step','step', 'epoch','validation_accuracy','validation_loss','train_accuracy_epoch','train_loss_epoch']
columns_to_remove_accuracy = ['step','train_loss_step','validation_loss','train_loss_epoch']
columns_to_remove_loss = ['step','train_accuracy_step','validation_accuracy','train_accuracy_epoch']

# Remove the specified column
df_loss = df.drop(columns_to_remove_loss, axis=1)
df_accuracy = df.drop(columns_to_remove_accuracy, axis=1)



df_loss.interpolate(method='linear').plot(x='epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot the values
df_accuracy.interpolate(method='linear').plot(x='epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
