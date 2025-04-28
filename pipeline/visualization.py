import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

def visualize_evaluation_results(logs_dir='../logs'):
    # Load all evaluation files
    model_files = [f for f in os.listdir(logs_dir) if f.endswith("_evaluation.json")]

    if not model_files:
        print(f"No evaluation files found in {logs_dir}")
        return

    all_results = []
    for file in model_files:
        with open(os.path.join(logs_dir, file), "r") as f:
            results = json.load(f)
            all_results.extend(results)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # ===============================
    # 1. Scatter Plot: Training Time vs Test Accuracy
    # ===============================
    plt.figure(figsize=(10, 6))
    plt.scatter(df['training_time'], df['test_accuracy'], color='dodgerblue', s=100)

    for i in range(len(df)):
        plt.text(df['training_time'][i], df['test_accuracy'][i], df['model_name'][i], fontsize=9)

    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Training Time')
    plt.grid(True)
    plt.tight_layout()

    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/accuracy_vs_training_time.png')
    plt.show()

    # ===============================
    # 2. Scatter Plot: Validation Accuracy vs Test Accuracy
    # ===============================
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='validation_accuracy', y='test_accuracy', data=df, hue='model_name', s=100)
    plt.plot([0.85, 1.0], [0.85, 1.0], 'r--')  # Ideal line
    plt.title("Validation vs Test Accuracy")
    plt.xlabel("Validation Accuracy")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig('./figures/validation_vs_test_accuracy.png')
    plt.show()

    # ===============================
    # 3. Scatter Plot: Test Loss vs Test Accuracy
    # ===============================
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='test_loss', y='test_accuracy', data=df, hue='model_name', s=100)
    plt.title("Test Loss vs Test Accuracy")
    plt.xlabel("Test Loss")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig('./figures/test_loss_vs_test_accuracy.png')
    plt.show()

    # ===============================
    # 4. Radar Chart: Model Comparison
    # ===============================
    # Normalize values
    normalized_df = df.copy()
    columns_to_normalize = ['validation_accuracy', 'test_accuracy', 'test_loss', 'training_time']

    for col in columns_to_normalize:
        if col in ['test_loss', 'training_time']:
            normalized_df[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Radar chart setup
    labels = columns_to_normalize
    num_vars = len(labels)

    plt.figure(figsize=(8, 8))
    for i, row in normalized_df.iterrows():
        values = row[labels].tolist()
        values += values[:1]  # repeat first value to close the loop
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        plt.polar(angles, values, label=row['model_name'])

    plt.xticks([n / float(num_vars) * 2 * pi for n in range(num_vars)], labels)
    plt.title('Model Comparison - Radar Chart')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()

    plt.savefig('./figures/model_comparison_radar_chart.png')
    plt.show()

    # ===============================
    # 5. Print Markdown Table
    # ===============================
    print(df[['model_name', 'validation_accuracy', 'test_accuracy', 'test_loss', 'training_time']].to_markdown(index=False))

