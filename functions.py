import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import kaggle
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel
from matplotlib.colors import LinearSegmentedColormap


def download_kaggle_dataset(dataset: str, save_path: str = 'data') -> None:
    """
    Downloads a dataset from Kaggle and saves it in the specified directory.
    If the directory does not exist, it will be created.

    Args:
    - dataset (str): The dataset identifier in the format 'username/dataset-name'.
    - save_path (str): The directory where the dataset should be saved. Default is 'data'.

    Raises:
    - kaggle.rest.ApiException: If the dataset download fails due to an API error.
    - OSError: If there is an error creating the directory.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    kaggle.api.dataset_download_files(dataset, path=save_path, unzip=True)


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the given training data.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.
    random_state (int, optional): Random state for reproducibility. Default is 42.

    Returns:
    RandomForestClassifier: Trained Random Forest model.
    """
    rf_model = RandomForestClassifier(random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the model on the test data.

    Parameters:
    model: Trained model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test labels.

    Returns:
    tuple: Predicted labels, classification report, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, report, cm

def plot_feature_importances(importances: np.ndarray, columns: list, title: str):
    """
    Plot the feature importances.

    Parameters:
    importances (np.ndarray): Feature importances.
    columns (list): List of feature names.
    title (str): Plot title.
    """
    feature_importances = pd.Series(importances, index=columns)
    feature_importances.sort_values().plot(kind='barh', figsize=(10, 6))
    plt.title(title)
    plt.show()

def plot_precision_recall_curve(y_test: pd.Series, y_scores: np.ndarray):
    """
    Plot the Precision-Recall vs Threshold curve.

    Parameters:
    y_test (pd.Series): True labels.
    y_scores (np.ndarray): Predicted scores.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    plt.plot(thresholds, precision[:-1], "b--", label="Precision")
    plt.plot(thresholds, recall[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="best")
    plt.title("Precision-Recall vs Threshold Curve")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.show()

def create_custom_colormap() -> tuple:
    """
    Create custom colormaps for the confusion matrix.

    Returns:
    tuple: Colormaps for true and false values.
    """
    true_colors = [(1.0, 1.0, 1.0), (0.0, 1.0, 0.0)]
    false_colors = [(1.0, 1.0, 1.0), (1.0, 0.0, 0.0)]
    true_cmap = LinearSegmentedColormap.from_list('true_cmap', true_colors, N=100)
    false_cmap = LinearSegmentedColormap.from_list('false_cmap', false_colors, N=100)
    return true_cmap, false_cmap

def get_color(value: float, is_true: bool, true_cmap: LinearSegmentedColormap, false_cmap: LinearSegmentedColormap) -> str:
    """
    Get the color for a value in the confusion matrix.

    Parameters:
    value (float): Normalized value.
    is_true (bool): Whether the value is a true positive/negative.
    true_cmap (LinearSegmentedColormap): Colormap for true values.
    false_cmap (LinearSegmentedColormap): Colormap for false values.

    Returns:
    str: Color as a hex string.
    """
    return true_cmap(value) if is_true else false_cmap(value)

def plot_confusion_matrix(cm: np.ndarray, title: str):
    """
    Plot the confusion matrix.

    Parameters:
    cm (np.ndarray): Confusion matrix.
    title (str): Plot title.
    """
    cm_df = pd.DataFrame(cm, index=['Actual No', 'Actual Yes'], columns=['Predicted No', 'Predicted Yes'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    true_cmap, false_cmap = create_custom_colormap()

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            is_true = (i == j)
            color = get_color(cm_normalized[i, j], is_true, true_cmap, false_cmap)
            ax.add_patch(plt.Rectangle([j, i], 1, 1, facecolor=color))
            ax.text(j + 0.5, i + 0.5, int(value), ha='center', va='center', fontsize=14, color='black')

    ax.set_xticks(np.arange(cm.shape[1]) + 0.5)
    ax.set_yticks(np.arange(cm.shape[0]) + 0.5)
    ax.set_xticklabels(['Predicted No', 'Predicted Yes'])
    ax.set_yticklabels(['Actual No', 'Actual Yes'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    ax.set_xlim([0, cm.shape[1]])
    ax.set_ylim([0, cm.shape[0]])
    ax.invert_yaxis()

    plt.show()

def perform_paired_t_test(metric_name: str, original_scores: np.ndarray, deduplicated_scores: np.ndarray, p_value_threshold: float = 0.05):
    """
    Perform a paired t-test and print the results.

    Parameters:
    metric_name (str): Name of the metric being tested.
    original_scores (np.ndarray): Scores from the original data.
    deduplicated_scores (np.ndarray): Scores from the deduplicated data.
    p_value_threshold (float, optional): Threshold for statistical significance. Default is 0.05.
    """
    t_stat, p_value = ttest_rel(original_scores, deduplicated_scores)
    print(f"Paired t-test for {metric_name}: t-statistic = {t_stat}, p-value = {p_value}")
    if p_value < p_value_threshold:
        print(f"The difference in {metric_name} between the original and deduplicated data is statistically significant.")
    else:
        print(f"The difference in {metric_name} between the original and deduplicated data is not statistically significant.")

def format_classification_report(report: dict) -> pd.DataFrame:
    """
    Format the classification report for better display.

    Parameters:
    report (dict): Classification report as a dictionary.

    Returns:
    pd.DataFrame: Formatted classification report.
    """
    rows = []
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            rows.append({
                'Class': label,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': metrics['support']
            })
    return pd.DataFrame(rows)

def format_classification_report(report: dict) -> pd.DataFrame:
    """
    Format the classification report for better display.

    Parameters:
    report (dict): Classification report as a dictionary.

    Returns:
    pd.DataFrame: Formatted classification report.
    """
    rows = []
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            rows.append({
                'Class': label,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': metrics['support']
            })
    return pd.DataFrame(rows)

def display_classification_report(report: dict, title: str = "Classification Report"):
    """
    Display the classification report as a well-formatted DataFrame.

    Parameters:
    report (dict): Classification report as a dictionary.
    title (str, optional): Title for the report. Default is "Classification Report".
    """
    df = format_classification_report(report)
    print(f"\n{title}:")
    display(df)
