import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_df_diff(df_orig: pd.DataFrame, df_recom: pd.DataFrame, threshold: float):
    """plot difference of df

    Args:
        df_orig (pd.DataFrame): original data.
        df_recom (pd.DataFrame): recommended data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    ax = axes[0]
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    sns.heatmap(df_orig.values, ax=ax)
    ax.set_title("original")
    ax = axes[1]
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    sns.heatmap(df_recom.values, ax=ax)
    ax.set_title("low rank approx.")
    ax = axes[2]
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    sns.heatmap(df_recom.values-df_orig.values > threshold, ax=ax)
    ax.set_title("difference")
