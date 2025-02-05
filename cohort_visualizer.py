# cohort_visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class CohortVisualizer:
    """
    A class to generate customizable visualizations for cohort analysis.
    Allows dynamic axis selection, titles, and faceting for intermediate DataFrames.
    """

    @staticmethod
    def scatter_plot(df: pd.DataFrame, x: str, y: str, hue: str = None, title: str = None, save_path: str = None):
        """
        Create a scatter plot with optional hue differentiation.

        :param df: The DataFrame to visualize.
        :param x: Column name for X-axis.
        :param y: Column name for Y-axis.
        :param hue: Column name for color differentiation (optional).
        :param title: Plot title (optional).
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="coolwarm", alpha=0.7)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title if title else f"Scatter Plot: {x} vs {y}")
        plt.legend(title=hue)
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def box_plot(df: pd.DataFrame, x: str, y: str, hue: str = None, title: str = None, save_path: str = None):
        """
        Create a box plot to visualize distributions across categories.

        :param df: The DataFrame to visualize.
        :param x: Column name for categorical X-axis.
        :param y: Column name for numeric Y-axis.
        :param hue: Column name for additional categorization (optional).
        :param title: Plot title (optional).
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=x, y=y, hue=hue, palette="coolwarm")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title if title else f"Box Plot: {x} vs {y}")
        plt.legend(title=hue)
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def bar_plot(df: pd.DataFrame, x: str, y: str, hue: str = None, title: str = None, save_path: str = None):
        """
        Create a box plot to visualize distributions across categories.

        :param df: The DataFrame to visualize.
        :param x: Column name for categorical X-axis.
        :param y: Column name for numeric Y-axis.
        :param hue: Column name for additional categorization (optional).
        :param title: Plot title (optional).
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x=x, y=y, hue=hue, palette="coolwarm")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title if title else f"Box Plot: {x} vs {y}")
        plt.legend(title=hue)
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def histogram(df: pd.DataFrame, column: str, bins: int = 30, title: str = None, save_path: str = None):
        """
        Create a histogram to visualize the distribution of a numerical variable.

        :param df: The DataFrame to visualize.
        :param column: Column name for histogram.
        :param bins: Number of bins (default: 30).
        :param title: Plot title (optional).
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=column, bins=bins, kde=True, color="blue", alpha=0.7)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(title if title else f"Histogram: {column}")
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def line_plot(df: pd.DataFrame, x: str, y: str, hue: str = None, title: str = None, save_path: str = None):
        """
        Create a line plot for time series or trends.

        :param df: The DataFrame to visualize.
        :param x: Column name for X-axis.
        :param y: Column name for Y-axis.
        :param hue: Column name for different lines (optional).
        :param title: Plot title (optional).
        """
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o", palette="coolwarm")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title if title else f"Line Plot: {x} vs {y}")
        plt.legend(title=hue)
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def facet_grid(df: pd.DataFrame, x: str, y: str, facet_col: str, kind: str = "scatter", title: str = None, save_path: str = None):
        """
        Create faceted plots across different categories.

        :param df: The DataFrame to visualize.
        :param x: Column name for X-axis.
        :param y: Column name for Y-axis.
        :param facet_col: Column name for faceting categories.
        :param kind: Type of plot ("scatter", "line", "box", "hist").
        :param title: Plot title (optional).
        """
        g = sns.FacetGrid(df, col=facet_col, col_wrap=4, height=4, sharex=False, sharey=False)

        if kind == "scatter":
            g.map_dataframe(sns.scatterplot, x=x, y=y, alpha=0.7)
        elif kind == "line":
            g.map_dataframe(sns.lineplot, x=x, y=y, marker="o")
        elif kind == "box":
            g.map_dataframe(sns.boxplot, x=x, y=y)
        elif kind == "hist":
            g.map_dataframe(sns.histplot, x=x, bins=30, kde=True)

        g.set_titles(col_template="{col_name}")
        g.set_axis_labels(x, y)
        g.fig.suptitle(title if title else f"Facet Grid: {x} vs {y} by {facet_col}", y=1.02)
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path)
        plt.show()
