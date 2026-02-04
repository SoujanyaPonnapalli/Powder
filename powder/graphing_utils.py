#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graphing utilities for visualizing simulation results.

Provides functions for creating line plots, histograms (PDF),
cumulative distribution plots (CDF), and comparison charts.
"""

import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass, astuple


@dataclass
class Line:
    x_values: list[float]
    y_values: list[float]
    name: str


@dataclass
class Graph:
    title: str
    x_axis_name: str
    y_axis_name: str
    lines: list[Line]


def make_fig(graph: Graph) -> go.Figure:
    """Create a line plot figure.

    Args:
        graph: Graph specification with title, axes, and lines.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for line in graph.lines:
        x_values, y_values, name = astuple(line)
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", name=name))

    fig.update_layout(
        title=graph.title,
        xaxis_title=graph.x_axis_name,
        yaxis_title=graph.y_axis_name,
        showlegend=True,
    )

    # fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    return fig


def make_pdf_histogram(
    samples: list[float],
    title: str,
    x_label: str,
    bins: int = 50,
    y_label: str = "Density",
    color: str = "steelblue",
    show_mean: bool = True,
    show_median: bool = True,
) -> go.Figure:
    """Create a probability density function (PDF) histogram.

    Args:
        samples: Data samples to plot.
        title: Plot title.
        x_label: X-axis label.
        bins: Number of histogram bins.
        y_label: Y-axis label.
        color: Bar color.
        show_mean: Whether to show a vertical line at the mean.
        show_median: Whether to show a vertical line at the median.

    Returns:
        Plotly Figure object.
    """
    if not samples:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no data)")
        return fig

    samples_arr = np.array(samples)

    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=samples_arr,
            nbinsx=bins,
            histnorm="probability density",
            name="PDF",
            marker_color=color,
            opacity=0.7,
        )
    )

    # Add mean line
    if show_mean:
        mean_val = np.mean(samples_arr)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top right",
        )

    # Add median line
    if show_median:
        median_val = np.median(samples_arr)
        fig.add_vline(
            x=median_val,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
    )

    return fig


def make_cdf_plot(
    samples: list[float],
    title: str,
    x_label: str,
    y_label: str = "Cumulative Probability",
    color: str = "steelblue",
    show_percentiles: list[float] | None = None,
) -> go.Figure:
    """Create a cumulative distribution function (CDF) plot.

    Args:
        samples: Data samples to plot.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        color: Line color.
        show_percentiles: Optional list of percentiles to mark (e.g., [50, 90, 99]).

    Returns:
        Plotly Figure object.
    """
    if not samples:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no data)")
        return fig

    samples_arr = np.array(samples)
    sorted_samples = np.sort(samples_arr)
    cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

    fig = go.Figure()

    # Add CDF line
    fig.add_trace(
        go.Scatter(
            x=sorted_samples,
            y=cdf,
            mode="lines",
            name="CDF",
            line=dict(color=color, width=2),
        )
    )

    # Add percentile markers
    if show_percentiles:
        for p in show_percentiles:
            p_val = np.percentile(samples_arr, p)
            fig.add_hline(
                y=p / 100,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"p{p}: {p_val:.2f}",
                annotation_position="right",
            )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=False,
    )

    return fig


def make_availability_boxplot(
    results_list: list,
    labels: list[str],
    title: str = "Availability Comparison",
) -> go.Figure:
    """Create a box plot comparing availability across different configurations.

    Args:
        results_list: List of MonteCarloResults objects.
        labels: Labels for each configuration.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for results, label in zip(results_list, labels):
        # Convert to percentage
        avail_percent = [a * 100 for a in results.availability_samples]
        fig.add_trace(
            go.Box(
                y=avail_percent,
                name=label,
                boxmean=True,
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title="Availability (%)",
        showlegend=False,
    )

    return fig


def make_time_to_loss_comparison(
    results_list: list,
    labels: list[str],
    title: str = "Time to Data Loss Comparison",
    actual: bool = True,
    time_unit: str = "days",
) -> go.Figure:
    """Create a box plot comparing time to data loss across configurations.

    Args:
        results_list: List of MonteCarloResults objects.
        labels: Labels for each configuration.
        title: Plot title.
        actual: If True, use actual loss times; if False, use potential loss times.
        time_unit: Time unit for display ("seconds", "hours", "days").

    Returns:
        Plotly Figure object.
    """
    # Time conversion factor
    if time_unit == "seconds":
        factor = 1.0
    elif time_unit == "hours":
        factor = 1 / 3600
    elif time_unit == "days":
        factor = 1 / 86400
    else:
        factor = 1.0

    fig = go.Figure()

    for results, label in zip(results_list, labels):
        if actual:
            samples = results.time_to_actual_loss_samples_filtered()
        else:
            samples = results.time_to_potential_loss_samples_filtered()

        # Convert time units
        samples_converted = [s * factor for s in samples]

        if samples_converted:
            fig.add_trace(
                go.Box(
                    y=samples_converted,
                    name=label,
                    boxmean=True,
                )
            )

    loss_type = "Actual" if actual else "Potential"
    fig.update_layout(
        title=f"{title} ({loss_type})",
        yaxis_title=f"Time to Data Loss ({time_unit})",
        showlegend=False,
    )

    return fig


def make_multi_cdf_plot(
    samples_list: list[list[float]],
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str = "Cumulative Probability",
) -> go.Figure:
    """Create overlaid CDF plots for comparing distributions.

    Args:
        samples_list: List of sample lists to compare.
        labels: Labels for each sample list.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    colors = ["steelblue", "coral", "green", "purple", "orange", "brown"]

    for i, (samples, label) in enumerate(zip(samples_list, labels)):
        if not samples:
            continue

        samples_arr = np.array(samples)
        sorted_samples = np.sort(samples_arr)
        cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=sorted_samples,
                y=cdf,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=True,
    )

    return fig
