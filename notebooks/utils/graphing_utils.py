#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    return fig
