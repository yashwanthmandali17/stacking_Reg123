"""Plotly chart builders for the House Price Stacking Regressor Dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = px.colors.qualitative.Bold


def plot_price_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Sale_Price", nbins=60,
        title="Sale Price Distribution",
        labels={"Sale_Price": "Sale Price ($)"},
        color_discrete_sequence=[PALETTE[0]],
        template="plotly_white",
    )
    fig.update_layout(bargap=0.05)
    return fig


def plot_sqft_vs_price(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="Total_SqFt", y="Sale_Price",
        color="Neighborhood",
        title="Total SqFt vs Sale Price",
        labels={"Total_SqFt": "Total Square Footage", "Sale_Price": "Sale Price ($)"},
        opacity=0.65,
        template="plotly_white",
    )
    return fig


def plot_neighborhood_boxplot(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="Neighborhood", y="Sale_Price",
        color="Neighborhood",
        title="Sale Price by Neighborhood",
        labels={"Sale_Price": "Sale Price ($)"},
        template="plotly_white",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_quality_bar(df: pd.DataFrame) -> go.Figure:
    avg = df.groupby("Overall_Qual")["Sale_Price"].median().reset_index()
    fig = px.bar(
        avg, x="Overall_Qual", y="Sale_Price",
        title="Median Sale Price by Overall Quality",
        labels={"Overall_Qual": "Overall Quality (1–10)", "Sale_Price": "Median Price ($)"},
        color="Overall_Qual",
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    return fig


def plot_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric = df.select_dtypes(include=[float, int])
    corr = numeric.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        template="plotly_white",
        height=550,
    )
    return fig


def plot_predictions_vs_actual(y_test: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode="markers",
        marker=dict(color=PALETTE[2], opacity=0.6, size=6),
        name="Predictions",
    ))
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    fig.add_trace(go.Scatter(
        x=lim, y=lim,
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Perfect Fit",
    ))
    fig.update_layout(
        title="Predicted vs Actual Sale Price",
        xaxis_title="Actual Price ($)",
        yaxis_title="Predicted Price ($)",
        template="plotly_white",
    )
    return fig


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    residuals = y_pred - y_test
    fig = go.Figure(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(color=PALETTE[4], opacity=0.55, size=6),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Residual Plot (Predicted vs Residual)",
        xaxis_title="Predicted Price ($)",
        yaxis_title="Residual ($)",
        template="plotly_white",
    )
    return fig
