import numpy as np

import plotly.graph_objs as go

from dataclasses import dataclass

from plotly.subplots import make_subplots

from .processing import (
    get_colors_from_numeric_values,
    combine_sequences
)

@dataclass
class PlotConfig:
    """General plot configuration"""
    width: int = 1000
    height: int = 800
    type: str = "png"
    font_size: int = 16
    marker_size: int = 5
    line_width: int = 3

@dataclass
class AxisConfig:
    """Axis styling configuration"""
    showgrid_x: bool = True
    showgrid_y: bool = True
    n_y_axis_ticks: int = 5
    linewidth: int = 1
    tickwidth: int = 1
    ticklen: int = 5
    gridwidth: int = 1

@dataclass
class LayoutConfig:
    """Layout and spacing configuration"""
    show_subplot_titles: bool = False
    vertical_spacing: float = 0.1

@dataclass
class LegendConfig:
    """Legend and labeling configuration"""
    color_bar_length = 0.4
    color_bar_orientation = "v"
    color_bar_x_pos = 1.05
    color_bar_y_pos = 0.5


def config_fig(fig,
               plot_width=800,
               plot_height=600,
               plot_type="png",
               plot_title_for_download="plot"):
    """
    Configure plotly figure with download options and toolbar settings.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure object
    plot_width : int, default 800
        Width of the plot in pixels
    plot_height : int, default 600
        Height of the plot in pixels
    plot_type : str, default "png"
        Format for downloading the plot (e.g., "png", "jpeg")
    plot_title_for_download : str, default "plot"
        Title for the downloaded plot file

    Returns
    -------
    go.Figure
        Configured plotly figure
    """

    # Append the file extension to the title for download
    plot_title_for_download += f".{plot_type}"

    config = {
        'toImageButtonOptions': {
            'format': plot_type,
            'filename': plot_title_for_download,
            'width': plot_width,
            'height': plot_height
        },
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'sendDataToCloud',
            'hoverClosestCartesian',
            'hoverCompareCartesian',
            'lasso2d',
            'select2d'
        ]
    }

    fig.update_layout(
        width=plot_width,
        height=plot_height
    )

    fig._config = config

    return fig

def plot_unfolding(
        pychemelt_sample,
        plot_config: PlotConfig = None,
        axis_config: AxisConfig = None,
        layout_config: LayoutConfig = None,
        legend_config: LegendConfig = None):

    """
    Plot the unfolding curves, including the signal and the predicted curves

    Parameters
    ----------

    pychemelt_sample:
        pychemelt.Sample object
    plot_config : PlotConfig, optional
        Configuration for the overall plot
    axis_config : AxisConfig, optional
        Configuration for the axes
    layout_config : LayoutConfig, optional
        Configuration for the layout
    legend_config : LegendConfig, optional
        configuration for the legend

    """

    # Set defaults for configuration objects
    plot_config = plot_config or PlotConfig()
    axis_config = axis_config or AxisConfig()
    layout_config = layout_config or LayoutConfig()
    legend_config = legend_config or LegendConfig()

    # Extract the minimum and maximum denaturation concentration
    concs = pychemelt_sample.denaturant_concentrations

    min_conc = np.min(concs)
    max_conc = np.max(concs)

    colors = get_colors_from_numeric_values(concs, min_conc, max_conc)

    n_subplots = pychemelt_sample.nr_signals

    # Set number of rows: 2 if less than 8 plots, else 3
    nrows = 2 if n_subplots < 9 else 3
    nrows = min(nrows, n_subplots)  # Do not exceed the number of plots - case n equal 1

    ncols = int(np.ceil(n_subplots / nrows))

    row_arr = np.arange(1, nrows + 1)
    col_arr = np.arange(1, ncols + 1)
    # Row and column counters for subplotting
    row_col_info = combine_sequences(row_arr, col_arr)

    subplot_titles = pychemelt_sample.signal_names

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=layout_config.vertical_spacing,
        subplot_titles=subplot_titles)

    subplot_idx = 0

    nr_den = pychemelt_sample.nr_den

    fittings_done = pychemelt_sample.global_fit_params is not None

    ys_fit = None

    for i in range(n_subplots):

        row = row_col_info[subplot_idx][0]
        col = row_col_info[subplot_idx][1]

        xs = pychemelt_sample.temp_lst_multiple[i]
        ys = pychemelt_sample.signal_lst_multiple[i]

        if fittings_done:
            ys_fit = pychemelt_sample.predicted_lst_multiple[i]

        for j,conc in enumerate(concs):

            color = colors[j]

            x = xs[j]
            y = ys[j]

            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode='markers',
                    marker=dict(size=plot_config.marker_size, color=color),
                    name=f'{conc:.2f} M',
                    showlegend=False
                ),
                row=row, col=col
            )

            if fittings_done:
                fig.add_trace(
                    go.Scatter(
                        x=x, y=ys_fit[j], mode='lines',
                        line=dict(color='black', width=plot_config.line_width),
                        showlegend=False,
                        hoverinfo='skip',
                        hovertemplate=None
                    ),
                    row=row, col=col
                )

        subplot_idx += 1

    # Update subplot layout with white background and axis styling
    fig.update_layout(
        font_family="Roboto",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(font=dict(size=plot_config.font_size - 1))
    )

    for i in range(n_subplots):

        row = row_col_info[i][0]
        col = row_col_info[i][1]

        # Set the x-axis title only for the last row
        title_text_x = 'Temperature (°C)' if row == nrows else ''

        # Set the y-axis title only for the first column
        title_text_y = 'Signal' if col == 1 else ''

        fig.update_xaxes(
            title_text=title_text_x,
            showgrid=axis_config.showgrid_x,
            gridwidth=axis_config.gridwidth,
            gridcolor='lightgray',
            showline=True,
            linewidth=axis_config.linewidth,
            linecolor='black',
            zeroline=False,
            tickcolor='black',
            ticks="outside",
            tickwidth=axis_config.tickwidth,
            ticklen=axis_config.ticklen,
            title_font_size=plot_config.font_size,
            tickfont_size=plot_config.font_size,
            col = col,
            row = row
        )

        fig.update_yaxes(
            title_text=title_text_y,
            showgrid=axis_config.showgrid_y,
            gridwidth=axis_config.gridwidth,
            gridcolor='lightgray',
            showline=True,
            linewidth=axis_config.linewidth,
            linecolor='black',
            zeroline=False,
            tickcolor='black',
            ticks="outside",
            tickwidth=axis_config.tickwidth,
            ticklen=axis_config.ticklen,
            title_font_size=plot_config.font_size,
            tickfont_size=plot_config.font_size,
            nticks = axis_config.n_y_axis_ticks,
            col=col,
            row=row
        )

    # Build colorbar dict using legend_config values (orientation and x/y position)
    # Choose sensible anchors depending on orientation
    _xanchor = 'center' if legend_config.color_bar_orientation == 'h' else 'left'
    _yanchor = 'top'    if legend_config.color_bar_orientation == 'h' else 'middle'

    colorbar_dict = dict(
        title='[Denaturant] (M)',
        tickvals=[min_conc, 0.5*(min_conc + max_conc), max_conc],
        ticktext=[f"{min_conc:.2g}", f"{(min_conc + max_conc) * 0.5:.2g}", f"{max_conc:.2g}"],
        len=legend_config.color_bar_length,
        outlinewidth=1,
        ticks='outside',
        tickfont=dict(size=plot_config.font_size - 1),
        orientation=legend_config.color_bar_orientation,
        x=legend_config.color_bar_x_pos,
        y=legend_config.color_bar_y_pos,
        xanchor=_xanchor,
        yanchor=_yanchor
    )

    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                colorscale='Viridis',
                cmin=min_conc,
                cmax=max_conc,
                colorbar=colorbar_dict
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    fig = config_fig(
        fig,
        plot_config.width,
        plot_config.height,
        plot_config.type
    )

    return fig
