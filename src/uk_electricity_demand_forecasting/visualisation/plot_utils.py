"""
This script contains standard user plotting settings to create
exploratory and result oriented visualations throughout the project.
"""

## Import Libraries
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import seaborn as sns
from pydantic import BaseModel, ConfigDict, Field


class Plotter(BaseModel):
    plot_with_plotly: bool = Field(default=True, description="Whether to use Plotly for plotting.")
    plot_with_seaborn: bool = Field(default=False, description="Whether to use Seaborn/Matplotlib for plotting.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    class PlotConfig(BaseModel):
        width: int = Field(..., description="Default width for plots.")
        height: int = Field(..., description="Default height for plots.")
        title: str = Field(default="Default Title", description="Default title for plots.")

        @classmethod
        def default_for_plotly(cls):
            """Default plot size settings for Plotly plots"""
            return cls(width=800, height=500)

        @classmethod
        def default_for_seaborn(cls):
            """Default plot size settings for Seaborn/Matplotlib plots"""
            return cls(width=8, height=5)

    settings: PlotConfig = Field(default_factory=PlotConfig)

    def apply_plotly_settings(self):
        """Applies the standard Plotly settings"""
        pio.templates.default = "simple_white"
        px.defaults.width = self.settings.width
        px.defaults.height = self.settings.height

        excel_style_template = dict(
            layout=dict(
                font=dict(family="Trebuchet MS, sans-serif", size=11, color="black"),
                title=dict(font=dict(size=18, color="black")),
                paper_bgcolor="white",
                plot_bgcolor="white",
                xaxis=dict(
                    showgrid=True,
                    gridcolor="lightgray",
                    zeroline=False,
                    showline=True,
                    linecolor="black",
                    linewidth=1,
                    ticks="outside",
                    tickcolor="black",
                    mirror=True,
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="lightgray",
                    zeroline=False,
                    showline=True,
                    linecolor="black",
                    linewidth=1,
                    ticks="outside",
                    tickcolor="black",
                    mirror=True,
                ),
                margin=dict(l=60, r=30, t=30, b=60),
                legend=dict(font=dict(size=12, color="black")),
            )
        )
        pio.templates["excel_style"] = excel_style_template
        pio.templates.default = "excel_style"

    def apply_seaborn_settings(self):
        """Applies the standard Seaborn/Matplotlib settings"""

        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (self.settings.width, self.settings.height)
        plt.rcParams["font.family"] = "Trebuchet MS, sans-serif"
        plt.rcParams["font.size"] = 11
        plt.rcParams["axes.titlesize"] = 18
        plt.rcParams["axes.titlecolor"] = "black"
        plt.rcParams["axes.labelcolor"] = "black"
        plt.rcParams["xtick.color"] = "black"
        plt.rcParams["ytick.color"] = "black"
        plt.rcParams["legend.fontsize"] = 12
        plt.rcParams["legend.edgecolor"] = "black"


Plotter.model_rebuild()

# # Application
# plotter = Plotter(plot_with_plotly=True)

# fig = px.line(data, x="x_column", y="y_column", title="Plot Title")
# fig.show()

# plotter = Plotter(plot_with_seaborn=True)
# sns.lineplot(data=data, x="x_column", y="y_column")
# plt.title("Plot Title")
# plt.show()
# ----------------------------------------------------------
# def plotly_user_standard_settings(pio, px):
#     """
#     This function enforces the standard settings for plotly plots
#     created throughout at various stages in the project
#     """
#     pio.templates.default = "simple_white"
#     px.defaults.width = 800
#     px.defaults.height = 500

#     excel_style_template = dict(
#         layout=dict(
#             font=dict(family="Trebuchet MS, sans-serif", size=11, color="black"),
#             title=dict(font=dict(size=18, color="black")),
#             paper_bgcolor="white",
#             plot_bgcolor="white",
#             xaxis=dict(
#                 showgrid=True,
#                 gridcolor="lightgray",
#                 zeroline=False,
#                 showline=True,
#                 linecolor="black",
#                 linewidth=1,
#                 ticks="outside",
#                 tickcolor="black",
#                 mirror=True,
#             ),
#             yaxis=dict(
#                 showgrid=True,
#                 gridcolor="lightgray",
#                 zeroline=False,
#                 showline=True,
#                 linecolor="black",
#                 linewidth=1,
#                 ticks="outside",
#                 tickcolor="black",
#                 mirror=True,
#             ),
#             margin=dict(l=60, r=30, t=30, b=60),
#             legend=dict(font=dict(size=12, color="black")),
#         )
#     )

#     # Register the template and set it as default
#     pio.templates["excel_style"] = excel_style_template
#     pio.templates.default = "excel_style"

#     return pio
