"""NLP Katas visualization helpers.

This module is automatically available in kata code execution.
Functions print special markers that the frontend interprets as rich content.
"""

import json
import base64
import io


def show_html(html: str) -> None:
    """Render HTML in the output panel."""
    print(f"<!--NLP_RICH:html-->\n{html}\n<!--/NLP_RICH-->")


def show_svg(svg: str) -> None:
    """Render SVG in the output panel."""
    print(f"<!--NLP_RICH:svg-->\n{svg}\n<!--/NLP_RICH-->")


def show_image(data: bytes, mime: str = "image/png") -> None:
    """Render a binary image (PNG, JPEG, etc.) as inline base64."""
    b64 = base64.b64encode(data).decode("ascii")
    data_uri = f"data:{mime};base64,{b64}"
    print(f"<!--NLP_RICH:image-->\n{data_uri}\n<!--/NLP_RICH-->")


def show_chart(chart_spec: dict) -> None:
    """Render an interactive chart from a JSON specification.

    chart_spec should have:
      - type: "bar" | "line" | "scatter" | "pie" | "heatmap"
      - title: str (optional)
      - labels: list[str]  (x-axis labels for bar/line)
      - datasets: list of { label: str, data: list[float], color?: str }

    For scatter:
      - datasets: list of { label: str, data: list of {"x": float, "y": float} }

    For heatmap:
      - x_labels: list[str], y_labels: list[str]
      - data: list[list[float]] (2D matrix)
    """
    print(f"<!--NLP_RICH:chart-->\n{json.dumps(chart_spec)}\n<!--/NLP_RICH-->")


def _patch_matplotlib():
    """Monkey-patch matplotlib.pyplot.show() to output base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _rich_show(*args, **kwargs):
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                        facecolor="none", edgecolor="none")
            buf.seek(0)
            show_image(buf.read(), "image/png")
            buf.close()
            plt.close("all")

        plt.show = _rich_show
    except ImportError:
        pass  # matplotlib not installed


# Auto-patch matplotlib on import
_patch_matplotlib()
