from __future__ import annotations

import panel as pn
import param


class PlaceholderPane(param.Parameterized):
    """Coming-soon placeholder for unimplemented dashboard panes."""

    def __init__(self, title: str, message: str = "Coming soon", **params):
        super().__init__(**params)
        self._title = title
        self._message = message

    def panel(self) -> pn.Column:
        """Return a centered placeholder column with title and message."""
        return pn.Column(
            pn.pane.HTML(
                f"<div style='padding:60px;text-align:center;color:#666'>"
                f"<h2 style='color:#999'>{self._title}</h2>"
                f"<p style='font-size:14px'>{self._message}</p>"
                f"</div>",
                sizing_mode="stretch_both",
            )
        )
