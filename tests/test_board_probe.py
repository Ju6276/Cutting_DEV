import pytest
import torch

from scripts.check_board_continue import phase_translation


def row(stage, x):
    return dict(stage=stage, pose=[[x, 0., 1., 0., 0., 0., 1.]])


def test_slide_excludes_drift_before_its_start():
    # First-contact subtraction would wrongly report 9 mm. The lateral command
    # starts after a separate phase drifted -1 mm and moves exactly 10 mm.
    records=[row("contact",0.), row("inward",-.001),
             row("slide",.004), row("slide",.009), row("lift",.02)]
    torch.testing.assert_close(phase_translation(records,"slide"),
                               torch.tensor([[.01,0.,0.]],dtype=torch.float64))


@pytest.mark.parametrize("records", [
    [row("slide",0.)],
    [row("contact",0.),row("lift",.01)],
    [row("contact",0.),row("slide",.01),row("hold",.02),row("slide",.03)],
    [row("contact",0.),row("slide",float("nan"))],
])
def test_incomplete_or_ambiguous_probe_is_rejected(records):
    with pytest.raises(ValueError):
        phase_translation(records,"slide")
