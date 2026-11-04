import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import shapely.geometry as sg

from tools import pcb_preflight


def gerber(*polygons):
    lines = ["%MOMM*%", "%FSLAX34Y34*%", "%LPD*%"]
    for polygon in polygons:
        lines.append("G36*")
        for i, (x, y) in enumerate(polygon.exterior.coords):
            lines.append(
                f"X{round(x * 10000):07d}Y{round(y * 10000):07d}"
                f"D{2 if i == 0 else 1:02d}*")
        lines.append("G37*")
    return "\n".join(lines) + "\nM02*\n"


class CopperEdgeTests(unittest.TestCase):
    def check_copper(self, copper, outline=None, slots=(), overrides=None):
        if outline is None:
            outline = sg.box(0, 0, 10, 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "board.GML").write_text(gerber(outline, *slots))
            (root / "board.GTL").write_text(gerber(*copper))
            (root / "board.GBL").write_text(gerber(*copper))
            audit = pcb_preflight.Audit()
            with patch.object(pcb_preflight, "ROOT", root):
                pcb_preflight.audit_copper_edges(audit, {
                    "board": "board",
                    **({"copper_edge_clearance_mm": {"GTL": 0.4}}
                       if overrides is None else overrides),
                })
            return audit

    def test_missing_setting_checks_both_layers_at_routed_default(self):
        audit = self.check_copper([sg.box(.2, .2, 9.8, 9.8)], overrides={})
        self.assertTrue(audit.passed)
        self.assertEqual(len(audit.checks), 2)
        self.assertTrue(all("needs 0.200 mm" in c.detail for c in audit.checks))
        audit = self.check_copper([sg.box(.19, 2, 1, 3)], overrides={})
        self.assertEqual([c.passed for c in audit.checks], [False, False])

    def test_partial_override_keeps_other_layer_enabled(self):
        audit = self.check_copper([sg.box(.3, .3, 9.7, 9.7)])
        self.assertEqual([c.passed for c in audit.checks], [False, True])

    def test_exact_clearance_passes(self):
        self.assertTrue(self.check_copper([sg.box(.4, .4, 9.6, 9.6)]).passed)

    def test_isolated_artwork_too_close_fails(self):
        audit = self.check_copper([
            sg.box(2, 2, 8, 8), sg.box(.3, 4, .35, 4.1)])
        self.assertFalse(audit.passed)
        self.assertIn("0.3000 mm", audit.checks[0].detail)

    def test_entirely_external_copper_fails(self):
        audit = self.check_copper([sg.box(11, 4, 12, 5)])
        self.assertFalse(audit.passed)
        self.assertIn("outside board", audit.checks[0].detail)

    def test_actual_perimeter_not_bounding_box(self):
        # A chamfered corner cuts through this otherwise safely inset pad.
        outline = sg.Polygon([(0, 2), (2, 0), (10, 0), (10, 10), (0, 10)])
        self.assertFalse(self.check_copper(
            [sg.box(.5, .5, 1, 1)], outline).passed)

    def test_internal_plated_slot_not_treated_as_outer_edge(self):
        self.assertTrue(self.check_copper(
            [sg.box(3.9, 3.9, 6.1, 6.1)],
            slots=[sg.box(4, 4, 6, 6)]).passed)

    def test_missing_outline_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = pcb_preflight.Audit()
            with patch.object(pcb_preflight, "ROOT", Path(directory)):
                pcb_preflight.audit_copper_edges(audit, {
                    "board": "missing",
                    "copper_edge_clearance_mm": {"GBL": 0.4},
                })
            self.assertFalse(audit.passed)


def region(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class CopperEdgeExceptionTests(unittest.TestCase):
    """A profile may let copper inside a named region come closer to the
    edge than the layer's limit, or reach past it (a negative clearance)."""

    check_copper = CopperEdgeTests.check_copper

    def check_exception(self, copper, *exceptions):
        return self.check_copper(copper, overrides={
            "copper_edge_exceptions": list(exceptions)})

    def exception(self, clearance, polygon=None, layer="GTL"):
        return {
            "layer": layer,
            "polygon": polygon or region(-1, 4, 1, 6),
            "clearance_mm": clearance,
            "reason": "J2 pads at the notch",
        }

    def checks(self, audit):
        return {c.name: c for c in audit.checks}

    def test_copper_inside_the_region_meets_the_exception(self):
        audit = self.check_exception(
            [sg.box(.1, 4.5, 1, 5), sg.box(2, 2, 8, 8)], self.exception(0.1))
        checks = self.checks(audit)
        self.assertTrue(checks["GTL copper-to-board-edge clearance"].passed)
        exception = checks["GTL copper-to-board-edge exception 1"]
        self.assertTrue(exception.passed)
        self.assertIn("J2 pads at the notch", exception.detail)
        self.assertIn("minimum 0.1000 mm; allowed 0.100 mm", exception.detail)
        # GBL has the same copper and no exception.
        self.assertFalse(checks["GBL copper-to-board-edge clearance"].passed)

    def test_copper_closer_than_the_exception_fails(self):
        audit = self.check_exception([sg.box(.05, 4.5, 1, 5)], self.exception(0.1))
        self.assertFalse(
            self.checks(audit)["GTL copper-to-board-edge exception 1"].passed)

    def test_negative_clearance_allows_an_overhang(self):
        copper = [sg.box(-.08, 4.5, 1, 5)]
        ok = self.checks(self.check_exception(copper, self.exception(-0.08)))
        self.assertTrue(ok["GTL copper-to-board-edge exception 1"].passed)
        self.assertIn("minimum -0.0800 mm", ok["GTL copper-to-board-edge exception 1"].detail)
        short = self.checks(self.check_exception(copper, self.exception(-0.05)))
        self.assertFalse(short["GTL copper-to-board-edge exception 1"].passed)

    def test_copper_outside_the_region_keeps_the_layer_limit(self):
        audit = self.check_exception(
            [sg.box(.1, 4.5, 1, 5), sg.box(.1, 8, 1, 9)], self.exception(0.1))
        checks = self.checks(audit)
        self.assertTrue(checks["GTL copper-to-board-edge exception 1"].passed)
        self.assertFalse(checks["GTL copper-to-board-edge clearance"].passed)

    def test_an_exception_covering_no_copper_fails(self):
        audit = self.check_exception(
            [sg.box(2, 2, 8, 8)], self.exception(0.1, region(-1, 0, 1, 1)))
        check = self.checks(audit)["GTL copper-to-board-edge exception 1"]
        self.assertFalse(check.passed)
        self.assertIn("covers no copper", check.detail)

    def test_a_malformed_exception_fails_closed(self):
        for bad in (
            {"layer": "GTL", "polygon": region(0, 0, 1, 1), "clearance_mm": 0.1},
            {"layer": "GXL", "polygon": region(0, 0, 1, 1), "clearance_mm": 0.1,
             "reason": "r"},
            {"layer": "GTL", "polygon": [[0, 0], [1, 1]], "clearance_mm": 0.1,
             "reason": "r"},
            {"layer": "GTL", "polygon": region(0, 0, 1, 1),
             "clearance_mm": float("nan"), "reason": "r"},
        ):
            audit = self.check_exception([sg.box(2, 2, 8, 8)], bad)
            self.assertFalse(audit.passed, bad)
            self.assertIn(
                "cannot read copper_edge_exceptions",
                " ".join(c.detail for c in audit.checks))
