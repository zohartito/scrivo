from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import sys

if sys.platform != "darwin":
    raise unittest.SkipTest("AppKit-only UI — macOS-only in v1")

import AppKit
import Quartz

from ui import StatusUI, screen_containing_point, window_info_is_visible


class _Screen:
    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        self._frame = AppKit.NSMakeRect(x, y, width, height)

    def frame(self):
        return self._frame


class OverlayPlacementTests(unittest.TestCase):
    def test_screen_selection_handles_stacked_displays(self):
        built_in = _Screen(0, 0, 1728, 1117)
        external = _Screen(-84, 1117, 1920, 1080)

        self.assertIs(
            screen_containing_point((built_in, external), AppKit.NSMakePoint(700, 500)),
            built_in,
        )
        self.assertIs(
            screen_containing_point((built_in, external), AppKit.NSMakePoint(700, 1600)),
            external,
        )

    def test_screen_selection_rejects_points_outside_every_display(self):
        screen = _Screen(0, 0, 1728, 1117)
        self.assertIsNone(
            screen_containing_point((screen,), AppKit.NSMakePoint(-500, -500))
        )

    def test_window_server_visibility_requires_onscreen_nonzero_alpha(self):
        self.assertTrue(window_info_is_visible([{
            Quartz.kCGWindowIsOnscreen: True,
            Quartz.kCGWindowAlpha: 1.0,
        }]))
        self.assertFalse(window_info_is_visible([{
            Quartz.kCGWindowIsOnscreen: False,
            Quartz.kCGWindowAlpha: 1.0,
        }]))
        self.assertFalse(window_info_is_visible([{
            Quartz.kCGWindowIsOnscreen: True,
            Quartz.kCGWindowAlpha: 0.0,
        }]))

    def test_hidden_live_panel_is_rebuilt_and_raised_once(self):
        status = StatusUI.__new__(StatusUI)
        status._visibility_generation = 7
        status._mode = "recording"
        old_panel = MagicMock()
        old_panel.windowNumber.return_value = 42
        replacement_panel = MagicMock()
        replacement_orb = MagicMock()
        status._panel = old_panel
        status._orb = MagicMock()
        messages = []
        status.on_overlay_event = messages.append

        hidden = [{
            Quartz.kCGWindowIsOnscreen: False,
            Quartz.kCGWindowAlpha: 1.0,
        }]
        with (patch.object(Quartz, "CGWindowListCopyWindowInfo", return_value=hidden),
              patch.object(status, "_make_island", return_value=replacement_panel),
              patch.object(status, "_make_orb", return_value=replacement_orb),
              patch.object(status, "_apply_indicator_state") as apply_state,
              patch.object(status, "_show_panel") as show_panel):
            status._verify_panel_visible(7, recovery_attempted=False)

        old_panel.orderOut_.assert_called_once_with(None)
        self.assertIs(status._panel, replacement_panel)
        self.assertIs(status._orb, replacement_orb)
        apply_state.assert_called_once_with()
        show_panel.assert_called_once_with(recovery_attempted=True)
        self.assertEqual(messages, ["! overlay hidden after raise — rebuilding panel"])

    def test_recovery_does_not_loop_if_replacement_stays_hidden(self):
        status = StatusUI.__new__(StatusUI)
        status._visibility_generation = 8
        status._mode = "recording"
        status._panel = MagicMock()
        status._panel.windowNumber.return_value = 43
        messages = []
        status.on_overlay_event = messages.append

        hidden = [{
            Quartz.kCGWindowIsOnscreen: False,
            Quartz.kCGWindowAlpha: 1.0,
        }]
        with patch.object(Quartz, "CGWindowListCopyWindowInfo", return_value=hidden):
            status._verify_panel_visible(8, recovery_attempted=True)

        status._panel.orderOut_.assert_not_called()
        self.assertEqual(messages, ["! overlay recovery failed — panel still hidden"])


class AlertIconTests(unittest.TestCase):
    """Scrivo.app runs Python as a child, and AppKit loads Python's bundle icon
    when the app finishes launching, so dialogs showed Python's rocket even
    after use_app_icon. Every dialog carries Scrivo's icon itself."""

    def test_every_dialog_is_built_with_scrivos_icon(self):
        import tempfile
        from pathlib import Path
        import ui
        with tempfile.TemporaryDirectory() as folder:
            icon = Path(folder) / "Scrivo.png"
            image = AppKit.NSImage.alloc().initWithSize_((16, 16))
            image.lockFocus(); AppKit.NSColor.orangeColor().set()
            AppKit.NSBezierPath.fillRect_(((0, 0), (16, 16))); image.unlockFocus()
            rep = AppKit.NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
            rep.representationUsingType_properties_(AppKit.NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(
                str(icon), True)
            ui.use_app_icon(icon)
        alert = ui.new_alert()
        self.assertIsNotNone(alert.icon())
        self.assertEqual(tuple(alert.icon().size()), (16.0, 16.0))

    def test_the_status_item_has_its_own_menu_bar_identity(self):
        # macOS keys icons by app + autosave name; the default "Item-0" made
        # Scrivo the same icon as any other Python menu bar app (2026-11-04).
        import ui
        from pathlib import Path
        self.assertNotEqual(ui.STATUS_ITEM_AUTOSAVE, "Item-0")
        source = (Path(__file__).resolve().parents[1] / "ui.py").read_text(encoding="utf-8")
        self.assertIn("self._status.setAutosaveName_(STATUS_ITEM_AUTOSAVE)", source)

    def test_no_dialog_is_built_without_it(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        counts = {name: (root / name).read_text(encoding="utf-8").count("NSAlert.alloc().init()")
                  for name in ("ui.py", "settings_window.py")}
        self.assertEqual(counts, {"ui.py": 1, "settings_window.py": 0}, "only new_alert() builds an NSAlert")


class QuietMicHintTests(unittest.TestCase):
    """The hint is display only: it changes what the island says, nothing else."""

    def status(self, state="recording"):
        status = StatusUI.__new__(StatusUI)
        status._indicator_state = state
        status._quiet = False
        status._heard_at = 100.0
        status.on_overlay_event = None
        status._position = MagicMock()
        return status

    def tick(self, status, rms, now):
        with patch("ui.time.monotonic", return_value=now):
            status.track_quiet(rms)

    def test_a_silent_mic_shows_the_hint_after_the_quiet_window(self):
        import ui
        status = self.status()
        self.tick(status, 0.015, 100.0 + ui.QUIET_HINT_S - 0.1)   # the AirPods clip's loudest block
        self.assertFalse(status._quiet)
        self.tick(status, 0.015, 100.0 + ui.QUIET_HINT_S)
        self.assertTrue(status._quiet)
        status._position.assert_called_once_with()

    def test_speech_clears_it_and_restarts_the_window(self):
        import ui
        status = self.status()
        self.tick(status, 0.0, 100.0 + ui.QUIET_HINT_S)
        self.tick(status, 0.03, 120.0)
        self.assertFalse(status._quiet)
        self.tick(status, 0.0, 120.0 + ui.QUIET_HINT_S - 0.1)
        self.assertFalse(status._quiet)

    def test_only_while_recording(self):
        import ui
        for state in ("waking", "transcribing", "idle"):
            status = self.status(state)
            self.tick(status, 0.0, 100.0 + 10 * ui.QUIET_HINT_S)
            self.assertFalse(status._quiet, state)


class MenuBarIconTests(unittest.TestCase):
    def status(self, frame_height=None, no_window=False):
        status = StatusUI.__new__(StatusUI)
        status._status = MagicMock()
        window = status._status.button.return_value.window.return_value
        if no_window:
            status._status.button.return_value.window.return_value = None
        else:
            window.frame.return_value = AppKit.NSMakeRect(1358, 1084, 29, frame_height)
        status.show_info = MagicMock()
        return status

    def test_a_zero_height_button_window_is_a_hidden_icon(self):
        # what a hidden status item reports in-process on macOS 27.2
        self.assertTrue(self.status(frame_height=0.0).icon_hidden())
        self.assertTrue(self.status(no_window=True).icon_hidden())
        self.assertFalse(self.status(frame_height=24.0).icon_hidden())

    def test_launch_check_alerts_once_when_hidden(self):
        import scrivo
        shutdown = MagicMock()
        shutdown.requested.return_value = False
        status = self.status(frame_height=0.0)
        logs = []
        with patch.object(scrivo, "log", logs.append):
            scrivo.check_menu_bar_icon(status, shutdown)
        status.show_info.assert_called_once_with(scrivo.ICON_HIDDEN_TITLE, scrivo.ICON_HIDDEN_MESSAGE)
        self.assertEqual(len(logs), 1)
        self.assertIn("menu bar icon is hidden", logs[0])

    def test_launch_check_is_silent_when_visible_or_stopping(self):
        import scrivo
        shutdown = MagicMock()
        shutdown.requested.return_value = False
        visible = self.status(frame_height=24.0)
        scrivo.check_menu_bar_icon(visible, shutdown)
        visible.show_info.assert_not_called()
        shutdown.requested.return_value = True
        hidden = self.status(frame_height=0.0)
        scrivo.check_menu_bar_icon(hidden, shutdown)
        hidden.show_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
