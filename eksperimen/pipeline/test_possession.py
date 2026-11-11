import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession import Tracking, classify, read_atd


FPS = 25
PITCH = (105.3, 68.0)


def tracking(n=140):
    return Tracking(np.tile([30.0, 30.0], (n, 1)), np.tile([0.8, 5.0], (n, 1)))


def event(code, kind, start, end):
    return {"code": code, "tags": {"Type": kind}, "start_frame": start, "end_frame": end}


class PossessionTests(unittest.TestCase):
    def test_missing_ball_is_not_dead(self):
        data = tracking(250)
        data.ball[20:160] = np.nan
        labels = classify(data, PITCH, FPS)
        self.assertTrue(np.all(labels[20:120] == "H"))
        self.assertTrue(np.all(labels[120:160] == "X"))
        self.assertEqual(labels[160], "H")
        self.assertNotIn("D", labels)

    def test_brief_opponent_touch_and_loose_ball_do_not_split_spell(self):
        data = tracking()
        data.distance[20:24] = [5.0, 0.6]
        data.distance[24:55] = [3.0, 3.2]
        self.assertTrue(np.all(classify(data, PITCH, FPS) == "H"))

    def test_confirmed_turnover_is_backdated_to_first_support(self):
        data = tracking()
        data.distance[40:] = [5.0, 0.8]
        labels = classify(data, PITCH, FPS)
        self.assertTrue(np.all(labels[:40] == "H"))
        self.assertTrue(np.all(labels[40:] == "A"))

    def test_crossing_stops_play_and_visible_throw_in_restarts_it(self):
        data = tracking()
        data.ball[21:29, 1] = [2, 1.5, 1.0, 0.5, -0.1, -0.5, -0.9, -1.3]
        data.ball[29:45] = np.nan
        data.ball[45:70, 1] = 0
        data.ball[70:84, 1] = np.arange(1, 15) * 0.25
        data.ball[84:, 1] = 4
        data.distance[60:] = [6.0, 1.0]
        labels = classify(data, PITCH, FPS, [event("SET PIECES", "Throw-in", 61, 78)])
        self.assertTrue(np.all(labels[:26] == "H"))
        self.assertTrue(np.all(labels[26:70] == "D"))
        self.assertTrue(np.all(labels[70:] == "A"))

    def test_goal_requires_crossing_and_center_restart(self):
        data = tracking(180)
        data.ball[:25] = [103.0, 34.0]
        data.ball[25:31, 0] = [103.8, 104.6, 105.4, 106.2, 107.0, 107.8]
        data.ball[25:31, 1] = 34
        data.ball[31:40] = np.nan
        # A visible decoy after the goal would otherwise release the crossing.
        data.ball[40:55, 0] = 95 + np.arange(15) * 0.3
        data.ball[40:55, 1] = 33
        data.ball[55:90] = np.nan
        data.ball[90:120] = [52.65, 34]
        data.ball[120:145, 0] = 52.65 + np.arange(1, 26) * 0.25
        data.ball[120:, 1] = 34
        data.ball[145:, 0] = 59
        data.distance[85:] = [5.0, 0.8]
        kickoff = event("SET PIECES", "Kick Off (After Goal)", 110, 134)
        labels = classify(data, PITCH, FPS, [event("GOALS", "", 21, 40), kickoff])
        self.assertTrue(np.all(labels[28:120] == "D"))
        self.assertTrue(np.all(labels[120:] == "A"))
        without_goal = classify(data, PITCH, FPS, [kickoff])
        self.assertEqual(without_goal[50], "H")

    def test_tag_with_no_support_does_not_insert_dead_ball(self):
        data = tracking()
        tags = [event("GOALS", "", 10, 25),
                event("SET PIECES", "Kick Off (After Goal)", 50, 75)]
        self.assertTrue(np.all(classify(data, PITCH, FPS, tags) == "H"))

    def test_noisy_single_outside_sample_does_not_insert_dead_ball(self):
        data = tracking()
        data.ball[:, 1] = 0.5
        data.ball[30, 1] = -0.4
        self.assertTrue(np.all(classify(data, PITCH, FPS) == "H"))

    def test_atd_frames_are_zero_based_and_bad_ids_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.txt"
            tracks = "0.20,0.20,0;0.30,0.30,0"
            path.write_text(f"1:{tracks}:0.20,0.20,0\n2:{tracks}:NaN,NaN,NaN\n")
            data = read_atd(path, np.array([0, 1]), 2, PITCH)
            self.assertAlmostEqual(data.distance[0, 0], 0)
            self.assertTrue(np.isnan(data.ball[1]).all())
            path.write_text(f"2:{tracks}:0.20,0.20,0\n")
            with self.assertRaisesRegex(ValueError, "consecutive"):
                read_atd(path, np.array([0, 1]), 1, PITCH)


if __name__ == "__main__":
    unittest.main()
