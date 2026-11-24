import unittest

from fuhe.list_episodes import row_from_meta


def meta(**overrides):
    data = {
        "episode_id": "ep",
        "recording_status": "complete",
        "source_group": {"name": "复核包装发货"},
        "timing": {"duration_s": 10, "start_timestamp_ns": 5},
        "sensors": {"cameras": [{"topic": "/camera/stand/rgb", "fps": 15, "width": 960, "height": 720}]},
        "devices": [{"mount": "head"}],
        "_prefix": "oss://quic-collect-data/prod/raw/v2/sources/x/",
    }
    data.update(overrides)
    return data


class ListEpisodesTest(unittest.TestCase):
    def test_keeps_a_complete_fuhe_episode(self):
        row = row_from_meta(meta())
        self.assertEqual(row["episode_id"], "ep")
        self.assertEqual(row["topic"], "/camera/stand/rgb")
        self.assertEqual(row["oss_prefix"], "oss://quic-collect-data/prod/raw/v2/sources/x/")
        self.assertEqual(row["epoch_ns"], 5)
        self.assertEqual(row["duration"], 10.0)

    def test_alias_names_stay_and_other_tasks_drop(self):
        self.assertEqual(row_from_meta(meta(source_group={"name": "复合包装发货"}))["task"], "复合包装发货")
        self.assertIsNone(row_from_meta(meta(source_group={"name": "供包"})))
        self.assertIsNone(row_from_meta(meta(recording_status="incomplete")))

    def test_topic_falls_back_to_the_mount(self):
        row = row_from_meta(meta(sensors={"cameras": []}, devices=[{"mount": "stand"}]))
        self.assertEqual(row["topic"], "/camera/stand/rgb")


if __name__ == "__main__":
    unittest.main()
