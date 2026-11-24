import unittest

from fuhe.chain import gpu_fits
from fuhe.kept_flora import choose_segment, names_from_ls, parse_oss_json, rows_from_listing


class KeptFloraTest(unittest.TestCase):
    def test_parse_ignores_ossutil_trailer(self):
        text = '{"layout": "segments"}\n0.08(s) elapsed\n'
        self.assertEqual(parse_oss_json(text)["layout"], "segments")

    def test_longest_unannotated_segment_goes_first(self):
        rows = [
            {"episode_id": "b", "seg": 0, "duration_s": 30, "frames": 100, "annotated": False, "failed": False},
            {"episode_id": "a", "seg": 1, "duration_s": 2, "frames": 24, "annotated": False, "failed": False},
            {"episode_id": "a", "seg": 0, "duration_s": 2, "frames": 1, "annotated": False, "failed": False},
            {"episode_id": "c", "seg": 0, "duration_s": 1, "frames": 10, "annotated": True, "failed": False},
        ]
        picked = choose_segment(rows)
        self.assertEqual((picked["episode_id"], picked["seg"]), ("b", 0))

    def test_listing_marks_annotated_without_stat(self):
        text = "\n".join([
            "LastModifiedTime                   Size(B)     StorageClass   ETAG                                  ObjectName",
            "2026-11-23 08:52:13 +0800 CST          449         Standard   89989ED869104EBC5119518A7CC2F130      oss://quic-research/wbx/data/fuhe_kept/ep/seg_005.json",
            "2026-11-23 06:35:34 +0800 CST      2965424         Standard   52BA2A6B6F712BF03FFC8190805A9713      oss://quic-research/wbx/data/fuhe_kept/ep/seg_006.mp4",
            "2026-11-23 06:35:34 +0800 CST          40         Standard   52BA2A6B6F712BF03FFC8190805A9714      oss://quic-research/wbx/data/fuhe_kept/ep/seg_006.flora_error.json",
            "oss://quic-research/wbx/data/fuhe_kept/ep/",
            "NextMarker: foo",
        ])
        names = names_from_ls(text)
        self.assertEqual(names, {"seg_005.json", "seg_006.mp4", "seg_006.flora_error.json"})
        spans = {"fps": 15, "segments": [
            {"seg": 5, "file": "seg_005.mp4", "src_start": 0, "src_end": 40, "frames": 600, "duration_s": 40},
            {"seg": 6, "file": "seg_006.mp4", "src_start": 40, "src_end": 41, "frames": 15, "duration_s": 1},
        ]}
        rows = rows_from_listing("ep", "oss://quic-research/wbx/data/fuhe_kept/ep", spans, names)
        by_seg = {row["seg"]: row for row in rows}
        self.assertTrue(by_seg[5]["annotated"])
        self.assertFalse(by_seg[5]["failed"])
        self.assertFalse(by_seg[6]["annotated"])
        self.assertTrue(by_seg[6]["failed"])
        self.assertFalse(by_seg[5]["rotated_180"])

    def test_one_h20_is_enough_and_three_small_cards_still_are(self):
        self.assertTrue(gpu_fits(1, 96 * 1024 ** 3))
        self.assertTrue(gpu_fits(3, 24 * 1024 ** 3))
        self.assertFalse(gpu_fits(1, 24 * 1024 ** 3))
        self.assertFalse(gpu_fits(2, 24 * 1024 ** 3))


if __name__ == "__main__":
    unittest.main()
