import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'agent_episode'))
from sound_schedule import resolve_sound_schedule


class SoundScheduleTests(unittest.TestCase):
    def test_official_schedule_is_unchanged_by_default(self):
        s=resolve_sound_schedule('episode',2,8,.25,64)
        self.assertEqual((s['runtime_onset_s'],s['runtime_offset_s']),(2,8))
        self.assertFalse(s['episode_modified'])
    def test_sustained_preserves_onset_and_covers_final_audio_block(self):
        s=resolve_sound_schedule('sustained',2,8,.25,64)
        self.assertEqual(s['runtime_onset_s'],2)
        self.assertEqual(s['runtime_offset_s'],18)
        self.assertGreater(s['runtime_offset_s'],65*.25)
    def test_longer_official_sound_is_not_shortened(self):
        self.assertEqual(resolve_sound_schedule('sustained',2,50,.25,64)['runtime_offset_s'],50)
    def test_continuous_covers_reset_and_full_benchmark_budget(self):
        s=resolve_sound_schedule('continuous',4,22,.25,500)
        self.assertEqual((s['official_onset_s'],s['official_offset_s']),(4,22))
        self.assertEqual(s['runtime_onset_s'],0)
        self.assertGreaterEqual(s['runtime_offset_s'],501*.25)
        self.assertFalse(s['episode_modified'])
    def test_invalid_inputs_are_rejected(self):
        for args in [('bad',2,8,.25,64),('episode',2,1,.25,64),('episode',2,8,0,64),('episode',2,8,.25,0)]:
            with self.assertRaises(ValueError):resolve_sound_schedule(*args)


if __name__=='__main__':unittest.main()
