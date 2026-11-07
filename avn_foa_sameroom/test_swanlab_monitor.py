"""Pure stdlib tests: no SDK, network, credential or training dependency."""
import json
from pathlib import Path
import tempfile
import unittest
from .swanlab_monitor import complete_history, epoch_metrics


class HistoryTests(unittest.TestCase):
    def test_incomplete_final_write_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.jsonl'
            path.write_text('{"epoch":1}\n{"epoch":2')
            self.assertEqual([r['epoch'] for r in complete_history(path)], [1])
            with path.open('a') as stream:
                stream.write('}\n')
            self.assertEqual([r['epoch'] for r in complete_history(path)], [1, 2])
    def test_duplicate_is_deduplicated_and_conflict_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.jsonl'
            path.write_text('{"epoch":1}\n{"epoch":1}\n')
            self.assertEqual(len(complete_history(path)), 1)
            path.write_text('{"epoch":1}\n{"epoch":1,"loss":2}\n')
            with self.assertRaises(ValueError):
                complete_history(path)
    def test_missing_epoch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.jsonl'
            path.write_text('{"epoch":1}\n{"epoch":3}\n')
            with self.assertRaises(ValueError):
                complete_history(path)
    def test_mapping_keeps_original_epoch_and_metric_values(self):
        row = dict(epoch=7, global_step=1260, train_loss=.1, learning_rates=[1e-5, 1e-4],
                   epoch_wall_s=140., elapsed_s=980., validation=dict(ER=.4,F20=.7,LE=15.,LR=.8,SELD=.3,loss=.08))
        mapped = epoch_metrics(row)
        self.assertEqual(mapped['training/epoch'], 7)
        self.assertEqual(mapped['val/SELD'], .3)
        self.assertEqual(mapped['val/LE'], 15.)
        row['validation']['SELD'] = float('nan')
        with self.assertRaises(ValueError):
            epoch_metrics(row)


if __name__ == '__main__':
    unittest.main()
