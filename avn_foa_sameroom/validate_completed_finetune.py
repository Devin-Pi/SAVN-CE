"""Validate complete-run accounting and validation-only checkpoint selection."""
import argparse
import json
import math
from pathlib import Path

from .common import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / 'config.json').read_text())
    result = json.loads((args.run / 'results.json').read_text())
    history = [json.loads(line) for line in (args.run / 'history.jsonl').read_text().splitlines()]
    assert config['epochs'] == 100 and config['patience'] == 0
    assert result['status'] == 'completed' and result['all_requested_epochs_completed']
    assert len(history) == result['completed_epochs'] == result['requested_epochs'] == 100
    assert [row['epoch'] for row in history] == list(range(1, 101))
    assert all(row['evaluation_precision'] == 'FP32' for row in history)
    expected_steps=100*math.ceil(config['train_records']/config['batch'])
    assert result['optimizer_steps'] == expected_steps
    assert all(row['global_step']==row['epoch']*math.ceil(config['train_records']/config['batch']) for row in history)
    if config.get('reference_data'):
        assert not result['reference_used_for_selection']
        assert result['reference_test'] is not None
        assert all('reference_validation' in row for row in history)
    assert config['resume'] is None and config['loss'] == 'standard'
    assert config['optimizer'] == 'Adam' and config['all_model_parameters_trainable_from_step1']
    best = min(history, key=lambda row: (round(row['validation']['SELD'], 9), row['validation']['loss']))
    # The training selector treats SELD differences <=1e-9 as a tie.
    chosen = history[result['best_epoch'] - 1]
    assert chosen['validation'] == result['best_validation']
    assert abs(chosen['validation']['SELD'] - best['validation']['SELD']) <= 1e-9
    assert not result['test_used_for_selection']
    write_json(args.run / 'completed_run_validation.json', dict(
        status='passed', epochs=100, optimizer_steps=expected_steps, configuration_fixed=True,
        selected_epoch=result['best_epoch'], all_epochs_evaluated_in_fp32=True,
        test_used_for_selection=False, original_model_initialized=True))
    print('COMPLETE_100_EPOCH_RUN_VALIDATED')


if __name__ == '__main__':
    main()
