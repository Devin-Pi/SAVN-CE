"""Guard complete-run accounting when the dataset size changes."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from avn_foa_sameroom.validate_completed_finetune import main


class AccountingTests(unittest.TestCase):
    def fixture(self,path,steps=27000,train=4320):
        score=dict(SELD=.2,loss=.01)
        config=dict(epochs=100,patience=0,train_records=train,batch=16,resume=None,loss='standard',optimizer='Adam',all_model_parameters_trainable_from_step1=True,reference_data='/reference')
        history=[dict(epoch=i,global_step=i*((train+15)//16),evaluation_precision='FP32',validation=score,reference_validation=dict(SELD=.99)) for i in range(1,101)]
        result=dict(status='completed',all_requested_epochs_completed=True,completed_epochs=100,requested_epochs=100,optimizer_steps=steps,reference_used_for_selection=False,reference_test=score,best_epoch=1,best_validation=score,test_used_for_selection=False)
        for name,obj in [('config',config),('results',result)]:
            (path/f'{name}.json').write_text(json.dumps(obj))
        (path/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    def test_dense27000_updates_pass(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);self.fixture(path)
            with patch('sys.argv',['check',d]):main()
            self.assertEqual(json.loads((path/'completed_run_validation.json').read_text())['optimizer_steps'],27000)
    def test_stale18000_updates_fail(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);self.fixture(path,steps=18000)
            with patch('sys.argv',['check',d]),self.assertRaises(AssertionError):main()
    def test_partial_last_batch_is_counted(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);self.fixture(path,steps=27100,train=4321)
            with patch('sys.argv',['check',d]):main()

if __name__=='__main__':unittest.main()
