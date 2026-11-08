"""Retry an independent SwanLab follower without ever signaling a trainer."""
import argparse
import fcntl
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--collector',required=True)
    p.add_argument('--run-dir',required=True);p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--training-pid',required=True);p.add_argument('--label',required=True);p.add_argument('--source',required=True)
    a=p.parse_args();a.state_dir.parent.mkdir(exist_ok=True,parents=True)
    lock=(a.state_dir.parent/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None
    def stop(signum,frame):
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    cmd=[a.python,'-u',a.collector,'--run-dir',a.run_dir,'--state-dir',str(a.state_dir),
        '--training-pid',a.training_pid,'--experiment-name',f'FOA21-short-{a.label}-current250ms-full100-seed7',
        '--id-prefix',f'avn-short-{a.label}-v1-','--source-file',a.source]
    try:
        while True:
            child=subprocess.Popen(cmd)
            outcome=child.wait()
            if outcome in (0,2):return
            print('Collector retry in30s; training untouched',flush=True);time.sleep(30)
    except KeyboardInterrupt:
        if child is not None:child.wait(timeout=30)


if __name__=='__main__':main()
