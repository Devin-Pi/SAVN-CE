"""Compare current single-source masked loss against the archived ADPIT class.

Parse only the class definition: do not import the old training entrypoint or
its logging/configuration services. This check reads no training audio.
"""
import argparse
import ast
from pathlib import Path

import torch
from torch import nn
from .common import sha, write_json
from .loader import masked_adpit_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = ast.parse(args.source.read_text())
    node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'MSELoss_ADPIT')
    namespace = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(args.source), 'exec'), namespace)
    original = namespace['MSELoss_ADPIT']()
    generator = torch.Generator().manual_seed(7)
    cases = []
    for classes in (13, 21):
        for seed in range(6):
            prediction = torch.randn(2, 10, 9 * classes, generator=generator, requires_grad=True)
            target = torch.zeros(2, 10, 6, 4, classes)
            for batch in range(2):
                for frame in range(10):
                    if (batch + frame + seed) % 3:
                        category = (frame + seed) % classes
                        direction = torch.randn(3, generator=generator)
                        direction /= direction.norm()
                        target[batch, frame, 0, 0, category] = 1
                        target[batch, frame, 0, 1:, category] = direction
            mask = torch.rand(2, 10, generator=generator) > .2
            current = masked_adpit_loss(prediction, target, mask)
            reference = original(prediction[mask].unsqueeze(0), target[mask].unsqueeze(0))
            torch.testing.assert_close(current, reference)
            a, = torch.autograd.grad(current, prediction, retain_graph=True)
            b, = torch.autograd.grad(reference, prediction)
            torch.testing.assert_close(a, b)
            cases.append(dict(classes=classes, case=seed, value_error=abs(float(current-reference)), gradient_error=float((a-b).abs().max())))
    write_json(args.output, dict(status='passed', original_source=str(args.source), original_sha256=sha(args.source), cases=cases))
    print('ARCHIVED_ADPIT_VALUE_AND_GRADIENT_EQUIVALENCE_PASSED', len(cases))


if __name__ == '__main__':
    main()
