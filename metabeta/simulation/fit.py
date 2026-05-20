import argparse
import sys

from metabeta.utils.templates import setupConfigParser, generateSimulationConfig


# fmt: off
def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data (template-based)
    parser.add_argument('--size', type=str, default='small')
    parser.add_argument('--family', type=int, default=0)
    parser.add_argument('--ds_type', type=str, default='sampled')
    parser.add_argument('--config', type=str)

    # common
    parser.add_argument('--method', type=str, default='nuts', help='nuts|advi|inla')
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--reintegrate', action='store_true')
    parser.add_argument('--partition', type=str, default='test', choices=['train', 'test', 'valid'])
    parser.add_argument('--epoch', type=int, default=None, help='Epoch number (required when --partition train)')

    # PyMC args
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tune', type=int, default=2000)
    parser.add_argument('--target_accept', type=float, default=0.8)
    parser.add_argument('--max_treedepth', type=int, default=10)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--mp_ctx', type=str, default='forkserver')
    parser.add_argument('--viter', type=int, default=100_000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--diagonal', action='store_true')

    # INLA args
    parser.add_argument('--n', type=int, default=None, help='Number of datasets to fit/reintegrate (INLA only; default: full batch)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing per-index fit files (INLA only)')
    parser.add_argument('--re-correlation', dest='re_correlation', default='diagonal', choices=['auto', 'diagonal'])
    parser.add_argument('--timeout', dest='timeout_s', type=int, default=120)

    return setupConfigParser(parser, generateSimulationConfig, 'Fit hierarchical datasets (NUTS, ADVI, or INLA).')
# fmt: on


if __name__ == '__main__':
    cfg = setup()
    for _k, _v in [
        ('method', 'nuts'), ('idx', 0), ('reintegrate', False), ('partition', 'test'),
        ('epoch', None), ('seed', 42), ('tune', 2000), ('target_accept', 0.8),
        ('max_treedepth', 10), ('draws', 1000), ('chains', 4), ('loop', False),
        ('mp_ctx', 'forkserver'), ('viter', 100_000), ('lr', 1e-3), ('diagonal', False),
        ('re_correlation', 'diagonal'), ('timeout_s', 120), ('n', None), ('force', False),
    ]:
        if not hasattr(cfg, _k):
            setattr(cfg, _k, _v)

    if cfg.partition == 'train' and cfg.epoch is None:
        print('error: --epoch is required when --partition train', file=sys.stderr)
        sys.exit(1)

    if cfg.method == 'inla':
        from metabeta.simulation.inla import InlaFitter
        fitter = InlaFitter(cfg)
        if cfg.reintegrate:
            fitter.reintegrate()
        else:
            fitter.go()
    else:
        from metabeta.simulation.pymc import Fitter
        fitter = Fitter(cfg)
        if cfg.reintegrate:
            fitter.reintegrate(methods=[cfg.method])
        else:
            fitter.go()
