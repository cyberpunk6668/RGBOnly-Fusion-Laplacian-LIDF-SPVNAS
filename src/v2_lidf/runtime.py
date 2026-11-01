"""Portable runtime constraints; no historical experiment paths are required."""
import hashlib
import json
from pathlib import Path

DISTRIBUTED_PROTOCOL = {
    'strategy': 'single_gradient_bucket_after_backward', 'torch_version': '2.1.2',
    'bucket_cap_mb': 512, 'first_bucket_cap_mb': 512, 'find_unused_parameters': False,
    'broadcast_buffers': True, 'gradient_as_bucket_view': False, 'static_graph': False,
}
TORCHSPARSE_COMMIT = '385f5ce8718fcae93540511b7f5832f4e71fd835'


def validate_distributed_config(config):
    if config.get('distributed') != DISTRIBUTED_PROTOCOL:
        raise ValueError('Training requires the pinned PyTorch 2.1.2 single gradient bucket protocol')


def validate_runtime(require_sparse=True):
    """Inspect versions and the installed sparse source, without launching CUDA."""
    import torch
    import torchvision
    import torch_scatter
    expected = [('torch', torch.__version__, '2.1.2+cu121'),
                ('torchvision', torchvision.__version__, '0.16.2+cu121'),
                ('torch_scatter', torch_scatter.__version__, '2.1.2+pt21cu121')]
    for name, actual, wanted in expected:
        if str(actual) != wanted:
            raise RuntimeError('%s=%s; this release requires %s' % (name, actual, wanted))
    result = {name: str(actual) for name, actual, _ in expected}
    if require_sparse:
        import importlib.util
        import importlib.metadata
        specification = importlib.util.find_spec('torchsparse')
        if specification is None or not specification.origin:
            raise RuntimeError('Pinned TorchSparse installation is missing')
        # Importing TorchSparse calls cuda.get_device_capability in this commit.
        # Source/ABI inspection must remain usable before selecting any GPU.
        package = Path(specification.origin).resolve().parent
        expected_sources = json.loads((Path(__file__).with_name('assets') / 'torchsparse_python_sources.json').read_text())
        for relative, digest in expected_sources['files'].items():
            path = package / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError('TorchSparse source differs from pinned commit: ' + relative)
        result['torchsparse_commit'] = expected_sources['commit']
        result['torchsparse_version'] = importlib.metadata.version('torchsparse')
    return result
