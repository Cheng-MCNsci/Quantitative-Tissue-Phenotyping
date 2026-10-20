"""Record package versions in the active DeepCell kernel without running inference.

Run inside your existing UVA DeepCell environment. Does not read credentials,
list environment variables, download models, or record personal workspace paths.
"""
import importlib.metadata
import json
import platform
from pathlib import Path


def capture_environment():
    packages = sorted(
        ({'name': dist.metadata.get('Name', 'unknown'), 'version': dist.version}
         for dist in importlib.metadata.distributions()),
        key=lambda item: item['name'].lower(),
    )
    record = {'python': platform.python_version(), 'platform': platform.system(),
              'packages': packages, 'inference_run': False}
    try:
        import tensorflow as tf
        build = tf.sysconfig.get_build_info()
        record['tensorflow'] = {
            'version': tf.__version__,
            'visible_gpu_count': len(tf.config.list_physical_devices('GPU')),
            'cuda_version': build.get('cuda_version'),
            'cudnn_version': build.get('cudnn_version'),
            'is_cuda_build': build.get('is_cuda_build'),
        }
    except ImportError:
        record['tensorflow'] = {'available': False}
    return record


if __name__ == '__main__':
    path = Path('deepcell_environment_capture.json')
    with path.open('x') as handle:
        json.dump(capture_environment(), handle, indent=2, default=str)
        handle.write('\n')
    print('Saved package versions and GPU visibility to', path.name)
