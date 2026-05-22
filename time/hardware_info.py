"""Capture hardware and library version metadata for the timing benchmark."""
import json
import platform
import sys
from pathlib import Path


def capture_hardware() -> dict:
    """Return a dict with CPU/RAM/OS/Python/library version metadata."""
    info: dict = {
        'platform': platform.platform(),
        'os': platform.system(),
        'os_release': platform.release(),
        'python_version': sys.version.split()[0],
        'processor': platform.processor() or platform.machine(),
    }

    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info()
        info['cpu_brand'] = cpu.get('brand_raw', '')
        info['cpu_hz_advertised'] = cpu.get('hz_advertised_friendly', '')
        info['cpu_count_logical'] = cpu.get('count', 0)
    except ImportError:
        info['cpu_brand'] = ''
        info['cpu_hz_advertised'] = ''
        info['cpu_count_logical'] = 0

    try:
        import psutil
        info['ram_total_gb'] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        info['cpu_count_physical'] = psutil.cpu_count(logical=False) or 0
    except ImportError:
        info['ram_total_gb'] = 0.0
        info['cpu_count_physical'] = 0

    # Library versions
    libs: dict = {}
    for name in ('numpy', 'scipy', 'nibabel', 'nilearn', 'sklearn', 'tensorflow', 'pandas', 'skimage', 'SimpleITK'):
        try:
            module = __import__(name)
            libs[name] = getattr(module, '__version__', 'unknown')
        except ImportError:
            libs[name] = 'not installed'
    info['library_versions'] = libs

    return info


def write_hardware_json(output_path: Path) -> None:
    """Capture hardware metadata and write to JSON file."""
    info = capture_hardware()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    out = Path(__file__).parent / 'outputs' / 'timing_hardware.json'
    write_hardware_json(out)
    print(f'Wrote {out}')
