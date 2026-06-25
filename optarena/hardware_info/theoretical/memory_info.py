import os
import subprocess


def check_sudo_privileges():
    """Check if the current user has sudo privileges by running a simple sudo command."""
    try:
        subprocess.run(['sudo', '-v'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


def _resolve_dump_file(dump_file=None):
    """Find a pre-generated dmidecode dump.

    Priority: explicit argument > OPTARENA_DMIDECODE_DUMP env var > the
    well-known cache location ``.optarena_cache/dmidecode.dump`` relative
    to the repo root. Returns the path string if it exists and is
    readable, else None.
    """
    if dump_file is None:
        dump_file = os.environ.get("OPTARENA_DMIDECODE_DUMP")
    if dump_file is None:
        # Default cache location (consistent with the HPL + SuiteSparse
        # cache the rest of optarena uses).
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / ".optarena_cache" / "dmidecode.dump"
        if candidate.exists():
            dump_file = str(candidate)
    if dump_file and os.path.isfile(dump_file) and os.access(dump_file, os.R_OK):
        return dump_file
    return None


def _run_dmidecode(dump_file=None):
    """Invoke ``dmidecode --type 17``.

    If a dump file is available (explicit, env, or default cache), use
    ``--from-dump`` which does NOT require sudo. Otherwise fall back to
    ``sudo dmidecode``, requiring sudo to be configured for the user.
    Returns the captured stdout text, or None on failure.

    Generate the dump file once with::

        sudo dmidecode --dump-bin /path/to/dmidecode.dump
        # or to use the optarena cache default:
        sudo dmidecode --dump-bin .optarena_cache/dmidecode.dump

    Then set ``OPTARENA_DMIDECODE_DUMP=/path/to/dmidecode.dump`` or place
    the file in the cache to make subsequent calls sudo-free.
    """
    dump = _resolve_dump_file(dump_file)
    if dump:
        proc = subprocess.run(
            ['dmidecode', '--from-dump', dump, '--type', '17'],
            capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout
        print(f"dmidecode --from-dump {dump} failed (rc={proc.returncode}); "
              f"falling back to live sudo path.")
    if not check_sudo_privileges():
        print("Cannot get memory information. The memory info is based on "
              "the dmidecode command and thus needs either sudo privileges "
              "or a pre-generated dump file (see _run_dmidecode docstring).")
        return None
    proc = subprocess.run(['sudo', 'dmidecode', '--type', '17'],
                          capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def get_memory_info(dump_file=None):
    """Parse ``dmidecode --type 17`` output into one dict per DIMM.

    :param dump_file: Optional path to a dmidecode binary dump file. When
        present, the parse runs from the dump and no sudo is required.
        See :func:`_run_dmidecode` for how to generate one.
    """
    stdout = _run_dmidecode(dump_file=dump_file)
    if not stdout:
        return None
    blocks = stdout.strip().split('\n\n')
    blocks = blocks[1:]
    devices = []
    for block in blocks:
        block_lines = block.split('\n')
        header = block_lines[0:2]
        block_lines = block_lines[2:]

        handle, dmi_type, structure_size = list(map(lambda x: x.strip(), header[0].split(',')))
        type_name = header[1].strip()

        block_dict = {'Handle': handle, 'Type number': dmi_type, 'Type name': type_name, 'Structure size': structure_size}
        for block_line in block_lines:
            key, value = block_line.split(":", 1)
            block_dict[key.strip()] = value.strip()
        devices.append(block_dict)
    return devices


def get_theoretical_bandwidth(dump_file=None):
    """returns the theoretical bandwidth in MB/s.

    Accepts the same ``dump_file`` argument as :func:`get_memory_info`
    so the caller can side-step the sudo gate.
    """
    mem_info = get_memory_info(dump_file=dump_file)
    if not mem_info:
        return 0
    used_channels = set()
    speed_mt = 0
    width = 0
    for device in mem_info:
        if device.get("Size", "No Module Installed") != "No Module Installed":
            used_channels.add(device.get("Bank Locator", "Unknown Channel"))
            device_speed = int(device.get("Configured Memory Speed", "0").split()[0])
            if speed_mt == 0:
                speed_mt = device_speed
            elif speed_mt != device_speed:
                raise NotImplementedError("The function for calculating theoretical bandwidth has not been designed to support multiple different memory speeds")

            device_width = int(device.get("Total Width", "0").split()[0])/8
            if width == 0:
                width = device_width
            elif width != device_width:
                raise NotImplementedError("The function for calculating theoretical bandwidth has not been designed to support multiple different memory widths")
    return speed_mt * width * len(used_channels)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Print theoretical memory "
                                  "bandwidth, optionally from a dump.")
    ap.add_argument("--dump", help="Path to a pre-generated dmidecode "
                                    "binary dump (sudo-free parse).")
    a = ap.parse_args()
    print(get_theoretical_bandwidth(dump_file=a.dump), "MB/s")