"""fiducial_decode must stay import-light.

It exists precisely so a caller that only wants the counter/ring readers -- a ROS node,
say -- does not pay for matplotlib, tqdm and (transitively, through ``rocsync.timeline``)
scikit-learn. Run in a subprocess: within the same interpreter as the rest of the suite,
an earlier test may already have imported these for its own reasons, which would make the
check pass for the wrong reason.
"""

import subprocess
import sys


def test_fiducial_decode_imports_without_the_plotting_stack():
    script = (
        "import sys; import rocsync.fiducial_decode; "
        "leaked = [m for m in ('matplotlib', 'tqdm', 'sklearn') if m in sys.modules]; "
        "assert not leaked, leaked"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
