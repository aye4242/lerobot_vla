#!/usr/bin/env python

"""Generic VLABench viewer entry point for LeRobot policies.

The implementation lives in ``run_smolvla_viewer`` for backward compatibility
with existing commands. New experiments should invoke this model-independent
entry point and select the checkpoint with ``--policy-path``.
"""

from run_smolvla_viewer import main


if __name__ == "__main__":
    main()
