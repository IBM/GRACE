# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("rdkit.Chem.Draw", MagicMock())
sys.modules.setdefault("rdkit.Chem.Draw.rdMolDraw2D", MagicMock())
