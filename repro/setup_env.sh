#!/bin/bash
# Moi truong Python 3.10 cho ImmortalTracker goc (khong sua code repo).
set -e
python3 -m venv ${ENV:-$HOME/imm_env}
. ${ENV:-$HOME/imm_env}/bin/activate
pip install "numpy<1.24" pyyaml tqdm "shapely<2" filterpy "numba<0.59" "pandas<2" pyquaternion cachetools "motmetrics<=1.1.3" descartes scikit-learn "opencv-python-headless<4.9" "matplotlib<3.6"
pip install --no-deps nuscenes-devkit==1.1.11
SP=$(python -c "import site;print(site.getsitepackages()[0])")
# motmetrics 1.1.3 + Python 3.10: collections.Iterable -> collections.abc
sed -i 's/from collections import OrderedDict, Iterable/from collections import OrderedDict\nfrom collections.abc import Iterable/' $SP/motmetrics/metrics.py
# PyYAML>=6: code goc goi yaml.load(f) khong co Loader -> shim o muc moi truong
printf 'import yaml as _y\n_o=_y.load\ndef _l(s, Loader=None):\n    return _o(s, Loader=Loader or _y.FullLoader)\n_y.load=_l\n' > $SP/_yamlshim.py
echo "import _yamlshim" > $SP/zz_yamlshim.pth
python -c "from nuscenes.eval.tracking.evaluate import TrackingEval; print('env ok')"
