import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _d = _HERE
while not _os.path.isdir(_os.path.join(_d, 'src')) and _os.path.dirname(_d) != _d: _d = _os.path.dirname(_d)
_sys.path[:0] = [_os.path.join(_d, 'src'), _HERE]
import numpy as np
from engines import LGPhysicsEngine
from config import LG_PARAMS

def eta_dir(Xt, Xq, v, ttf):
    e = LGPhysicsEngine(**LG_PARAMS); e.calibrate((Xt @ v)[:, None], ttf)
    return e.get_eta((Xq @ v)[:, None])


def unit_slope(v, units, cyc, w=20):
    """Least-squares slope over the trailing w cycles of the same unit, never across units."""
    out = np.zeros_like(v, dtype=float)
    for un in np.unique(units):
        m = np.where(units == un)[0]; o = m[np.argsort(cyc[m])]
        x = cyc[o].astype(float); y = v[o].astype(float); n = len(o)
        cx, cy = np.concatenate([[0], np.cumsum(x)]), np.concatenate([[0], np.cumsum(y)])
        cxx, cxy = np.concatenate([[0], np.cumsum(x * x)]), np.concatenate([[0], np.cumsum(x * y)])
        i = np.arange(n); a = np.maximum(0, i - w + 1); k = (i - a + 1).astype(float)
        sx, sy = cx[i + 1] - cx[a], cy[i + 1] - cy[a]; sxx, sxy = cxx[i + 1] - cxx[a], cxy[i + 1] - cxy[a]
        den = k * sxx - sx * sx
        out[o] = np.where((k >= 3) & (den > 0), (k * sxy - sx * sy) / np.where(den > 0, den, 1), 0.0)
    return out

