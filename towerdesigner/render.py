"""Render PNG pictures of an exported design (run as its own process so OpenGL gets a clean start).

python -m towerdesigner.render <export folder> tower|jig <out.png>
"""
import os
import subprocess
import sys

import numpy as np


LIGHT = np.array([0.4, -0.5, 0.75])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def lit_mesh(path: str, rgb, alpha=1.0):
    """Mesh item with simple two-sided Lambert shading baked into the face colours (always bright)."""
    import cadquery as cq
    import pyqtgraph.opengl as gl
    shp = cq.importers.importStep(path).val()
    verts, tris = shp.tessellate(0.3, 0.3)
    V = np.array([[p.x, p.y, p.z] for p in verts])
    F = np.array(tris)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
    k = 0.45 + 0.55 * np.abs(n @ LIGHT)
    cols = np.c_[np.outer(k, rgb), np.full(len(F), alpha)]
    return gl.GLMeshItem(vertexes=V, faces=F, faceColors=np.clip(cols, 0, 1), smooth=False, shader=None,
                         glOptions="opaque" if alpha >= 1 else "translucent")


def render(folder: str, what: str, out: str):
    import cadquery as cq
    import pyqtgraph.opengl as gl
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    v = gl.GLViewWidget()
    v.resize(1000, 1100)
    v.setBackgroundColor((255, 255, 255))

    def add(path, color):
        v.addItem(lit_mesh(path, color[:3]))

    add(os.path.join(folder, "tower.step"), (0.86, 0.62, 0.36, 1))
    if what == "jig":
        add(os.path.join(folder, "jig", "jig_assembly.step"), (0.3, 0.55, 0.95, 1))
        v.setCameraPosition(distance=330, elevation=18, azimuth=35)
    else:
        md = gl.MeshData.cylinder(rows=2, cols=48, radius=[12.5, 12.5], length=290)
        pole = gl.GLMeshItem(meshdata=md, smooth=True, color=(0.5, 0.55, 0.65, 0.4), shader="balloon",
                             glOptions="translucent")
        pole.translate(0, 0, -20)
        v.addItem(pole)
        v.setCameraPosition(distance=330, elevation=14, azimuth=35)
    v.opts["center"] = QtGui.QVector3D(0, 0, 105)
    v.show()
    QtCore.QTimer.singleShot(1500, lambda: (_save_cropped(v.grabFramebuffer(), out), app.quit()))
    app.exec()


def render_spec(spec_path: str, out: str):
    """Render a JSON scene: {"items": [{"file", "rgb"}], "center", "distance", "elevation", "azimuth"}."""
    import json
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph.opengl as gl

    with open(spec_path) as f:
        sc = json.load(f)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    v = gl.GLViewWidget()
    v.resize(*sc.get("size", (900, 1000)))
    v.setBackgroundColor((255, 255, 255))
    for it in sc["items"]:
        if it["file"] and os.path.exists(it["file"]):
            v.addItem(lit_mesh(it["file"], it["rgb"]))
    v.setCameraPosition(distance=sc.get("distance", 340), elevation=sc.get("elevation", 20),
                        azimuth=sc.get("azimuth", 35))
    v.opts["center"] = QtGui.QVector3D(*sc.get("center", (0, 0, 100)))
    v.show()
    QtCore.QTimer.singleShot(1500, lambda: (_save_cropped(v.grabFramebuffer(), out), app.quit()))
    app.exec()


def _save_cropped(img, out):
    """Trim the white margin round the picture."""
    from PySide6 import QtGui
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), np.uint8).reshape(h, img.bytesPerLine() // 4, 4)[:, :w, :3]
    mask = (arr < 245).any(axis=2)
    ys, xs = np.nonzero(mask)
    if len(xs):
        pad = 12
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, w - 1)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, h - 1)
        img = img.copy(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    img.save(out)


def _cmd(*args) -> list[str]:
    """Run this module in a fresh process (also works inside the packaged app)."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--render", *args]
    return [sys.executable, "-m", "towerdesigner.render", *args]


def render_spec_subprocess(scene: dict, out: str) -> str | None:
    import json
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(scene, f)
        path = f.name
    try:
        subprocess.run(_cmd("scene", path, out),
                       check=True, capture_output=True, timeout=180)
        return out if os.path.exists(out) else None
    except (subprocess.SubprocessError, OSError):
        return None
    finally:
        os.unlink(path)


def render_subprocess(folder: str, what: str, out: str) -> str | None:
    try:
        subprocess.run(_cmd(folder, what, out),
                       check=True, capture_output=True, timeout=180)
        return out if os.path.exists(out) else None
    except (subprocess.SubprocessError, OSError):
        return None


def run(argv: list[str]):
    if argv[0] == "scene":
        render_spec(argv[1], argv[2])
    else:
        render(argv[0], argv[1], argv[2])


if __name__ == "__main__":
    run(sys.argv[1:])
