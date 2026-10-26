"""Tower Designer — Mac desktop app (PySide6 + pyqtgraph OpenGL)."""
from __future__ import annotations

import datetime as _dt
import os
import sys
import threading
import traceback

import numpy as np
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets

from .analysis import analyse
from .export import export_all, load_design, save_design
from .families import FAMILY_NAMES, SPACES, build
from .materials import GLUES, WOODS
from .gluetest import load_glue_factors, save_glue_factors
from .woodtest import WoodTest, apply_saved, load_wood_test, save_wood_test
from .optimiser import Candidate, Optimiser, fit_to_budget, seeds
from .spec import Spec
from .structure import Infeasible

DESIGNS_DIR = os.path.expanduser("~/Documents/Tower Designs")
REFERENCE_DESIGN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "reference_tower.json")
def util_color(u: float) -> tuple[float, float, float, float]:
    u = float(np.clip(u, 0, 1))
    if u < 0.5:
        k = u / 0.5
        return (0.2 + 0.8 * k, 0.65 + 0.15 * k, 0.3 - 0.2 * k, 1.0)
    k = (u - 0.5) / 0.5
    return (1.0 - 0.1 * k, 0.8 - 0.65 * k, 0.1, 1.0)


# ---------------------------------------------------------------------------------------------
# 3D view
# ---------------------------------------------------------------------------------------------
def tower_mesh(st, colors):
    verts, faces, fcols = [], [], []
    box_faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                          [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    for k, m in enumerate(st.members):
        p0, p1 = st.nodes[m.i].copy(), st.nodes[m.j].copy()
        off = st.parts[m.part].offset
        if off:
            mid = (p0 + p1) / 2
            r = np.array([mid[0], mid[1], 0.0])
            r /= np.linalg.norm(r) or 1
            p0, p1 = p0 + r * off, p1 + r * off
        dz = st.parts[m.part].info.get("dz", 0.0)
        if dz:
            p0, p1 = p0 + np.array([0, 0, dz]), p1 + np.array([0, 0, dz])
        x = p1 - p0
        L = np.linalg.norm(x)
        if L < 1e-9:
            continue
        x /= L
        ref = m.ref if m.ref is not None else np.array([0, 0, 1.0])
        y = ref - (ref @ x) * x
        if np.linalg.norm(y) < 1e-6:
            y = np.array([1.0, 0, 0]) - x[0] * x
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        hw, ht = m.sec.size_y / 2, m.sec.size_z / 2
        corners = []
        for p in (p0, p1):
            for sy, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corners.append(p + y * hw * sy + z * ht * sz)
        base = len(verts)
        verts.extend(corners)
        faces.extend((box_faces + base).tolist())
        fcols.extend([colors[k]] * 12)
    return np.array(verts), np.array(faces), np.array(fcols)


class TowerView(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor((245, 242, 236))
        grid = gl.GLGridItem()
        grid.setSize(300, 300)
        grid.setSpacing(10, 10)
        grid.setColor((180, 170, 160, 120))
        self.addItem(grid)
        self.items_dyn = []
        self.setCameraPosition(distance=360, elevation=16, azimuth=40)

    def show_structure(self, st, util=None, mode="util", spec: Spec | None = None):
        for it in self.items_dyn:
            self.removeItem(it)
        self.items_dyn = []
        if st is None:
            return
        if mode == "util" and util is not None:
            cols = [util_color(u) for u in util]
        else:
            cols = [(*KIND_COLORS.get(m.kind, (0.8, 0.6, 0.4)), 1.0) for m in st.members]
        v, f, c = tower_mesh(st, cols)
        mesh = gl.GLMeshItem(vertexes=v, faces=f, faceColors=c, smooth=False, shader="shaded",
                             drawEdges=False)
        self.addItem(mesh)
        self.items_dyn.append(mesh)
        if spec:
            H = st.height
            md = gl.MeshData.cylinder(rows=2, cols=40, radius=[spec.pole_d / 2] * 2, length=H + 60)
            pole = gl.GLMeshItem(meshdata=md, smooth=True, color=(0.55, 0.6, 0.68, 0.35), shader="balloon",
                                 glOptions="translucent")
            pole.translate(0, 0, -10)
            self.addItem(pole)
            self.items_dyn.append(pole)
            r = max(np.hypot(st.nodes[t][0], st.nodes[t][1]) for t in st.top_nodes) + 15
            md2 = gl.MeshData.cylinder(rows=1, cols=48, radius=[r, r], length=4)
            plate = gl.GLMeshItem(meshdata=md2, smooth=True, color=(0.35, 0.45, 0.85, 0.35), shader="balloon",
                                  glOptions="translucent")
            plate.translate(0, 0, H + 0.5)
            self.addItem(plate)
            self.items_dyn.append(plate)
        self.opts["center"] = QtGui.QVector3D(0, 0, st.height / 2)
        self.update()


# ---------------------------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------------------------
class OptWorker(QtCore.QObject):
    progress = QtCore.Signal(float, str, object)
    finished = QtCore.Signal(list, list)
    failed = QtCore.Signal(str)

    def __init__(self, spec: Spec):
        super().__init__()
        self.spec = spec
        self.stop = threading.Event()

    @QtCore.Slot()
    def run(self):
        try:
            opt = Optimiser(self.spec, progress=lambda f, m, b: self.progress.emit(f, m, b), stop=self.stop)
            res = opt.run()
            self.finished.emit(res, opt.state.log)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ExportWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, family, params, spec, outdir):
        super().__init__()
        self.args = (family, params, spec, outdir)

    @QtCore.Slot()
    def run(self):
        try:
            r = export_all(*self.args, progress=self.progress.emit)
            self.finished.emit(r)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---------------------------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------------------------
def _dspin(lo, hi, val, step=0.1, dec=2, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(" " + suffix)
    return w


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tower Designer")
        self.resize(1440, 900)
        self.spec = apply_saved(Spec(glue_factors=load_glue_factors()))
        self.family = "stick"
        self.params: dict = {}
        self.st = None
        self.res = None
        self.candidates: list[Candidate] = []
        self.thread = None
        self.worker = None
        self.last_export = None
        self._building_editor = False

        self.view = TowerView()
        self.setCentralWidget(self.view)
        self._build_left()
        self._build_right()
        self._build_toolbar()
        self.statusBar().showMessage("Ready")
        self._update_glue_label()
        self._update_wood_label()
        self.w_density.blockSignals(True)
        self.w_density.setValue(self.spec.measured_density)
        self.w_density.blockSignals(False)

        self.debounce = QtCore.QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(250)
        self.debounce.timeout.connect(self.reanalyse)

        # start with an educated guess so the window is not empty
        fam0 = self.spec.families[0]
        ss = seeds(fam0, self.spec)
        s = next((x for x in ss if x.get("n_sides") == 6 and x.get("pattern", "X") == "X"
                  and x.get("joint") == "lap"), ss[0])
        self.load_design(fam0, fit_to_budget(fam0, s, self.spec))

    # -- left: brief + optimiser -----------------------------------------------------------------
    def _build_left(self):
        dock = QtWidgets.QDockWidget("Brief and search", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        sp = self.spec

        g1 = QtWidgets.QGroupBox("Requirements")
        f1 = QtWidgets.QFormLayout(g1)
        self.w_height = _dspin(210.5, 400, sp.height, 0.5, 1, "mm")
        self.w_pole = _dspin(5, 60, sp.pole_d, 0.5, 1, "mm")
        self.w_clear = _dspin(0.2, 10, sp.clearance, 0.1, 1, "mm")
        self.w_mass = _dspin(1, 100, sp.max_mass, 0.1, 2, "g")
        self.w_margin = _dspin(0, 20, (sp.margin - 1) * 100, 0.5, 1, "%")
        f1.addRow("Tower height", self.w_height)
        f1.addRow("Pole diameter", self.w_pole)
        f1.addRow("Pole clearance (each side)", self.w_clear)
        f1.addRow("Max mass", self.w_mass)
        f1.addRow("Margin per piece/joint", self.w_margin)
        self.w_joints = QtWidgets.QSpinBox()
        self.w_joints.setRange(0, 5000)
        self.w_joints.setSingleStep(50)
        self.w_joints.setValue(sp.max_joints)
        self.w_joints.setSpecialValueText("no limit")
        self.w_joints.setToolTip("Build effort: the most glue joints you're prepared to make. More joints = more legs "
                                 "and storeys = stronger, but far more work.")
        f1.addRow("Max glue joints", self.w_joints)
        self.w_slot = _dspin(2.8, 4.0, sp.jig_slot, 0.1, 1, "mm")
        self.w_slot.setToolTip("Print fit_test_coupon.stl, find the smallest slot a stick slides into with light "
                               "friction, and enter it here. Every jig slot and notch is sized from it.")
        f1.addRow("Jig slot size", self.w_slot)
        lay.addWidget(g1)

        g2 = QtWidgets.QGroupBox("Wood and glue")
        f2 = QtWidgets.QFormLayout(g2)
        self.w_wood = QtWidgets.QComboBox()
        for k, wd in WOODS.items():
            self.w_wood.addItem(wd.name, k)
        self.w_wood.setCurrentIndex(max(0, self.w_wood.findData(sp.wood)))
        self.w_density = _dspin(0, 600, 0, 5, 0, "kg/m³")
        self.w_density.setSpecialValueText("book value")
        self.w_density.setMinimumWidth(130)
        self.w_density.setToolTip("Weigh a whole sheet: density = mass ÷ (915 × width × thickness).")
        self.w_quality = _dspin(0.4, 1.0, sp.quality, 0.05, 2)
        self.w_glue = QtWidgets.QComboBox()
        self.w_glue.addItem("Best glue for each joint (recommended)", "mixed")
        self.w_glue.addItem("Best single glue", "auto")
        for k, gl_ in GLUES.items():
            self.w_glue.addItem(f"{gl_.name} only", k)
        self.w_glue.setCurrentIndex(max(0, self.w_glue.findData(sp.glue)))
        self.w_gfactor = _dspin(0.5, 20, sp.glue_factor, 0.1, 2, "×")
        self.w_gfactor.setToolTip("How much more glue real hand-gluing leaves than the bare physics. "
                                  "4.8 matches a rule of thumb of 5 g of glue per 15 g of wood on a typical tower.")
        b_cal = QtWidgets.QPushButton("Calibrate…")
        b_cal.setToolTip("Set the factor from a tower you weighed (glue grams per wood grams).")
        b_cal.clicked.connect(self.calibrate_glue)
        gfrow = QtWidgets.QHBoxLayout()
        gfrow.addWidget(self.w_gfactor, 1)
        gfrow.addWidget(b_cal)
        b_test = QtWidgets.QPushButton("Glue test…")
        b_test.setToolTip("Glue and weigh a few standard joints per glue; sets each glue's own factor.")
        b_test.clicked.connect(self.glue_test)
        gfrow.addWidget(b_test)
        self.lbl_gf = QtWidgets.QLabel("")
        self.lbl_gf.setWordWrap(True)
        self.lbl_gf.setStyleSheet("color:#888")
        f2.addRow("Wood", self.w_wood)
        b_wood = QtWidgets.QPushButton("Wood test…")
        b_wood.setToolTip("Crush and buckle a few sample sticks to measure your wood's real strength.")
        b_wood.clicked.connect(self.wood_test)
        drow = QtWidgets.QHBoxLayout()
        drow.addWidget(self.w_density, 1)
        drow.addWidget(b_wood)
        f2.addRow("Measured density", drow)
        self.lbl_wood = QtWidgets.QLabel("")
        self.lbl_wood.setWordWrap(True)
        self.lbl_wood.setStyleSheet("color:#888")
        f2.addRow("", self.lbl_wood)
        f2.addRow("Build quality factor", self.w_quality)
        f2.addRow("Glue", self.w_glue)
        self.w_symglue = QtWidgets.QCheckBox("Same glue all the way round (load side unknown)")
        self.w_symglue.setChecked(sp.symmetric_glue)
        self.w_symglue.setToolTip("Every matching joint round the tower gets the glue its worst-loaded copy needs.")
        f2.addRow("", self.w_symglue)
        f2.addRow("Glue hand factor", gfrow)
        f2.addRow("", self.lbl_gf)
        lay.addWidget(g2)

        g3 = QtWidgets.QGroupBox("Realism (advanced)")
        f3 = QtWidgets.QFormLayout(g3)
        self.w_uneven = _dspin(0, 100, sp.uneven * 100, 5, 0, "%")
        self.w_sway = _dspin(0, 10, sp.sway, 0.25, 2, "mm")
        self.w_lateral = _dspin(0, 10, sp.lateral * 100, 0.5, 1, "%")
        self.w_imp = _dspin(0.0, 0.02, sp.imperfection, 0.001, 3)
        f3.addRow("Uneven bearing", self.w_uneven)
        f3.addRow("Built out of true by", self.w_sway)
        f3.addRow("Sideways load", self.w_lateral)
        f3.addRow("Member crookedness", self.w_imp)
        self.w_char = _dspin(0, 1, sp.char, 0.05, 2, "mm")
        self.w_char.setToolTip("Burnt wood on each laser-cut edge that carries no load.")
        f3.addRow("Laser char per cut edge", self.w_char)
        lay.addWidget(g3)

        g4 = QtWidgets.QGroupBox("Design search")
        f4 = QtWidgets.QFormLayout(g4)
        self.w_fams = {}
        fam_box = QtWidgets.QVBoxLayout()
        for k, name in FAMILY_NAMES.items():
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(k in sp.families)
            self.w_fams[k] = cb
            fam_box.addWidget(cb)
        f4.addRow("Shapes to try", fam_box)
        self.w_effort = QtWidgets.QComboBox()
        for k, lab in (("quick", "Quick (~20 s)"), ("normal", "Normal (~1–2 min)"), ("thorough", "Thorough (~5 min)")):
            self.w_effort.addItem(lab, k)
        self.w_effort.setCurrentIndex(1)
        self.w_obj = QtWidgets.QComboBox()
        self.w_obj.addItem("Hold the most weight (under the mass limit)", "load")
        self.w_obj.addItem("Best load ÷ own mass", "efficiency")
        self.w_sticks = QtWidgets.QCheckBox("Allow hand-cut Bunnings sticks")
        self.w_sticks.setChecked(True)
        f4.addRow("Effort", self.w_effort)
        f4.addRow("Goal", self.w_obj)
        f4.addRow(self.w_sticks)
        lay.addWidget(g4)

        self.btn_run = QtWidgets.QPushButton("Design my tower")
        self.btn_run.setMinimumHeight(38)
        f = self.btn_run.font()
        f.setBold(True)
        self.btn_run.setFont(f)
        self.btn_run.clicked.connect(self.start_search)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_search)
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.btn_run, 3)
        hb.addWidget(self.btn_stop, 1)
        lay.addLayout(hb)
        self.prog = QtWidgets.QProgressBar()
        self.prog.setRange(0, 1000)
        self.lbl_prog = QtWidgets.QLabel("")
        self.lbl_prog.setWordWrap(True)
        lay.addWidget(self.prog)
        lay.addWidget(self.lbl_prog)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        lay.addWidget(self.log, 1)

        for wdg in (self.w_height, self.w_pole, self.w_clear, self.w_mass, self.w_margin, self.w_density,
                    self.w_quality, self.w_uneven, self.w_sway, self.w_lateral, self.w_imp, self.w_gfactor,
                    self.w_char):
            wdg.valueChanged.connect(self._spec_changed)
        for wdg in (self.w_wood, self.w_glue, self.w_obj):
            wdg.currentIndexChanged.connect(self._spec_changed)
        self.w_sticks.toggled.connect(self._spec_changed)
        self.w_joints.valueChanged.connect(self._spec_changed)
        self.w_slot.valueChanged.connect(self._spec_changed)
        self.w_symglue.toggled.connect(self._spec_changed)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setMinimumWidth(470)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.w_wood.setMinimumContentsLength(18)
        self.w_wood.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def read_spec(self) -> Spec:
        return Spec(
            height=self.w_height.value(), pole_d=self.w_pole.value(), clearance=self.w_clear.value(),
            max_mass=self.w_mass.value(), margin=1 + self.w_margin.value() / 100, max_joints=self.w_joints.value(), jig_slot=self.w_slot.value(),
            wood=self.w_wood.currentData(), measured_density=self.w_density.value(),
            measured_fc=self.spec.measured_fc, measured_E=self.spec.measured_E,
            quality=self.w_quality.value(), glue=self.w_glue.currentData(), glue_factor=self.w_gfactor.value(),
            glue_factors=dict(self.spec.glue_factors), symmetric_glue=self.w_symglue.isChecked(),
            imperfection=self.w_imp.value(), lateral=self.w_lateral.value() / 100,
            uneven=self.w_uneven.value() / 100, sway=self.w_sway.value(), char=self.w_char.value(),
            objective=self.w_obj.currentData(), effort=self.w_effort.currentData(),
            families=[k for k, cb in self.w_fams.items() if cb.isChecked()],
            allow_sticks=self.w_sticks.isChecked(),
        )

    def write_spec(self, sp: Spec):
        widgets = [self.w_height, self.w_pole, self.w_clear, self.w_mass, self.w_margin, self.w_density,
                   self.w_quality, self.w_uneven, self.w_sway, self.w_lateral, self.w_imp, self.w_gfactor,
                   self.w_glue, self.w_wood, self.w_char]
        for wdg in widgets:
            wdg.blockSignals(True)
        self.w_height.setValue(sp.height)
        self.w_pole.setValue(sp.pole_d)
        self.w_clear.setValue(sp.clearance)
        self.w_mass.setValue(sp.max_mass)
        self.w_margin.setValue((sp.margin - 1) * 100)
        self.w_density.setValue(sp.measured_density)
        self.w_quality.setValue(sp.quality)
        self.w_uneven.setValue(sp.uneven * 100)
        self.w_sway.setValue(sp.sway)
        self.w_lateral.setValue(sp.lateral * 100)
        self.w_imp.setValue(sp.imperfection)
        self.w_gfactor.setValue(sp.glue_factor)
        self.w_char.setValue(sp.char)
        self.w_joints.setValue(sp.max_joints)
        self.w_slot.setValue(sp.jig_slot)
        self.w_symglue.setChecked(sp.symmetric_glue)
        if not sp.glue_factors:
            sp.glue_factors = load_glue_factors()
        self.w_wood.setCurrentIndex(max(0, self.w_wood.findData(sp.wood)))
        self.w_glue.setCurrentIndex(max(0, self.w_glue.findData(sp.glue)))
        for wdg in widgets:
            wdg.blockSignals(False)
        self.spec = sp
        self._update_glue_label()

    def _spec_changed(self, *_):
        self.spec = self.read_spec()
        self._update_glue_label()
        self._update_wood_label()
        self.debounce.start()

    # -- right: results ----------------------------------------------------------------------------
    def _build_right(self):
        dock = QtWidgets.QDockWidget("Design", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        tabs = QtWidgets.QTabWidget()
        self.summary = QtWidgets.QTextBrowser()
        self.summary.setOpenExternalLinks(True)
        tabs.addTab(self.summary, "Result")

        cw = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(cw)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "Shape", "Load kg", "Mass g", "Glue", "Fails by"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._pick_candidate)
        cl.addWidget(QtWidgets.QLabel("Best designs found (click to view):"))
        cl.addWidget(self.table)
        tabs.addTab(cw, "Candidates")

        ew = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(ew)
        top = QtWidgets.QHBoxLayout()
        self.w_family = QtWidgets.QComboBox()
        for k, name in FAMILY_NAMES.items():
            self.w_family.addItem(name, k)
        self.w_family.currentIndexChanged.connect(self._family_changed)
        top.addWidget(QtWidgets.QLabel("Shape:"))
        top.addWidget(self.w_family, 1)
        el.addLayout(top)
        self.edit_form_holder = QtWidgets.QWidget()
        self.edit_form = QtWidgets.QFormLayout(self.edit_form_holder)
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.edit_form_holder)
        el.addWidget(sc, 1)
        hb = QtWidgets.QHBoxLayout()
        b_fit = QtWidgets.QPushButton("Fit widths to mass limit")
        b_fit.clicked.connect(self.fit_current)
        hb.addWidget(b_fit)
        el.addLayout(hb)
        tabs.addTab(ew, "Edit")

        self.parts_table = QtWidgets.QTableWidget(0, 6)
        self.parts_table.setHorizontalHeaderLabels(["Part", "Section", "Length mm", "Max mass mg", "Grain", "Working %"])
        self.parts_table.horizontalHeader().setStretchLastSection(True)
        self.parts_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tabs.addTab(self.parts_table, "Parts")

        vw = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(vw)
        self.w_colour = QtWidgets.QComboBox()
        self.w_colour.addItem("Colour by how hard each piece works", "util")
        self.w_colour.addItem("Colour by piece type", "kind")
        self.w_colour.currentIndexChanged.connect(self.refresh_view)
        vl.addWidget(self.w_colour)
        lg = QtWidgets.QLabel("Green = lightly loaded · yellow = half · red = fails first.\n"
                              "Grey tube = 25 mm test pole · blue disc = load plate.\n"
                              "Drag to rotate, scroll to zoom, ⌘-drag to pan.")
        lg.setWordWrap(True)
        vl.addWidget(lg)
        vl.addStretch(1)
        tabs.addTab(vw, "View")

        dock.setWidget(tabs)
        dock.setMinimumWidth(430)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        a_open = QtGui.QAction("Open design…", self)
        a_open.triggered.connect(self.open_design)
        a_save = QtGui.QAction("Save design…", self)
        a_save.triggered.connect(self.save_design)
        a_exp = QtGui.QAction("Export build pack (3D, jig, laser, instructions)…", self)
        a_exp.triggered.connect(self.export)
        self.a_folder = QtGui.QAction("Open last export", self)
        self.a_folder.setEnabled(False)
        self.a_folder.triggered.connect(self.open_export)
        for a in (a_open, a_save, a_exp, self.a_folder):
            tb.addAction(a)

    # -- design editing ----------------------------------------------------------------------------
    def load_design(self, family: str, params: dict):
        self.family = family
        full = {p.name: params.get(p.name, p.decode(0.5)) for p in SPACES[family]}
        self.params = full
        self._building_editor = True
        self.w_family.blockSignals(True)
        self.w_family.setCurrentIndex(self.w_family.findData(family))
        self.w_family.blockSignals(False)
        while self.edit_form.rowCount():
            self.edit_form.removeRow(0)
        self.edit_widgets = {}
        for p in SPACES[family]:
            v = full[p.name]
            if p.kind == "choice":
                w = QtWidgets.QComboBox()
                for c in p.choices:
                    w.addItem(str(c), c)
                w.setCurrentIndex(max(0, list(p.choices).index(v) if v in p.choices else 0))
                w.currentIndexChanged.connect(self._param_changed)
            elif p.kind == "int":
                w = QtWidgets.QSpinBox()
                w.setRange(int(p.lo), int(p.hi))
                w.setValue(int(v))
                w.valueChanged.connect(self._param_changed)
            else:
                w = _dspin(p.lo, p.hi, float(v), 0.1 if p.hi - p.lo < 30 else 1.0, 2)
                w.valueChanged.connect(self._param_changed)
            self.edit_widgets[p.name] = w
            self.edit_form.addRow(p.label, w)
        self._building_editor = False
        self.reanalyse()

    def _param_changed(self, *_):
        if self._building_editor:
            return
        for p in SPACES[self.family]:
            w = self.edit_widgets[p.name]
            self.params[p.name] = w.currentData() if p.kind == "choice" else w.value()
        self.debounce.start()

    def _family_changed(self, *_):
        fam = self.w_family.currentData()
        s = seeds(fam, self.spec)
        self.load_design(fam, fit_to_budget(fam, s[min(2, len(s) - 1)], self.spec))

    def _update_glue_label(self):
        f = self.spec.factors()
        meas = self.spec.glue_factors
        self.lbl_gf.setText("Factors: " + ", ".join(
            f"{GLUES[g].name.split(' (')[0]} {f[g]:.1f}{' (measured)' if g in meas else ''}" for g in GLUES))

    def _update_wood_label(self):
        w = self.spec.wood_props(False)
        if self.spec.measured_fc > 0:
            src = f"Measured: crushing {self.spec.measured_fc:.1f} MPa"
            src += f", stiffness {self.spec.measured_E:,.0f} MPa" if self.spec.measured_E > 0 else ""
        else:
            src = f"Book values (crushing {w.f_c_L:.1f} MPa, stiffness {w.E_L:,.0f} MPa) — run the wood test"
        self.lbl_wood.setText(src)

    def wood_test(self):
        dlg = WoodTestDialog(self, self.spec)
        if dlg.exec():
            t = dlg.test
            save_wood_test(t)
            t.apply(self.spec)
            self.w_wood.blockSignals(True)
            self.w_wood.setCurrentIndex(self.w_wood.findData("teacher"))
            self.w_wood.blockSignals(False)
            self.w_density.blockSignals(True)
            self.w_density.setValue(self.spec.measured_density)
            self.w_density.blockSignals(False)
            self._update_wood_label()
            self.reanalyse()

    def glue_test(self):
        dlg = GlueTestDialog(self, self.spec)
        if dlg.exec():
            self.spec.glue_factors = dlg.result_factors
            save_glue_factors(self.spec.glue_factors)
            self._update_glue_label()
            self.reanalyse()

    def calibrate_glue(self):
        if self.st is None:
            return
        from .analysis import calibrate_glue_factor
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Calibrate glue")
        fl = QtWidgets.QFormLayout(dlg)
        info = QtWidgets.QLabel("Enter the glue and wood mass of a tower you know about (weigh the wood before "
                                "gluing and the tower after). The factor is set so the CURRENT design would "
                                "carry that ratio.")
        info.setWordWrap(True)
        fl.addRow(info)
        wg = _dspin(0.01, 100, 5.0, 0.1, 2, "g glue")
        ww = _dspin(0.1, 200, 15.0, 0.5, 2, "g wood")
        fl.addRow("Glue", wg)
        fl.addRow("Wood", ww)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok |
                                        QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        fl.addRow(bb)
        if dlg.exec():
            f = calibrate_glue_factor(self.st, self.spec, wg.value(), ww.value())
            self.w_gfactor.setValue(f)

    def fit_current(self):
        self.load_design(self.family, fit_to_budget(self.family, self.params, self.spec))

    def reanalyse(self):
        try:
            self.st = build(self.family, self.params, self.spec)
            self.res = analyse(self.st, self.spec, accurate=True)
        except Infeasible as e:
            self.st, self.res = None, None
            self.summary.setHtml(f"<h3>Not buildable</h3><p>{e}</p>")
            self.view.show_structure(None)
            return
        except Exception as e:  # keep the app alive on odd parameter combinations
            self.st, self.res = None, None
            self.summary.setHtml(f"<h3>Analysis failed</h3><pre>{e}</pre>")
            return
        self.refresh_view()
        self._fill_summary()
        self._fill_parts()

    def refresh_view(self, *_):
        if self.st is not None:
            self.view.show_structure(self.st, self.res.util if self.res else None, self.w_colour.currentData(),
                                     self.spec)

    def _fill_summary(self):
        r, st, sp = self.res, self.st, self.spec
        ok = r.mass.total <= sp.max_mass
        mcol = "#2b8a3e" if ok else "#c92a2a"
        info = st.info
        lines = [
            f"<h2 style='margin-bottom:0'>{FAMILY_NAMES[st.family]}</h2>",
            f"<p style='color:#666'>{info.get('n')} sides · base Ø{2 * info.get('base_r', 0):.0f} mm → top "
            f"Ø{2 * info.get('top_r', 0):.0f} mm · {len(st.levels) - 1} levels · height {st.height:.0f} mm</p>",
            f"<p>Predicted failure load<br><span style='font-size:26px;font-weight:700;color:#e8590c'>"
            f"{r.load_kg:.1f} kg</span>  <span style='color:#666'>({r.efficiency:,.0f}× its own mass)</span></p>",
            f"<p>Mass (with +{(sp.margin - 1) * 100:.0f} % on every piece and joint)<br>"
            f"<span style='font-size:22px;font-weight:700;color:{mcol}'>{r.mass.total:.2f} g</span> "
            f"<span style='color:#666'>/ {sp.max_mass:.1f} g limit — wood {r.mass.wood:.2f} g, glue {r.mass.glue:.2f} g</span></p>",
            f"<p><b>Fails first by:</b> {r.governing}</p>",
            f"<p><b>Build effort:</b> {r.n_parts} pieces, {r.n_joints} glue joints"
            + (f" <span style='color:#c92a2a'>(over your limit of {sp.max_joints})</span>"
               if sp.max_joints and r.n_joints > sp.max_joints else "") + "</p>",
            f"<p><b>Whole-tower buckling:</b> {r.lam_global / 9.81:.0f} kg<br>",
        ]
        from collections import Counter
        cnt = Counter(r.joint_glue)
        grow = ""
        for g, gl_ in GLUES.items():
            js = [j for j, c in enumerate(r.joint_glue) if c == g]
            if not js and not (g == r.laminate_glue_id and r.mass.laminate_glue > 0):
                continue
            mass_g = sum(r.mass.per_joint[j] for j in js) + (r.mass.laminate_glue if g == r.laminate_glue_id else 0)
            grow += f"<tr><td>{gl_.name}</td><td align=right>{cnt.get(g, 0)}</td><td align=right>{mass_g:.2f} g</td></tr>"
        lines.append(f"<p><b>Glue</b> ({'chosen per joint' if r.glue == 'mixed' else GLUES[r.glue].name}) — "
                     f"{r.mass.glue:.2f} g on {r.mass.wood:.2f} g wood = <b>{r.mass.glue_ratio * 15:.1f} g per 15 g</b> "
                     f"<span style='color:#666'>(rule of thumb: 5 g per 15 g)</span></p>"
                     "<table cellspacing=0 cellpadding=3 border=1 style='border-collapse:collapse'>"
                     "<tr><th>Glue</th><th>Joints</th><th>Dried mass</th></tr>" + grow + "</table>")
        weakest = np.argsort(-r.util)[:6]
        rows = "".join(f"<tr><td>{st.parts[st.members[k].part].name}</td><td>{st.members[k].sec.label()}</td>"
                       f"<td align=right>{r.util[k] * 100:.0f} %</td><td align=right>{st.members[k].grain_deg:.0f}°</td></tr>"
                       for k in weakest)
        lines.append("<p><b>Hardest-working pieces</b></p><table cellspacing=0 cellpadding=3 border=1 "
                     "style='border-collapse:collapse'><tr><th>Piece</th><th>Section</th><th>Load</th><th>Grain</th></tr>"
                     + rows + "</table>")
        if not ok:
            lines.append("<p style='color:#c92a2a'><b>Over the mass limit.</b> Use “Fit widths to mass limit” on the Edit tab.</p>")
        self.summary.setHtml("".join(lines))

    def _fill_parts(self):
        st, r = self.st, self.res
        from .lasercut import part_ids
        ids = part_ids(st)
        self.parts_table.setRowCount(0)
        for i, p in enumerate(st.parts):
            if p.kind == "corner":
                continue
            u = max((r.util[m] for m in p.members), default=0)
            g = max((st.members[m].grain_deg for m in p.members), default=0)
            row = self.parts_table.rowCount()
            self.parts_table.insertRow(row)
            vals = [f"{ids[i]}  {p.name}", p.sec.label() if p.sec else "", f"{p.length:.1f}",
                    f"{r.mass.per_part[i] * 1000:.0f}", f"{g:.0f}°", f"{u * 100:.0f}"]
            for c, v in enumerate(vals):
                self.parts_table.setItem(row, c, QtWidgets.QTableWidgetItem(v))
        self.parts_table.resizeColumnsToContents()

    # -- search ------------------------------------------------------------------------------------
    def start_search(self):
        self.spec = self.read_spec()
        if not self.spec.families:
            QtWidgets.QMessageBox.warning(self, "Tower Designer", "Tick at least one shape to try.")
            return
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.clear()
        self.log.appendPlainText(f"Searching {', '.join(FAMILY_NAMES[f] for f in self.spec.families)} "
                                 f"on {os.cpu_count()} cores…")
        self.thread = QtCore.QThread()
        self.worker = OptWorker(self.spec)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def stop_search(self):
        if self.worker:
            self.worker.stop.set()
            self.lbl_prog.setText("Stopping after the current batch…")

    def _on_progress(self, frac, msg, best):
        self.prog.setValue(int(frac * 1000))
        txt = msg
        if best is not None:
            txt += f"\nBest so far: {FAMILY_NAMES[best.family]}, {best.load_kg:.1f} kg at {best.mass:.2f} g"
        self.lbl_prog.setText(txt)

    def _on_done(self, cands, log):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        for line in log:
            self.log.appendPlainText(line)
        self.candidates = cands
        self.table.setRowCount(0)
        for k, c in enumerate(cands, start=1):
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, v in enumerate([str(k), FAMILY_NAMES[c.family], f"{c.load_kg:.1f}", f"{c.mass:.2f}",
                                     c.glue.upper(), c.governing]):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        if cands:
            self.table.selectRow(0)
            self.statusBar().showMessage(f"Best: {cands[0].load_kg:.1f} kg at {cands[0].mass:.2f} g", 0)
        else:
            self.statusBar().showMessage("No buildable design found — try relaxing the brief.")

    def _on_fail(self, tb):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log.appendPlainText(tb)
        QtWidgets.QMessageBox.critical(self, "Search failed", tb[-1500:])

    def _pick_candidate(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        c = self.candidates[rows[0].row()]
        self.load_design(c.family, c.params)

    # -- files -------------------------------------------------------------------------------------
    def save_design(self):
        os.makedirs(DESIGNS_DIR, exist_ok=True)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save design", os.path.join(DESIGNS_DIR, "tower.json"),
                                                        "Tower design (*.json)")
        if path:
            save_design(path, self.family, self.params, self.spec)

    def open_design(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open design", DESIGNS_DIR, "Tower design (*.json)")
        if path:
            fam, params, sp = load_design(path)
            self.write_spec(sp)
            self.load_design(fam, params)

    def export(self):
        if self.st is None:
            QtWidgets.QMessageBox.warning(self, "Tower Designer", "The current design is not buildable.")
            return
        os.makedirs(DESIGNS_DIR, exist_ok=True)
        name = f"{self.family}-{self.res.load_kg:.0f}kg-{_dt.datetime.now():%Y%m%d-%H%M}"
        default = os.path.join(DESIGNS_DIR, name)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export build pack to folder", default, "Folder")
        if not path:
            return
        self.statusBar().showMessage("Exporting…")
        self.ethread = QtCore.QThread()
        self.eworker = ExportWorker(self.family, dict(self.params), self.spec, path)
        self.eworker.moveToThread(self.ethread)
        self.ethread.started.connect(self.eworker.run)
        self.eworker.progress.connect(lambda m: self.statusBar().showMessage(f"Exporting: {m}…"))
        self.eworker.finished.connect(self._export_done)
        self.eworker.failed.connect(lambda tb: QtWidgets.QMessageBox.critical(self, "Export failed", tb[-1500:]))
        self.eworker.finished.connect(self.ethread.quit)
        self.eworker.failed.connect(self.ethread.quit)
        self.ethread.start()

    def _export_done(self, r):
        self.last_export = r["outdir"]
        self.a_folder.setEnabled(True)
        msg = f"Exported to {r['outdir']}"
        if r["notes"]:
            msg += "\n\nNotes:\n" + "\n".join(r["notes"])
        self.statusBar().showMessage(f"Exported to {r['outdir']}", 0)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.join(r["outdir"], "instructions.html")))
        QtWidgets.QMessageBox.information(self, "Build pack ready", msg)

    def open_export(self):
        if self.last_export:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.last_export))

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop.set()
        super().closeEvent(e)


class WoodTestDialog(QtWidgets.QDialog):
    """Enter density, crush and buckling readings; work out the wood's real strength and stiffness."""

    def __init__(self, parent, spec: Spec):
        super().__init__(parent)
        from . import woodtest as wt
        self.wt, self.spec = wt, spec
        self.setWindowTitle("Wood test")
        self.resize(640, 640)
        t0 = load_wood_test() or WoodTest()
        lay = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "<b>How:</b> (1) weigh some whole sticks; (2) crush 5 short pieces, 12 mm long with ends sanded square, "
            "on a scale that reads to 25 kg+ and record each peak; (3) press 3 straight 150 mm pieces in the "
            "printed end caps until they bow sideways and record the steady load. "
            "<b>Export test kit</b> gives the printable helpers and a sheet with the full method.")
        info.setWordWrap(True)
        lay.addWidget(info)

        g1 = QtWidgets.QGroupBox("1. Density")
        f1 = QtWidgets.QFormLayout(g1)
        self.side = _dspin(1, 10, t0.side, 0.05, 2, "mm")
        self.n = QtWidgets.QSpinBox()
        self.n.setRange(1, 100)
        self.n.setValue(t0.n_sticks)
        self.slen = _dspin(10, 2000, t0.stick_len, 1, 0, "mm")
        self.smass = _dspin(0, 500, t0.stick_mass, 0.01, 3, "g")
        f1.addRow("Stick size (square)", self.side)
        f1.addRow("Number of sticks weighed", self.n)
        f1.addRow("Length of each stick", self.slen)
        f1.addRow("Total mass", self.smass)
        lay.addWidget(g1)

        g2 = QtWidgets.QGroupBox("2. Crush test (short pieces)")
        f2 = QtWidgets.QFormLayout(g2)
        self.clen = _dspin(3, 30, t0.crush_len, 0.5, 1, "mm")
        f2.addRow("Sample length", self.clen)
        row = QtWidgets.QHBoxLayout()
        self.crush = []
        for k in range(6):
            w = _dspin(0, 200, t0.crush_kg[k] if k < len(t0.crush_kg) else 0, 0.1, 2, "kg")
            self.crush.append(w)
            row.addWidget(w)
        f2.addRow("Peak readings", row)
        lay.addWidget(g2)

        g3 = QtWidgets.QGroupBox("3. Buckling test (long pieces)")
        f3 = QtWidgets.QFormLayout(g3)
        self.blen = _dspin(50, 400, t0.buckle_len, 1, 0, "mm")
        self.bK = QtWidgets.QComboBox()
        self.bK.addItem("Printed square-socket end caps (K = 0.65)", wt.K_CAPS)
        self.bK.addItem("Rounded/pointed ends (K = 1.0)", wt.K_PINNED)
        self.bK.setCurrentIndex(0 if abs(t0.buckle_K - wt.K_CAPS) < 0.01 else 1)
        f3.addRow("Sample length", self.blen)
        f3.addRow("Ends", self.bK)
        row2 = QtWidgets.QHBoxLayout()
        self.buck = []
        for k in range(4):
            w = _dspin(0, 100, t0.buckle_kg[k] if k < len(t0.buckle_kg) else 0, 0.05, 2, "kg")
            self.buck.append(w)
            row2.addWidget(w)
        f3.addRow("Buckling loads", row2)
        lay.addWidget(g3)

        self.out = QtWidgets.QLabel("")
        self.out.setWordWrap(True)
        self.out.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lay.addWidget(self.out)
        bb = QtWidgets.QHBoxLayout()
        b_kit = QtWidgets.QPushButton("Export test kit…")
        b_kit.clicked.connect(self.export_kit)
        b_clear = QtWidgets.QPushButton("Use book values")
        b_clear.clicked.connect(self.clear)
        self.b_ok = QtWidgets.QPushButton("Apply")
        self.b_ok.setDefault(True)
        self.b_ok.clicked.connect(self.accept)
        b_cancel = QtWidgets.QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        bb.addWidget(b_kit)
        bb.addWidget(b_clear)
        bb.addStretch(1)
        bb.addWidget(b_cancel)
        bb.addWidget(self.b_ok)
        lay.addLayout(bb)
        for w in [self.side, self.slen, self.smass, self.clen, self.blen] + self.crush + self.buck:
            w.valueChanged.connect(self._recalc)
        self.n.valueChanged.connect(self._recalc)
        self.bK.currentIndexChanged.connect(self._recalc)
        self._recalc()

    def _read(self) -> WoodTest:
        return WoodTest(side=self.side.value(), n_sticks=self.n.value(), stick_len=self.slen.value(),
                        stick_mass=self.smass.value(), crush_len=self.clen.value(),
                        crush_kg=[w.value() for w in self.crush if w.value() > 0],
                        buckle_len=self.blen.value(), buckle_K=self.bK.currentData(),
                        buckle_kg=[w.value() for w in self.buck if w.value() > 0])

    def _recalc(self, *_):
        t = self._read()
        book = WOODS["teacher"]
        rho, fc, E = t.density(), t.fc(), t.E()
        lines = [
            f"Density: <b>{rho:.0f} kg/m³</b>" if rho else "Density: —",
            f"Crushing strength: <b>{fc:.1f} MPa</b> (book value used now: {book.f_c_L:.1f})" if fc else "Crushing strength: —",
            f"Stiffness: <b>{E:,.0f} MPa</b> (book value used now: {book.E_L:,.0f})" if E else "Stiffness: — (optional)",
        ]
        warn = t.checks()
        if warn:
            lines.append("<span style='color:#c92a2a'>" + "<br>".join(warn) + "</span>")
        self.out.setText("<br>".join(lines))
        self.b_ok.setEnabled(bool(rho and fc))
        self.test = t

    def clear(self):
        self.spec.measured_fc = self.spec.measured_E = self.spec.measured_density = 0.0
        for w in self.crush + self.buck + [self.smass]:
            w.setValue(0)
        self._recalc()
        self.test = WoodTest()
        self.accept()

    def export_kit(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the wood test kit",
                                                          os.path.expanduser("~/Documents"))
        if not path:
            return
        out = os.path.join(path, "Wood test kit")
        self.wt.export_kit(out, self.spec)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.join(out, "wood_test.html")))


class GlueTestDialog(QtWidgets.QDialog):
    """Enter the before/after masses of the standard test joints; work out each glue's factor."""

    def __init__(self, parent, spec: Spec):
        super().__init__(parent)
        from . import gluetest as gt
        self.gt, self.spec = gt, spec
        self.setWindowTitle("Glue test")
        self.resize(900, 420)
        lay = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            f"<b>How:</b> for each glue, weigh one {gt.HOST_LEN:.0f} mm length of 5×5 rod together with "
            f"{gt.N_JOINTS} strips ({gt.STRIP[0]:g}×{gt.STRIP[1]:g}×{gt.STRIP[2]:g} mm). Glue each strip's end onto the "
            "rod <i>exactly the way you'll glue the tower</i>, let it cure fully, and weigh again. Enter both masses "
            "below. <b>Export test kit</b> gives you the laser file for the strips, a printable jig that holds them "
            "square, and a printable sheet with the full method.")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.table = QtWidgets.QTableWidget(len(GLUES), 7)
        self.table.setHorizontalHeaderLabels(["Glue", "Joints", "Before (g)", "After (g)", "Measured per joint",
                                              "App expects now", "New factor"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.rows = {}
        for r, (g, gl_) in enumerate(GLUES.items()):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(gl_.name))
            n = QtWidgets.QSpinBox()
            n.setRange(1, 50)
            n.setValue(gt.N_JOINTS)
            b = _dspin(0, 200, 0, 0.01, 3, "g")
            a = _dspin(0, 200, 0, 0.01, 3, "g")
            for w in (n, b, a):
                w.valueChanged.connect(self._recalc)
            self.table.setCellWidget(r, 1, n)
            self.table.setCellWidget(r, 2, b)
            self.table.setCellWidget(r, 3, a)
            for c in (4, 5, 6):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(""))
            self.rows[g] = (r, n, b, a)
        lay.addWidget(self.table)
        self.note = QtWidgets.QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color:#c92a2a")
        lay.addWidget(self.note)
        bb = QtWidgets.QHBoxLayout()
        b_kit = QtWidgets.QPushButton("Export test kit…")
        b_kit.clicked.connect(self.export_kit)
        b_reset = QtWidgets.QPushButton("Forget measurements")
        b_reset.clicked.connect(self.reset)
        b_ok = QtWidgets.QPushButton("Apply")
        b_ok.setDefault(True)
        b_ok.clicked.connect(self.accept)
        b_cancel = QtWidgets.QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        for w in (b_kit, b_reset):
            bb.addWidget(w)
        bb.addStretch(1)
        bb.addWidget(b_cancel)
        bb.addWidget(b_ok)
        lay.addLayout(bb)
        self.result_factors = dict(spec.glue_factors)
        self._recalc()

    def _recalc(self, *_):
        gt = self.gt
        out = dict(self.spec.glue_factors)
        notes = []
        for g, (r, n, b, a) in self.rows.items():
            m = gt.Measurement(g, n.value(), b.value(), a.value())
            ev = gt.evaluate(m, self.spec)
            entered = b.value() > 0 and a.value() > 0
            self.table.item(r, 4).setText(f"{ev['per_joint_mg']:.1f} mg" if entered else "")
            self.table.item(r, 5).setText(f"{ev['model_now_mg']:.1f} mg")
            if entered and ev["factor"]:
                out[g] = ev["factor"]
                self.table.item(r, 6).setText(f"{ev['factor']:.2f}  (was {self.spec.factors()[g]:.2f})")
            else:
                self.table.item(r, 6).setText(f"{self.spec.factors()[g]:.2f} (unchanged)")
            if entered and ev["warning"]:
                notes.append(f"{GLUES[g].name}: {ev['warning']}")
        self.note.setText("\n".join(notes))
        self.result_factors = out
        self.table.resizeColumnsToContents()

    def reset(self):
        for g, (r, n, b, a) in self.rows.items():
            b.setValue(0)
            a.setValue(0)
        self.spec.glue_factors = {}
        self._recalc()
        self.result_factors = {}

    def export_kit(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder for the glue test kit",
                                                          os.path.expanduser("~/Documents"))
        if not path:
            return
        out = os.path.join(path, "Glue test kit")
        self.gt.export_kit(out, self.spec)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.join(out, "glue_test.html")))


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Tower Designer")
    win = MainWindow()
    win.show()
    # "Tower Designer.app <design.json>" opens that design straight away; with none, the reference tower
    paths = [a for a in sys.argv[1:] if a.endswith(".json") and os.path.exists(a)] or [REFERENCE_DESIGN]
    for arg in paths:
        if os.path.exists(arg):
            fam, params, sp = load_design(arg)
            sp.glue_factors = sp.glue_factors or load_glue_factors()
            win.write_spec(sp)
            win.load_design(fam, params)
            win.statusBar().showMessage("Opened the reference tower" if arg == REFERENCE_DESIGN else f"Opened {arg}", 0)
    sys.exit(app.exec())
