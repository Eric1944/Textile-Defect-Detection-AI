"""Application windows. Every window is created once and then shown / hidden (lecture: persistent windows)."""
import datetime, time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QAction, QPixmap, QColor, QIcon, QFont, QKeySequence
from PyQt6.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, QFrame,
                             QComboBox, QProgressBar, QSlider, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QListWidget, QListWidgetItem, QLineEdit, QSpinBox, QDoubleSpinBox, QDialog,
                             QFormLayout, QCheckBox, QStatusBar, QSizePolicy, QAbstractItemView, QApplication, QListView)

from .i18n import tr, LANGS
from .core import save_settings
from .theme import TH, build_qss, restyle_plots, style_plot, ORANGE, GREEN, RED, BLUE, YELLOW, GRAY, PURPLE, CLASS_COLORS
from .widgets import (BaseWindow, card, Lamp, KpiTile, ProbBars, ConveyorView, BinView, Gauge, Donut, StateTimeline, SignalTrace,
                      PIX, heat_pixmap, L, B, page, fnum, fpct, fdur, retranslate_tree, retranslate_cards)
from .stats_window import StatsWindow
from .extra_windows import AlarmWindow, BatchWindow


# =====================================================================
# Process window 1 — AI inspection
# =====================================================================
class InspectionWindow(BaseWindow):
    title_key = 'win_inspect'

    def __init__(self, ctrl):
        super().__init__(ctrl); self.resize(1380, 820)
        self.cur = None; self.heat = None; self.history = []
        root = QHBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(12)
        left = QVBoxLayout(); root.addLayout(left, 3)
        cf, cl, _ = card('conveyor', '🏭'); self.view = ConveyorView(ctrl); cl.addWidget(self.view); left.addWidget(cf, 1)
        # manual control
        mf, ml, _ = card('manual_ctrl', '🕹')
        row = QHBoxLayout()
        self.b_jog = B('jog'); self.b_jog.pressed.connect(lambda: ctrl.set_jog(True)); self.b_jog.released.connect(lambda: ctrl.set_jog(False))
        self.b_insp = B('inspect_now', 'Accent', ctrl.manual_inspect)
        self.b_rej = B('push_now', None, lambda: ctrl.manual_decide(True)); self.b_rej.setStyleSheet(f'color:{RED}; font-weight:800;')
        self.b_acc = B('accept_now', None, lambda: ctrl.manual_decide(False)); self.b_acc.setStyleSheet(f'color:{GREEN}; font-weight:800;')
        self.b_open = B('open_image', None, self.open_image)
        for b in (self.b_jog, self.b_insp, self.b_rej, self.b_acc, self.b_open):
            row.addWidget(b)
        ml.addLayout(row)
        row2 = QHBoxLayout(); row2.addWidget(L('speed'))
        self.speed = QSlider(Qt.Orientation.Horizontal); self.speed.setRange(3, 30); self.speed.setValue(int(ctrl.settings['speed'] * 100))
        self.speed.valueChanged.connect(self.on_speed); row2.addWidget(self.speed, 1)
        self.speed_lbl = QLabel(); self.speed_lbl.setMinimumWidth(40); row2.addWidget(self.speed_lbl); ml.addLayout(row2)
        self.hint = L('manual_hint', 'Sub'); ml.addWidget(self.hint)
        left.addWidget(mf)
        hf, hl, _ = card('history', '🕘')
        self.hist = QListWidget(); self.hist.setViewMode(QListView.ViewMode.IconMode); self.hist.setFlow(QListView.Flow.LeftToRight)
        self.hist.setWrapping(False); self.hist.setIconSize(QSize(64, 64)); self.hist.setFixedHeight(104)
        self.hist.setMovement(QListView.Movement.Static); self.hist.itemClicked.connect(self.on_hist)
        hl.addWidget(self.hist); left.addWidget(hf)
        # right column: current frame and AI result
        right = QVBoxLayout(); root.addLayout(right, 2)
        f1, l1, _ = card('current_image', '📷')
        self.img = QLabel(); self.img.setFixedSize(260, 260); self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img.setStyleSheet('background:#0B1117; border-radius:8px; color:#90A4AE;')
        l1.addWidget(self.img, alignment=Qt.AlignmentFlag.AlignCenter)
        self.file_lbl = QLabel(''); self.file_lbl.setObjectName('Sub'); l1.addWidget(self.file_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        r = QHBoxLayout()
        self.b_explain = B('explain', 'Accent', self.explain, tip='explain_tip')
        self.chk_heat = QCheckBox(tr.t('show_heat')); self.chk_heat.setProperty('tkey', 'show_heat'); self.chk_heat.setChecked(True)
        self.chk_heat.toggled.connect(self.redraw_img)
        self.chk_freeze = QCheckBox(tr.t('freeze')); self.chk_freeze.setProperty('tkey', 'freeze')
        r.addWidget(self.b_explain); r.addWidget(self.chk_heat); r.addWidget(self.chk_freeze); l1.addLayout(r)
        self.heat_lbl = QLabel(''); self.heat_lbl.setObjectName('Sub'); self.heat_lbl.setWordWrap(True); l1.addWidget(self.heat_lbl)
        right.addWidget(f1)
        f2, l2, _ = card('ai_result', '🧠')
        g = QGridLayout(); g.setVerticalSpacing(6)
        self.v_cls, self.v_dec, self.v_truth, self.v_pdef, self.v_thr, self.v_lat = (QLabel('—') for _ in range(6))
        self.v_cls.setStyleSheet('font-size:16pt; font-weight:800;'); self.v_dec.setStyleSheet('font-size:12pt; font-weight:800;')
        self.conf_bar = QProgressBar(); self.conf_bar.setRange(0, 1000); self.conf_bar.setFormat('%p%')
        self.pdef_bar = QProgressBar(); self.pdef_bar.setRange(0, 1000); self.pdef_bar.setFormat('%p%')
        rows = [('class', self.v_cls), ('confidence', self.conf_bar), ('p_defect', self.pdef_bar), ('threshold', self.v_thr),
                ('decision', self.v_dec), ('truth', self.v_truth), ('latency', self.v_lat)]
        for k, (key, w) in enumerate(rows):
            g.addWidget(L(key, 'Sub'), k, 0); g.addWidget(w, k, 1)
        l2.addLayout(g)
        l2.addWidget(L('probs', 'Sub')); self.bars = ProbBars(); l2.addWidget(self.bars)
        right.addWidget(f2, 1)
        ctrl.tick.connect(self.view.update)
        ctrl.result.connect(self.show_result)
        ctrl.explained.connect(self.on_explained)
        ctrl.state_changed.connect(self.update_mode)
        ctrl.state_changed.connect(lambda: self.show_result(self.cur, add=False, force=True) if self.cur else None)
        self.on_speed(self.speed.value()); self.update_mode()

    def on_speed(self, v):
        self.ctrl.settings['speed'] = v / 100
        self.speed_lbl.setText(f'{v / 100:.2f}'.replace('.', ','))

    def update_mode(self):
        manual = self.ctrl.mode == 'manual'
        sim = self.ctrl.plc.kind == 'sim'
        self.b_jog.setEnabled(manual and sim)
        for b in (self.b_insp, self.b_rej, self.b_acc):
            b.setEnabled(manual)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, tr.t('open_image'), '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if path:
            self.ctrl.inspect_file(path)

    def explain(self):
        if self.cur:
            self.chk_freeze.setChecked(True)            # keep this frame while the map is computed
            self.b_explain.setEnabled(False); self.heat_lbl.setText(tr.t('explain_busy'))
            self.ctrl.explain(self.cur['path'])

    def on_explained(self, r):
        self.b_explain.setEnabled(True)
        if 'error' in r:
            self.heat_lbl.setText(r['error']); return
        if self.cur and r['path'] == self.cur['path']:
            self.heat = r
            self.heat_lbl.setText(tr.t('explain_done', cls=tr.cls(r['name']), g=r['grid'], ms=fnum(r['ms'], 0)))
            self.chk_heat.setChecked(True); self.redraw_img()

    def redraw_img(self):
        if not self.cur:
            return
        if self.heat is not None and self.chk_heat.isChecked() and self.heat['path'] == self.cur['path']:
            pm = heat_pixmap(self.cur['path'], self.heat['heat'], 260)
        else:
            pm = QPixmap(self.cur['path']).scaled(260, 260, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.img.setPixmap(pm)

    def on_hist(self, item):
        r = item.data(Qt.ItemDataRole.UserRole)
        if r:
            self.chk_freeze.setChecked(True); self.show_result(r, add=False, force=True)

    def show_result(self, r, add=True, force=False):
        if add and (r.get('source') != 'operator'):
            self.history.insert(0, r); self.history = self.history[:30]
            defect = r.get('decision_defect', r['defect'])
            it = QListWidgetItem(QIcon(PIX.get(r['path'])), tr.cls(r['name']))
            it.setForeground(QColor(RED if defect else GREEN)); it.setData(Qt.ItemDataRole.UserRole, r)
            self.hist.insertItem(0, it)
            while self.hist.count() > 30:
                self.hist.takeItem(self.hist.count() - 1)
        if self.chk_freeze.isChecked() and not force and self.cur is not None:
            return
        if self.cur is None or self.cur['path'] != r['path']:
            self.heat_lbl.setText('')
        self.cur = r
        self.redraw_img(); self.file_lbl.setText(Path(r['path']).name)
        defect = r.get('decision_defect', r['defect'])
        col = RED if defect else GREEN
        self.v_cls.setText(tr.cls(r['name'])); self.v_cls.setStyleSheet(f'font-size:16pt; font-weight:800; color:{CLASS_COLORS.get(r["name"], TH["text"])};')
        self.conf_bar.setValue(int(r['confidence'] * 1000)); self.pdef_bar.setValue(int(r['p_defect'] * 1000))
        self.v_thr.setText(f'{r["threshold"]:.3f}'.replace('.', ','))
        txt = tr.t('dec_defect' if defect else 'dec_good')
        if r.get('source') == 'manual':
            txt += '  (?)'
        self.v_dec.setText(txt); self.v_dec.setStyleSheet(f'font-size:12pt; font-weight:800; color:{col};')
        truth = r.get('truth')
        if truth:
            ok = (truth != 'good') == defect
            self.v_truth.setText(f'{tr.cls(truth)}  {"✔" if ok else "✖"}')
            self.v_truth.setStyleSheet(f'color:{GREEN if ok else RED}; font-weight:700;')
        else:
            self.v_truth.setText('—'); self.v_truth.setStyleSheet('')
        self.v_lat.setText(f'{r["latency_ms"]:.1f} ms'.replace('.', ','))
        names = self.ctrl.model.names
        self.bars.set([(tr.cls(n), float(v), CLASS_COLORS.get(n, ORANGE)) for n, v in zip(names, r['probs'])])


# =====================================================================
# Process window 2 — sorting
# =====================================================================
class SortingWindow(BaseWindow):
    title_key = 'win_sort'

    def __init__(self, ctrl):
        super().__init__(ctrl); self.resize(1280, 760)
        root = QHBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(12)
        lf, ll, _ = card('win_sort', '📦'); self.bins = BinView(ctrl); ll.addWidget(self.bins)
        self.mis = L('misclassified', 'Sub'); ll.addWidget(self.mis)
        root.addWidget(lf, 3)
        right = QVBoxLayout(); root.addLayout(right, 2)
        kf, kl, _ = card('pusher', '🔧')
        self.lamp_push = Lamp(ORANGE, 30); self.v_strokes = QLabel('0'); self.v_strokes.setObjectName('Big')
        r = QHBoxLayout(); r.addWidget(self.lamp_push); r.addWidget(L('pushes', 'Sub')); r.addStretch(); r.addWidget(self.v_strokes); kl.addLayout(r)
        right.addWidget(kf)
        df, dl, _ = card('by_class', '🍩'); self.donut = Donut('kpi_defect'); dl.addWidget(self.donut); right.addWidget(df, 1)
        rf, rl, _ = card('last_items', '🧾')
        self.list = QListWidget(); self.list.setIconSize(QSize(40, 40)); rl.addWidget(self.list); right.addWidget(rf, 1)
        ctrl.tick.connect(self.on_tick); ctrl.result.connect(self.on_result)

    def on_tick(self):
        if not self.isVisible():
            return
        self.bins.update(); self.lamp_push.set(self.ctrl.plc.Q['pusher'])
        self.v_strokes.setText(str(self.ctrl.sim.push_strokes))

    def on_result(self, r):
        if r.get('source') == 'file' or (r.get('source') == 'manual' and 'decision_defect' not in r):
            return
        defect = r.get('decision_defect', r['defect'])
        it = QListWidgetItem(QIcon(PIX.get(r['path'])),
                             f'#{r.get("item", "")}  {tr.cls(r["name"])}  {r["confidence"] * 100:.1f} %  →  '
                             + tr.t('dec_defect' if defect else 'dec_good'))
        it.setForeground(QColor(RED if defect else GREEN))
        self.list.insertItem(0, it)
        while self.list.count() > 80:
            self.list.takeItem(self.list.count() - 1)
        names = [n for n in self.ctrl.names if n != 'good']
        by = {n: 0 for n in names}
        for rec in self.ctrl.stats.records:
            if rec['decision'] == 'defect' and rec['pred'] in by:
                by[rec['pred']] += 1
        self.donut.set([(tr.cls(n), by[n], CLASS_COLORS.get(n, ORANGE)) for n in names if by[n]])
        s = self.ctrl.stats.summary(self.ctrl.names)
        self.mis.setText(tr.t('misclassified', fp=s['fp'], fn=s['fn']))


# =====================================================================
# PLC simulation window
# =====================================================================
class PlcWindow(BaseWindow):
    title_key = 'win_plc'
    INPUTS = [('%I0.0', 'start', 'tag_start'), ('%I0.1', 'stop_nc', 'tag_stop'), ('%I0.2', 'estop_nc', 'tag_estop'), ('%I0.3', 'sensor', 'tag_sensor')]
    OUTPUTS = [('%Q0.0', 'motor', 'tag_motor', GREEN), ('%Q0.1', 'pusher', 'tag_pusher', ORANGE), ('%Q0.2', 'green', 'tag_green', GREEN),
               ('%Q0.3', 'red', 'tag_red', RED), ('%Q0.4', 'alarm', 'tag_alarm', RED)]
    DB = [('ImageRequest', 'Bool', 'PLC'), ('ResultReady', 'Bool', 'AI'), ('Defect', 'Bool', 'AI'), ('DefectClass', 'Int', 'AI'),
          ('Confidence', 'Real', 'AI'), ('GoodCount', 'Int', 'PLC'), ('BadCount', 'Int', 'PLC')]

    def __init__(self, ctrl):
        super().__init__(ctrl); self.resize(1240, 900)
        root = QGridLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(12)
        fi, li, _ = card('inputs', '⬅'); self.in_lamps = {}
        for addr, key, tkey in self.INPUTS:
            r = QHBoxLayout(); lamp = Lamp(GREEN); self.in_lamps[key] = lamp
            a = QLabel(addr); a.setObjectName('Mono')
            r.addWidget(lamp); r.addWidget(a); r.addWidget(L(tkey)); r.addStretch(); li.addLayout(r)
        rb = QHBoxLayout()
        rb.addWidget(B('start', 'Start', ctrl.cmd_start)); rb.addWidget(B('stop', 'Stop', ctrl.cmd_stop)); li.addLayout(rb)
        root.addWidget(fi, 0, 0)
        fo, lo, _ = card('outputs', '➡'); self.out_lamps = {}
        for addr, key, tkey, col in self.OUTPUTS:
            r = QHBoxLayout(); lamp = Lamp(col); self.out_lamps[key] = lamp
            lamp.clicked.connect(lambda k=key: self.force(k))
            a = QLabel(addr); a.setObjectName('Mono')
            r.addWidget(lamp); r.addWidget(a); r.addWidget(L(tkey)); r.addStretch(); lo.addLayout(r)
        lo.addWidget(L('force_hint', 'Sub'))
        root.addWidget(fo, 0, 1)
        fd, ld, _ = card('db_vision', '🗄')
        self.db = QTableWidget(len(self.DB), 4); self.db.setHorizontalHeaderLabels(['Name', 'Type', 'Writer', 'Value'])
        self.db.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch); self.db.verticalHeader().setVisible(False)
        self.db.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for i, (n, t, w) in enumerate(self.DB):
            for j, v in enumerate((n, t, w, '')):
                self.db.setItem(i, j, QTableWidgetItem(v))
        ld.addWidget(self.db); root.addWidget(fd, 1, 0)
        fn, ln, _ = card('networks', '🪜'); self.nw = {}
        for k in range(1, 6):
            lab = L(f'n{k}'); lab.setWordWrap(True); self.nw[k] = lab; ln.addWidget(lab)
        self.cycle = QLabel(); self.cycle.setObjectName('Sub'); ln.addWidget(self.cycle)
        root.addWidget(fn, 1, 1)
        ft, lt, _ = card('trace', '📟'); self.trace = SignalTrace(ctrl); lt.addWidget(self.trace)
        h = L('trace_hint', 'Sub'); h.setWordWrap(True); lt.addWidget(h)
        root.addWidget(ft, 2, 0, 1, 2)
        root.setRowStretch(1, 1); root.setRowStretch(2, 1)
        ctrl.tick.connect(self.on_tick)

    def force(self, key):
        plc = self.ctrl.plc
        if plc.kind == 'sim' and plc.manual and key in plc.force:
            plc.force[key] = not plc.force[key]

    def on_tick(self):
        if not self.isVisible():
            return
        plc = self.ctrl.plc
        for k, lamp in self.in_lamps.items():
            lamp.set(bool(plc.I[k]))
        for k, lamp in self.out_lamps.items():
            lamp.set(bool(plc.Q[k]))
        for i, (n, t, w) in enumerate(self.DB):
            v = plc.DB[n]
            txt = ('TRUE' if v else 'FALSE') if t == 'Bool' else (f'{v:.3f}'.replace('.', ',') if t == 'Real' else str(v))
            if n == 'DefectClass' and self.ctrl.model.ready and 0 <= v < len(self.ctrl.model.names):
                txt += f'  ({tr.cls(self.ctrl.model.names[v])})'
            it = self.db.item(i, 3); it.setText(txt)
            it.setForeground(QColor(GREEN if (t == 'Bool' and v) else TH['text']))
        now = time.time()
        for k, lab in self.nw.items():
            act = now - plc.last_active[k] < 0.6
            lab.setStyleSheet('padding:8px; border-radius:6px; ' + (f'background:{ORANGE}; color:white; font-weight:700;' if act
                                                                     else f'background:{TH["header"]};'))
        src = tr.t('plc_opc') if plc.kind == 'opc' else tr.t('plc_sim')
        mode = tr.t('mode_' + self.ctrl.mode)
        self.cycle.setText(f'{tr.t("scan_time")}: {plc.cycle_ms:.3f} ms · {tr.t("plc_scans")}: {plc.scans} · {src} · {mode}'.replace('.', ','))
        self.trace.update()


# =====================================================================
# Event log
# =====================================================================
class LogWindow(BaseWindow):
    title_key = 'win_log'
    TYPES = ['INFO', 'AI', 'PLC', 'ALARM']
    COLORS = {'INFO': None, 'AI': BLUE, 'PLC': ORANGE, 'ALARM': RED}

    def __init__(self, ctrl):
        super().__init__(ctrl); self.resize(1080, 640)
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12)
        top = QHBoxLayout(); top.addWidget(L('filter'))
        self.flt = QComboBox(); self.flt.currentIndexChanged.connect(self.reload); top.addWidget(self.flt)
        self.search = QLineEdit(); self.search.textChanged.connect(self.reload); top.addWidget(self.search, 1)
        self.count = QLabel(); self.count.setObjectName('Sub'); top.addWidget(self.count)
        top.addWidget(B('clear', None, self.clear)); top.addWidget(B('export_csv', None, self.export)); root.addLayout(top)
        self.table = QTableWidget(0, 3); self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False); self.table.setAlternatingRowColors(True); root.addWidget(self.table)
        ctrl.logged.connect(self.add)
        self.retranslate()

    def retranslate(self):
        super().retranslate()
        cur = self.flt.currentIndex(); self.flt.blockSignals(True); self.flt.clear()
        self.flt.addItems([tr.t('all')] + [tr.t('ev_' + t.lower()) for t in self.TYPES])
        self.flt.setCurrentIndex(max(0, cur)); self.flt.blockSignals(False)
        self.search.setPlaceholderText('🔍 ' + tr.t('search'))
        self.table.setHorizontalHeaderLabels([tr.t('col_time'), tr.t('col_type'), tr.t('col_msg')])
        self.reload()

    def _ok(self, typ, msg):
        i = self.flt.currentIndex()
        q = self.search.text().strip().lower()
        return (i <= 0 or self.TYPES[i - 1] == typ) and (not q or q in msg.lower())

    def add(self, t, typ, msg):
        if not self._ok(typ, msg):
            return
        self.table.insertRow(0)
        for j, v in enumerate((t, tr.t('ev_' + typ.lower()), msg)):
            it = QTableWidgetItem(v); it.setForeground(QColor(self.COLORS.get(typ) or TH['text'])); self.table.setItem(0, j, it)
        if self.table.rowCount() > 2000:
            self.table.removeRow(self.table.rowCount() - 1)
        self.count.setText(f'{self.table.rowCount()} / {len(self.ctrl.events)}')

    def reload(self):
        self.table.setRowCount(0)
        for t, typ, msg in self.ctrl.events[-2000:]:
            self.add(t, typ, msg)
        self.count.setText(f'{self.table.rowCount()} / {len(self.ctrl.events)}')

    def clear(self):
        self.ctrl.events.clear(); self.table.setRowCount(0); self.count.setText('0')

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, tr.t('export_csv'), f'zurnals_{datetime.datetime.now():%Y%m%d_%H%M}.csv', 'CSV (*.csv)')
        if path:
            with open(path, 'w', encoding='utf-8-sig') as f:
                f.write('time;type;message\n')
                for t, typ, msg in self.ctrl.events:
                    f.write(f'{t};{typ};{msg}\n')
            self.ctrl.log('INFO', tr.t('log_export', f=path))


# =====================================================================
# Settings
# =====================================================================
class SettingsWindow(BaseWindow):
    title_key = 'win_settings'

    def __init__(self, ctrl):
        super().__init__(ctrl); self.resize(820, 820)
        s = ctrl.settings
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(10)
        f1, l1, _ = card('s_model', '🧠')
        r = QHBoxLayout(); self.model_dir = QLineEdit(s['model_dir']); r.addWidget(self.model_dir, 1)
        r.addWidget(B('s_browse', None, self.browse)); l1.addLayout(r)
        self.model_info = QLabel(); self.model_info.setObjectName('Sub'); self.model_info.setWordWrap(True); l1.addWidget(self.model_info)
        root.addWidget(f1)
        f2, l2, _ = card('s_threshold', '🎚')
        r = QHBoxLayout(); self.thr = QDoubleSpinBox(); self.thr.setRange(0.01, 0.99); self.thr.setDecimals(3); self.thr.setSingleStep(0.01)
        self.thr.setValue(ctrl.threshold); r.addWidget(self.thr)
        r.addWidget(B('s_thr_model', None, self.reset_thr)); r.addStretch()
        l2.addLayout(r); h = L('s_threshold_hint', 'Sub'); h.setWordWrap(True); l2.addWidget(h)
        form = QFormLayout()
        self.spacing = QDoubleSpinBox(); self.spacing.setRange(0.12, 0.6); self.spacing.setSingleStep(0.02); self.spacing.setValue(s['spacing'])
        form.addRow(L('s_interval'), self.spacing); l2.addLayout(form)
        root.addWidget(f2)
        f4, l4, _ = card('s_alarms', '🚨')
        form = QFormLayout()
        self.a_rate = QDoubleSpinBox(); self.a_rate.setRange(5, 100); self.a_rate.setSuffix(' %'); self.a_rate.setValue(float(s['alarm_rate']))
        self.a_consec = QSpinBox(); self.a_consec.setRange(2, 50); self.a_consec.setValue(int(s['alarm_consec']))
        self.a_lat = QDoubleSpinBox(); self.a_lat.setRange(10, 10000); self.a_lat.setSuffix(' ms'); self.a_lat.setValue(float(s['alarm_latency']))
        self.a_drift = QDoubleSpinBox(); self.a_drift.setRange(0.01, 0.5); self.a_drift.setSingleStep(0.01); self.a_drift.setDecimals(2)
        self.a_drift.setValue(float(s['drift_margin']))
        form.addRow(L('s_al_rate'), self.a_rate); form.addRow(L('s_al_consec'), self.a_consec)
        form.addRow(L('s_al_lat'), self.a_lat); form.addRow(L('s_al_drift'), self.a_drift)
        l4.addLayout(form); root.addWidget(f4)
        f3, l3, _ = card('s_opc', '🔌')
        form = QFormLayout()
        self.ep = QLineEdit(s['opc_endpoint']); self.ns = QSpinBox(); self.ns.setRange(0, 20); self.ns.setValue(int(s['opc_ns']))
        self.dbn = QLineEdit(s['opc_db'])
        form.addRow(L('s_endpoint'), self.ep); form.addRow(L('s_ns'), self.ns); form.addRow(L('s_db'), self.dbn)
        l3.addLayout(form)
        self.nodes_hint = QLabel(); self.nodes_hint.setObjectName('Sub'); self.nodes_hint.setWordWrap(True); l3.addWidget(self.nodes_hint)
        l3.addWidget(B('s_test', None, self.test_opc))
        root.addWidget(f3)
        row = QHBoxLayout(); row.addStretch(); row.addWidget(B('save', 'Accent', self.save)); row.addWidget(B('close', None, self.hide))
        root.addLayout(row); root.addStretch()
        ctrl.state_changed.connect(self.refresh_info)
        self.refresh_info()

    def refresh_info(self):
        m = self.ctrl.model
        if m.ready:
            tm = self.ctrl.test_metrics()
            self.model_info.setText(f'{m.meta.get("model")} · {m.size}×{m.size} · F1 {fnum(tm.get("f1"), 3)} · '
                                    f'AUC {fnum(tm.get("auc"), 3)} · demo: {len(m.demo)} · {tr.t("m_thr_def")}: {m.default_threshold:.3f}'.replace('.', ','))
        else:
            self.model_info.setText(tr.t('no_model'))
        self.thr.blockSignals(True); self.thr.setValue(self.ctrl.threshold); self.thr.blockSignals(False)
        db = self.dbn.text() or 'DB_Vision'
        self.nodes_hint.setText(f'ns={self.ns.value()};s="{db}"."ImageRequest" … HmiStart, HmiStop, HmiEStop, PartSensor, ImageRequest, '
                                'ResultReady, Defect, DefectClass, Confidence, GoodCount, BadCount, Motor, Pusher, LampGreen, LampRed, Alarm')

    def browse(self):
        d = QFileDialog.getExistingDirectory(self, tr.t('s_model'), self.model_dir.text())
        if d:
            self.model_dir.setText(d)

    def reset_thr(self):
        if self.ctrl.model.ready:
            self.thr.setValue(self.ctrl.model.default_threshold)

    def test_opc(self):
        try:
            from asyncua.sync import Client
        except ImportError:
            QMessageBox.warning(self, tr.t('err_title'), tr.t('s_opc_missing')); return
        try:
            cl = Client(self.ep.text(), timeout=3)
            try:
                cl.connect(); cl.disconnect()
            except Exception:
                try:
                    cl.tloop.stop()
                except Exception:
                    pass
                raise
            QMessageBox.information(self, 'OPC UA', tr.t('s_conn_ok'))
        except Exception as e:
            QMessageBox.warning(self, 'OPC UA', f'{tr.t("s_conn_fail")}\n{e}')

    def save(self):
        s = self.ctrl.settings
        new_dir = self.model_dir.text()
        s.update(spacing=self.spacing.value(), opc_endpoint=self.ep.text(), opc_ns=self.ns.value(), opc_db=self.dbn.text(),
                 alarm_rate=self.a_rate.value(), alarm_consec=self.a_consec.value(), alarm_latency=self.a_lat.value(),
                 drift_margin=self.a_drift.value())
        if new_dir != s['model_dir'] or not self.ctrl.model.ready:
            s['model_dir'] = new_dir
            if self.ctrl.load_model(new_dir):
                self.ctrl.sim.reset()
        md = self.ctrl.model.default_threshold if self.ctrl.model.ready else None
        s['threshold'] = None if md is not None and abs(self.thr.value() - md) < 1e-6 else self.thr.value()
        save_settings(s); self.refresh_info()
        self.ctrl.log('INFO', tr.t('s_saved')); self.ctrl.state_changed.emit()


class AboutDialog(QDialog):
    def __init__(self, ctrl, parent=None):
        super().__init__(parent); self.setWindowTitle(tr.t('win_about')); self.resize(680, 520)
        lay = QVBoxLayout(self); lay.setContentsMargins(18, 18, 18, 18)
        t = QLabel('🧵 ' + tr.t('app_title')); t.setObjectName('Title'); lay.addWidget(t)
        a = QLabel(tr.t('about_text')); a.setWordWrap(True); lay.addWidget(a)
        f = QLabel(tr.t('about_features')); f.setWordWrap(True); f.setObjectName('Sub'); lay.addWidget(f)
        if ctrl.model.ready:
            m = ctrl.model.meta; tm = ctrl.test_metrics()
            info = QLabel(f'{m.get("model")} · {m.get("created", "")}\n'
                          f'Accuracy {fpct(tm.get("acc"))} · TPR {fpct(tm.get("recall"))} · FPR {fpct(tm.get("fpr"))} · '
                          f'AUC {fnum(tm.get("auc"), 3)} · {tr.t("threshold")} {fnum(m.get("defect_threshold", 0), 3)}')
            info.setObjectName('Sub'); lay.addWidget(info)
        keys = QLabel(tr.t('shortcuts')); keys.setObjectName('Sub'); keys.setWordWrap(True); lay.addWidget(keys)
        lay.addStretch(); b = QPushButton(tr.t('close')); b.clicked.connect(self.accept); lay.addWidget(b)


# =====================================================================
# Main window
# =====================================================================
class MainWindow(QMainWindow):
    NAV_ICONS = {'win_inspect': '🎥', 'win_sort': '📦', 'win_stats': '📊', 'win_plc': '⚙', 'win_alarms': '🚨',
                 'win_batch': '🗂', 'win_log': '📝', 'win_settings': '🛠'}

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.rate_hist, self.lat_hist, self.thr_hist = [], [], []
        self.windows = {'win_inspect': InspectionWindow(ctrl), 'win_sort': SortingWindow(ctrl), 'win_stats': StatsWindow(ctrl),
                        'win_plc': PlcWindow(ctrl), 'win_alarms': AlarmWindow(ctrl), 'win_batch': BatchWindow(ctrl),
                        'win_log': LogWindow(ctrl), 'win_settings': SettingsWindow(ctrl)}
        self.resize(1560, 940)
        central = QWidget(); page(central); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(14, 10, 14, 10); root.setSpacing(10)
        # ---- header
        head = QHBoxLayout()
        logo = QLabel('🧵'); logo.setStyleSheet('font-size:30pt;'); head.addWidget(logo)
        tl = QVBoxLayout(); tl.setSpacing(0); tl.addWidget(L('app_title', 'Title')); tl.addWidget(L('app_sub', 'Sub')); head.addLayout(tl); head.addStretch()
        self.clock = QLabel(); self.clock.setObjectName('Big'); head.addWidget(self.clock); head.addSpacing(16)
        self.cb_lang = QComboBox(); self.cb_lang.addItems(list(LANGS.values()))
        self.cb_lang.setCurrentIndex(list(LANGS).index(tr.lang)); self.cb_lang.currentIndexChanged.connect(lambda i: ctrl.set_lang(list(LANGS)[i]))
        self.cb_mode = QComboBox(); self.cb_mode.currentIndexChanged.connect(lambda i: ctrl.set_mode(['auto', 'manual'][i]) if i >= 0 else None)
        self.cb_plc = QComboBox(); self.cb_plc.activated.connect(self.on_plc_source)
        for key, cb in (('language', self.cb_lang), ('mode', self.cb_mode), ('plc_source', self.cb_plc)):
            box = QVBoxLayout(); box.setSpacing(2); box.addWidget(L(key, 'Sub')); box.addWidget(cb); head.addLayout(box)
        self.b_theme = QPushButton(); self.b_theme.setObjectName('Tool'); self.b_theme.clicked.connect(self.toggle_theme)
        self.b_theme.setFixedSize(46, 46); head.addWidget(self.b_theme, alignment=Qt.AlignmentFlag.AlignBottom)
        root.addLayout(head)
        # ---- alarm banner
        self.banner = QFrame(); self.banner.setObjectName('Alarm')
        bl = QHBoxLayout(self.banner); bl.setContentsMargins(14, 8, 10, 8)
        self.banner_txt = QLabel(); bl.addWidget(self.banner_txt, 1)
        bb = B('open_alarms', None, lambda: self.windows['win_alarms'].toggle()); bl.addWidget(bb)
        ba = B('ack_all', None, lambda: ctrl.ack()); bl.addWidget(ba)
        self.banner.hide(); root.addWidget(self.banner)
        # ---- KPI tiles
        kp = QHBoxLayout(); kp.setSpacing(10); self.k = {}
        for key, col, sp, ic in [('kpi_state', GRAY, False, '●'), ('kpi_total', None, False, '📦'), ('kpi_good', GREEN, False, '✅'),
                                 ('kpi_defect', RED, False, '⛔'), ('kpi_rate', ORANGE, True, '📉'), ('kpi_acc', BLUE, False, '🎯'),
                                 ('throughput', PURPLE, True, '🚀'), ('kpi_lat', None, True, '⏱'), ('oee', GREEN, False, '⚙'),
                                 ('kpi_plc', BLUE, False, '🔌')]:
            self.k[key] = KpiTile(key, col, spark=sp, icon=ic); kp.addWidget(self.k[key])
        root.addLayout(kp)
        body = QHBoxLayout(); body.setSpacing(12); root.addLayout(body, 1)
        # ---- navigation
        nf, nl, _ = card('m_windows', '🧭')
        self.nav = {}
        for k, (key, w) in enumerate(self.windows.items()):
            b = QPushButton(); b.setObjectName('Nav'); b.setCheckable(True); b.setProperty('nkey', key)
            b.setToolTip(f'F{k + 2}')
            b.clicked.connect(lambda _, ww=w: ww.toggle()); nl.addWidget(b); self.nav[key] = b
        nl.addStretch()
        self.b_report = B('quick_report', 'Accent', lambda: self.windows['win_stats'].export_report('pdf')); nl.addWidget(self.b_report)
        nl.addWidget(B('win_about', None, self.about))
        nf.setFixedWidth(330); body.addWidget(nf)
        # ---- centre: conveyor + gauges + timeline
        center = QVBoxLayout(); center.setSpacing(12); body.addLayout(center, 1)
        of, ol, _ = card('conveyor', '🏭'); self.view = ConveyorView(ctrl, compact=True); ol.addWidget(self.view); center.addWidget(of, 3)
        bottom = QHBoxLayout(); bottom.setSpacing(12); center.addLayout(bottom, 2)
        gf, gl, _ = card('live_gauges', '🎛')
        gr = QHBoxLayout()
        self.g_oee = Gauge('oee', 100, '%')
        self.g_rate = Gauge('kpi_rate', 100, '%', invert=True)
        self.g_thr = Gauge('g_thr', 30, '', decimals=1, zones=[(0.33, RED), (0.66, YELLOW), (1.0, GREEN)])
        for g_ in (self.g_oee, self.g_rate, self.g_thr):
            gr.addWidget(g_)
        gl.addLayout(gr); bottom.addWidget(gf, 3)
        tf, tlay, _ = card('state_timeline', '📊'); self.timeline = StateTimeline(ctrl, 300); tlay.addWidget(self.timeline)
        self.durs = QLabel(); self.durs.setObjectName('Sub'); self.durs.setWordWrap(True); tlay.addWidget(self.durs); tlay.addStretch()
        bottom.addWidget(tf, 2)
        # ---- right: control, production order, last result
        right = QVBoxLayout(); right.setSpacing(12); body.addLayout(right)
        cf, cl, _ = card('control', '🕹')
        self.b_start = B('start', 'Start', ctrl.cmd_start); self.b_stop = B('stop', 'Stop', ctrl.cmd_stop)
        self.b_estop = B('estop', 'EStop'); self.b_estop.setCheckable(True); self.b_estop.toggled.connect(self.on_estop)
        for b in (self.b_start, self.b_stop, self.b_estop):
            b.setMinimumHeight(50); cl.addWidget(b)
        right.addWidget(cf)
        pf, pl, _ = card('order', '📋')
        r = QHBoxLayout(); r.addWidget(L('order_target'))
        self.sp_target = QSpinBox(); self.sp_target.setRange(1, 100000); self.sp_target.setValue(int(ctrl.settings['order_target'])); r.addWidget(self.sp_target)
        pl.addLayout(r)
        self.chk_auto = QCheckBox(tr.t('order_autostop')); self.chk_auto.setProperty('tkey', 'order_autostop')
        self.chk_auto.setChecked(bool(ctrl.settings.get('order_autostop', True)))
        self.chk_auto.toggled.connect(lambda v: ctrl.settings.update(order_autostop=v)); pl.addWidget(self.chk_auto)
        r = QHBoxLayout(); self.b_order = B('order_start', 'Accent', self.on_order); self.b_ocancel = B('order_cancel', None, ctrl.order_cancel)
        r.addWidget(self.b_order); r.addWidget(self.b_ocancel); pl.addLayout(r)
        self.pb_order = QProgressBar(); self.pb_order.setObjectName('Order'); pl.addWidget(self.pb_order)
        self.l_order = QLabel('—'); self.l_order.setObjectName('Sub'); self.l_order.setWordWrap(True); pl.addWidget(self.l_order)
        right.addWidget(pf)
        rf, rl, _ = card('ai_result', '🧠')
        self.last_img = QLabel(); self.last_img.setFixedSize(120, 120); self.last_img.setStyleSheet('background:#0B1117; border-radius:8px;')
        rl.addWidget(self.last_img, alignment=Qt.AlignmentFlag.AlignCenter)
        self.last_txt = QLabel('—'); self.last_txt.setWordWrap(True); self.last_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rl.addWidget(self.last_txt); rl.addStretch()
        right.addWidget(rf, 1)
        for f in (cf, pf, rf):
            f.setFixedWidth(290)
        # ---- menu & status bar
        self.build_menu()
        self.sb = QStatusBar(); self.setStatusBar(self.sb); self.sb_lbl = QLabel(); self.sb.addPermanentWidget(self.sb_lbl, 1)
        ctrl.tick.connect(self.on_tick); ctrl.result.connect(self.on_result); ctrl.state_changed.connect(self.retranslate)
        ctrl.alarms_changed.connect(self.update_banner); ctrl.order_changed.connect(self.update_order)
        ctrl.error.connect(lambda m: self.sb.showMessage('⚠ ' + m, 8000))
        self.slow = QTimer(self); self.slow.timeout.connect(self.slow_update); self.slow.start(500)
        self.retranslate(); self.update_order()
        if not ctrl.model.ready:
            QTimer.singleShot(400, lambda: QMessageBox.warning(self, tr.t('err_title'), tr.t('no_model')))

    # ---------------------------------------------------------------- menu
    def build_menu(self):
        mb = self.menuBar(); mb.clear()
        m = mb.addMenu(tr.t('m_file'))
        self._act(m, 'm_open_img', 'Ctrl+O', self.windows['win_inspect'].open_image)
        self._act(m, 'm_batch', 'Ctrl+B', self.windows['win_batch'].toggle)
        m.addSeparator()
        self._act(m, 'm_export', 'Ctrl+E', self.windows['win_stats'].export)
        self._act(m, 'export_html', 'Ctrl+H', lambda: self.windows['win_stats'].export_report('html'))
        self._act(m, 'export_pdf', 'Ctrl+P', lambda: self.windows['win_stats'].export_report('pdf'))
        self._act(m, 'm_screenshot', 'Ctrl+Shift+S', self.screenshot)
        m.addSeparator(); self._act(m, 'm_exit', 'Ctrl+Q', self.close)
        m = mb.addMenu(tr.t('m_line'))
        self._act(m, 'start', 'F10', self.ctrl.cmd_start); self._act(m, 'stop', 'F11', self.ctrl.cmd_stop)
        self._act(m, 'estop', 'F12', lambda: self.b_estop.toggle())
        m.addSeparator()
        self._act(m, 'mode_auto', 'Ctrl+1', lambda: self.cb_mode.setCurrentIndex(0))
        self._act(m, 'mode_manual', 'Ctrl+2', lambda: self.cb_mode.setCurrentIndex(1))
        m.addSeparator(); self._act(m, 'ack_all', 'Ctrl+K', lambda: self.ctrl.ack())
        self._act(m, 'reset_stats', None, self.windows['win_stats'].reset)
        m = mb.addMenu(tr.t('m_windows'))
        for k, (key, w) in enumerate(self.windows.items()):
            self._act(m, key, f'F{k + 2}', w.toggle)
        m.addSeparator(); self._act(m, 'm_all_windows', 'Ctrl+Shift+W', self.show_all)
        m = mb.addMenu(tr.t('m_view'))
        self._act(m, 'm_theme', 'Ctrl+T', self.toggle_theme)
        self._act(m, 'm_fullscreen', 'Ctrl+F', self.toggle_full)
        m = mb.addMenu(tr.t('m_help')); self._act(m, 'win_about', 'F1', self.about)

    def _act(self, menu, key, sc, slot):
        a = QAction(tr.t(key), self)
        if sc:
            a.setShortcut(QKeySequence(sc))
        a.triggered.connect(slot); menu.addAction(a); return a

    # ---------------------------------------------------------------- language / theme
    def retranslate(self):
        self.setWindowTitle(tr.t('app_title')); retranslate_tree(self.centralWidget()); retranslate_cards(self.centralWidget())
        self.cb_lang.blockSignals(True); self.cb_lang.setCurrentIndex(list(LANGS).index(tr.lang)); self.cb_lang.blockSignals(False)
        for kt in self.k.values():
            kt.retranslate()
        for key, b in self.nav.items():
            b.setText(f'{self.NAV_ICONS[key]}   {tr.t(key)}')
        self.cb_mode.blockSignals(True); self.cb_mode.clear(); self.cb_mode.addItems([tr.t('mode_auto'), tr.t('mode_manual')])
        self.cb_mode.setCurrentIndex(0 if self.ctrl.mode == 'auto' else 1); self.cb_mode.blockSignals(False)
        self.cb_plc.blockSignals(True); self.cb_plc.clear(); self.cb_plc.addItems([tr.t('plc_sim'), tr.t('plc_opc')])
        self.cb_plc.setCurrentIndex(1 if self.ctrl.plc.kind == 'opc' else 0); self.cb_plc.blockSignals(False)
        self.b_estop.setText(tr.t('estop_reset') if self.b_estop.isChecked() else tr.t('estop'))
        self.b_theme.setText('☀' if TH.dark else '🌙'); self.b_theme.setToolTip(tr.t('m_theme'))
        if TH.name != self.ctrl.settings.get('theme', 'dark'):
            self.apply_theme(self.ctrl.settings.get('theme', 'dark'))
        self.build_menu(); self.update_banner(); self.update_order()
        if self.ctrl.last_result:
            self.on_result(self.ctrl.last_result)
        for g_ in self.findChildren(QWidget):
            g_.update()

    def apply_theme(self, name):
        TH.set(name)
        QApplication.instance().setStyleSheet(build_qss())
        restyle_plots()
        self.b_theme.setText('☀' if TH.dark else '🌙')
        for w in [self] + list(self.windows.values()):
            for c in w.findChildren(QWidget):
                c.update()

    def toggle_theme(self):
        self.ctrl.set_theme('light' if TH.dark else 'dark')

    def toggle_full(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def show_all(self):
        for w in self.windows.values():
            w.show()

    def screenshot(self):
        path, _ = QFileDialog.getSaveFileName(self, tr.t('m_screenshot'), f'ekrans_{datetime.datetime.now():%Y%m%d_%H%M%S}.png', 'PNG (*.png)')
        if path:
            self.grab().save(path); self.ctrl.log('INFO', tr.t('log_export', f=path))

    # ---------------------------------------------------------------- actions
    def on_plc_source(self, i):
        self.ctrl.use_opc(i == 1)

    def on_estop(self, pressed):
        self.b_estop.setText(tr.t('estop_reset') if pressed else tr.t('estop'))
        self.ctrl.cmd_estop(pressed)

    def on_order(self):
        self.ctrl.settings['order_autostop'] = self.chk_auto.isChecked()
        self.ctrl.order_start(self.sp_target.value())

    def about(self):
        AboutDialog(self.ctrl, self).exec()

    # ---------------------------------------------------------------- updates
    def update_banner(self):
        al = self.ctrl.unacked()
        if al:
            act = [a for a in al if a['active']]
            first = (act or al)[0]
            more = f'   (+{len(al) - 1})' if len(al) > 1 else ''
            self.banner_txt.setText(f'⚠  {first["time"]}  ·  {tr.t("al_" + first["key"], **first["kw"])}{more}')
            self.banner.show()
        else:
            self.banner.hide()

    def update_order(self):
        done, target, bad, eta = self.ctrl.order_progress()
        self.pb_order.setRange(0, max(1, target)); self.pb_order.setValue(done)
        self.pb_order.setFormat(f'{done} / {target}')
        if self.ctrl.order_active:
            self.l_order.setText(tr.t('order_running', b=bad, eta=fdur(eta)))
        elif self.ctrl.order_t0 is not None and done >= target:
            self.l_order.setText('✔ ' + tr.t('order_done'))
        else:
            self.l_order.setText(tr.t('order_idle'))
        self.b_ocancel.setEnabled(self.ctrl.order_active)

    def on_tick(self):
        c = self.ctrl; plc = c.plc
        self.view.update()
        st = c.line_state()
        col = {'estop': RED, 'manual': BLUE, 'inspect': YELLOW, 'run': GREEN, 'stop': GRAY}[st]
        self.k['kpi_state'].set(tr.t('st_' + {'run': 'running', 'stop': 'stopped', 'inspect': 'inspect', 'estop': 'estop', 'manual': 'manual'}[st]), col)
        for key, b in self.nav.items():
            vis = self.windows[key].isVisible()
            if b.isChecked() != vis:
                b.setChecked(vis)

    def slow_update(self):
        c = self.ctrl; plc = c.plc; st = c.stats
        s = st.summary(c.names); oee = c.oee()
        self.clock.setText(datetime.datetime.now().strftime('%H:%M:%S'))
        self.k['kpi_total'].set(str(s['n'])); self.k['kpi_good'].set(str(s['good'])); self.k['kpi_defect'].set(str(s['bad']))
        self.k['kpi_rate'].set(fnum(s['rate'], 1, ' %')); self.k['kpi_rate'].set_spark(st.rolling_rate(20)[1])
        self.k['kpi_acc'].set(fpct(s['acc']))
        self.k['throughput'].set(fnum(s['throughput'], 1)); self.k['throughput'].set_spark(st.throughput_series(15)[1])
        self.k['kpi_lat'].set('—' if s['latency'] is None else f'{s["latency"]:.0f} ms')
        self.k['kpi_lat'].set_spark([r['latency_ms'] for r in st.records[-60:]])
        self.k['oee'].set(fpct(oee['OEE']))
        link = tr.t('link_ok') if plc.kind == 'opc' else tr.t('link_sim')
        self.k['kpi_plc'].set(link, GREEN if plc.kind == 'opc' else BLUE)
        self.g_oee.set(None if oee['OEE'] is None else oee['OEE'] * 100)
        self.g_rate.set(s['rolling'] if s['rolling'] is not None else None)
        ideal = c.ideal_rate() * 60
        self.g_thr.vmax = max(5.0, round(ideal * 1.2)); self.g_thr.set(s['throughput'] if s['n'] else None)
        self.timeline.update()
        d = c.states.dur
        self.durs.setText('   '.join(f'{tr.t("state_" + k)}: {fdur(d[k])}' for k in ('run', 'inspect', 'stop', 'estop', 'manual')))
        if c.unacked() and int(time.time() * 2) % 2 == 0:
            self.banner.setStyleSheet('QFrame#Alarm { background: #8E0000; }')
        else:
            self.banner.setStyleSheet('')
        model = c.model.meta.get('model', '—') if c.model.ready else '—'
        self.sb_lbl.setText(f'  🕒 {datetime.datetime.now():%d.%m.%Y %H:%M:%S}   ·   🧠 AI: {model}   ·   {tr.t("threshold")}: {c.threshold:.3f}   ·   '
                            f'⚙ {tr.t("scan_time")}: {plc.cycle_ms:.2f} ms   ·   {tr.t("mode")}: {tr.t("mode_" + c.mode)}   ·   '
                            f'🚨 {len(c.unacked())}   ·   {tr.t("sess_time")}: {fdur(time.time() - st.t0)}'.replace('.', ','))
        if c.order_active:
            self.update_order()

    def on_result(self, r):
        self.last_img.setPixmap(QPixmap(r['path']).scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
        defect = r.get('decision_defect', r['defect'])
        self.last_txt.setText(f'<b style="color:{RED if defect else GREEN}; font-size:13pt">{tr.cls(r["name"])}</b><br>'
                              f'{r["confidence"] * 100:.1f} % · P(def) {r["p_defect"]:.3f}'.replace('.', ','))

    def closeEvent(self, e):
        for w in self.windows.values():
            w.close()
        self.ctrl.shutdown(); e.accept()
