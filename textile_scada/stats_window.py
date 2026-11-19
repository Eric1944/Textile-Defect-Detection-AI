"""Statistics dashboard: 5 tabs (overview / OEE, AI quality, defects, performance, model monitoring – MLOps)
and session report export (CSV, HTML, PDF, PNG)."""
import base64, datetime, time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, QBuffer, QIODevice, QUrl, QSizeF, QMarginsF
from PyQt6.QtGui import QColor, QTextDocument, QPdfWriter, QPageSize, QPageLayout, QImage
from PyQt6.QtWidgets import (QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGridLayout, QTabWidget, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView, QSlider, QFileDialog, QApplication,
                             QMessageBox, QDoubleSpinBox)

from .i18n import tr
from .core import STATES
from .theme import TH, style_plot, ORANGE, GREEN, RED, BLUE, YELLOW, GRAY, PURPLE, TEAL, CLASS_COLORS, STATE_COLORS
from .widgets import (BaseWindow, card, KpiTile, Gauge, Donut, StateTimeline, L, B, fnum, fpct, fdur, page)


def mkplot(min_h=200):
    pw = pg.PlotWidget(); pw.setMinimumHeight(min_h); pw.setMenuEnabled(False)
    return style_plot(pw)


def table(rows, cols, stretch=True):
    t = QTableWidget(rows, cols); t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.verticalHeader().setVisible(False); t.setAlternatingRowColors(True)
    t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    t.verticalHeader().setDefaultSectionSize(25)
    if stretch:
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    return t


def set_cell(t, i, j, text, color=None, bg=None, bold=False, center=True):
    it = t.item(i, j)
    if it is None:
        it = QTableWidgetItem(); t.setItem(i, j, it)
    it.setText(text)
    if center:
        it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    it.setForeground(QColor(color or TH['text']))
    it.setBackground(QColor(bg) if bg is not None else QColor(0, 0, 0, 0))
    f = it.font(); f.setBold(bool(bold)); it.setFont(f)


class StatsWindow(BaseWindow):
    title_key = 'win_stats'
    KPIS = [('kpi_total', None, False), ('kpi_good', GREEN, False), ('kpi_defect', RED, False), ('kpi_rate', ORANGE, True),
            ('kpi_acc', BLUE, False), ('recall_def', BLUE, False), ('precision_def', BLUE, False),
            ('f1_def', PURPLE, False), ('specificity', TEAL, False), ('auc', PURPLE, False),
            ('throughput', None, True), ('lat_mean', None, True), ('lat_p95', None, False), ('oee', GREEN, True)]

    def __init__(self, ctrl):
        super().__init__(ctrl); self.resize(1500, 960)
        self.axis_labels = []          # (plot, axis, key)
        self.oee_hist = []
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(10)
        tiles = QGridLayout(); tiles.setSpacing(8); self.t = {}
        for i, (k, col, sp) in enumerate(self.KPIS):
            self.t[k] = KpiTile(k, col, spark=sp); tiles.addWidget(self.t[k], i // 7, i % 7)
        root.addLayout(tiles)
        self.tabs = QTabWidget(); root.addWidget(self.tabs, 1)
        self.tab_keys = ['tab_overview', 'tab_ai', 'tab_defects', 'tab_perf', 'tab_model']
        builders = [self._build_overview, self._build_ai, self._build_defects, self._build_perf, self._build_model]
        for key, build in zip(self.tab_keys, builders):
            w = QWidget(); page(w); build(w); self.tabs.addTab(w, tr.t(key))
        row = QHBoxLayout()
        self.info = QLabel(); self.info.setObjectName('Sub'); row.addWidget(self.info); row.addStretch()
        row.addWidget(B('export_csv', None, self.export))
        row.addWidget(B('export_html', None, lambda: self.export_report('html')))
        row.addWidget(B('export_pdf', 'Accent', lambda: self.export_report('pdf')))
        row.addWidget(B('export_png', None, self.export_png))
        row.addWidget(B('reset_stats', None, self.reset))
        root.addLayout(row)
        self.tabs.currentChanged.connect(lambda _: self.refresh(force=True))
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(700)
        self.retranslate()

    # ------------------------------------------------------------------ tabs
    def _ax(self, pw, axis, key):
        self.axis_labels.append((pw, axis, key))

    def _build_overview(self, w):
        g = QGridLayout(w); g.setContentsMargins(10, 10, 10, 10); g.setSpacing(10)
        f, l, _ = card('oee_title', '⚙')
        row = QHBoxLayout()
        self.g_oee = Gauge('oee', 100, '%'); self.g_a = Gauge('availability', 100, '%')
        self.g_p = Gauge('performance', 100, '%'); self.g_q = Gauge('quality', 100, '%')
        for x in (self.g_oee, self.g_a, self.g_p, self.g_q):
            row.addWidget(x)
        l.addLayout(row)
        self.oee_info = QLabel(); self.oee_info.setObjectName('Sub'); self.oee_info.setWordWrap(True); l.addWidget(self.oee_info)
        g.addWidget(f, 0, 0, 1, 2)
        f, l, _ = card('state_durations', '⏱'); self.d_states = Donut('seconds'); l.addWidget(self.d_states); g.addWidget(f, 0, 2)
        f, l, _ = card('st_rate_time', '📉'); self.p_rate = mkplot(); self.p_rate.addLegend(offset=(10, 10))
        self.c_roll = self.p_rate.plot([], [], pen=pg.mkPen(ORANGE, width=2), name='20')
        self.c_cum = self.p_rate.plot([], [], pen=pg.mkPen(RED, width=3), name='Σ')
        self.lim_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(RED, width=1, style=Qt.PenStyle.DotLine)); self.p_rate.addItem(self.lim_line)
        self._ax(self.p_rate, 'bottom', 'st_items'); self._ax(self.p_rate, 'left', 'pct')
        l.addWidget(self.p_rate); g.addWidget(f, 1, 0)
        f, l, _ = card('st_throughput_time', '🚀'); self.p_thr = mkplot()
        self.b_thr = pg.BarGraphItem(x=[], height=[], width=12, brush=pg.mkBrush(BLUE), pen=pg.mkPen(None)); self.p_thr.addItem(self.b_thr)
        self.ideal_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(GREEN, width=2, style=Qt.PenStyle.DashLine)); self.p_thr.addItem(self.ideal_line)
        self._ax(self.p_thr, 'bottom', 'st_time'); self._ax(self.p_thr, 'left', 'pcs_min')
        l.addWidget(self.p_thr); g.addWidget(f, 1, 1)
        f, l, _ = card('st_session', '🗂'); self.t_sess = table(10, 2)
        self.t_sess.horizontalHeader().setVisible(False); l.addWidget(self.t_sess); g.addWidget(f, 1, 2)
        f, l, _ = card('state_timeline', '📊'); self.timeline = StateTimeline(self.ctrl, 600); l.addWidget(self.timeline)
        g.addWidget(f, 2, 0, 1, 3)
        g.setRowStretch(1, 1)

    def _build_ai(self, w):
        g = QGridLayout(w); g.setContentsMargins(10, 10, 10, 10); g.setSpacing(10)
        f, l, _ = card('st_cm', '🧮')
        self.cm = QTableWidget(5, 5); self.cm.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cm.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cm.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cm.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.cm.setStyleSheet('QHeaderView::section { font-size: 8pt; padding: 3px; }')
        l.addWidget(self.cm); g.addWidget(f, 0, 0)
        f, l, _ = card('per_class', '📋'); self.pc = table(5, 7); l.addWidget(self.pc)
        self.pc.setStyleSheet('QHeaderView::section { font-size: 8pt; padding: 3px; }')
        self.pc.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.macro = QLabel(); self.macro.setObjectName('Sub'); l.addWidget(self.macro); g.addWidget(f, 0, 1)
        f, l, _ = card('whatif', '🎚')
        h = L('whatif_hint', 'Sub'); h.setWordWrap(True); l.addWidget(h)
        r = QHBoxLayout()
        self.wi_slider = QSlider(Qt.Orientation.Horizontal); self.wi_slider.setRange(1, 99)
        self.wi_spin = QDoubleSpinBox(); self.wi_spin.setRange(0.01, 0.99); self.wi_spin.setDecimals(3); self.wi_spin.setSingleStep(0.01)
        self.wi_slider.valueChanged.connect(lambda v: self.wi_spin.setValue(v / 100))
        self.wi_spin.valueChanged.connect(self._whatif_changed)
        r.addWidget(self.wi_slider, 1); r.addWidget(self.wi_spin); l.addLayout(r)
        self.wi = table(8, 3); self.wi.horizontalHeader().setVisible(True); self.wi.verticalHeader().setDefaultSectionSize(22)
        self.wi.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents); l.addWidget(self.wi, 1)
        rb = QHBoxLayout(); rb.addWidget(B('wi_current', None, self._wi_current)); rb.addWidget(B('wi_apply', 'Accent', self._wi_apply)); l.addLayout(rb)
        g.addWidget(f, 0, 2)
        f, l, _ = card('roc', '📈'); self.p_roc = mkplot(220)
        self.p_roc.plot([0, 1], [0, 1], pen=pg.mkPen(GRAY, width=1, style=Qt.PenStyle.DashLine))
        self.c_roc = self.p_roc.plot([], [], pen=pg.mkPen(PURPLE, width=3), fillLevel=0, brush=pg.mkBrush(142, 36, 170, 40))
        self.s_roc = pg.ScatterPlotItem(size=12); self.p_roc.addItem(self.s_roc)
        self.p_roc.setXRange(0, 1); self.p_roc.setYRange(0, 1.02)
        self._ax(self.p_roc, 'bottom', 'fpr_axis'); self._ax(self.p_roc, 'left', 'tpr_axis')
        self.l_roc = QLabel(); self.l_roc.setObjectName('Sub'); l.addWidget(self.p_roc); l.addWidget(self.l_roc); g.addWidget(f, 1, 0)
        f, l, _ = card('prc', '📈'); self.p_pr = mkplot(220)
        self.c_pr = self.p_pr.plot([], [], pen=pg.mkPen(TEAL, width=3), fillLevel=0, brush=pg.mkBrush(0, 137, 123, 40))
        self.s_pr = pg.ScatterPlotItem(size=12); self.p_pr.addItem(self.s_pr)
        self.p_pr.setXRange(0, 1); self.p_pr.setYRange(0, 1.02)
        self._ax(self.p_pr, 'bottom', 'recall_axis'); self._ax(self.p_pr, 'left', 'precision_axis')
        self.l_pr = QLabel(); self.l_pr.setObjectName('Sub'); l.addWidget(self.p_pr); l.addWidget(self.l_pr); g.addWidget(f, 1, 1)
        f, l, _ = card('hist_pdef', '📊'); self.p_hist = mkplot(220); self.p_hist.addLegend(offset=(10, 10))
        self.h_good = self.p_hist.plot([0, 1], [0], stepMode='center', fillLevel=0, brush=pg.mkBrush(46, 157, 74, 110),
                                       pen=pg.mkPen(GREEN, width=2), name=tr.cls('good'))
        self.h_bad = self.p_hist.plot([0, 1], [0], stepMode='center', fillLevel=0, brush=pg.mkBrush(211, 47, 47, 110),
                                      pen=pg.mkPen(RED, width=2), name='defect')
        self.h_thr = pg.InfiniteLine(angle=90, pen=pg.mkPen(ORANGE, width=2, style=Qt.PenStyle.DashLine)); self.p_hist.addItem(self.h_thr)
        self.h_wi = pg.InfiniteLine(angle=90, pen=pg.mkPen(BLUE, width=1, style=Qt.PenStyle.DotLine)); self.p_hist.addItem(self.h_wi)
        self._ax(self.p_hist, 'bottom', 'p_defect'); self._ax(self.p_hist, 'left', 'count')
        l.addWidget(self.p_hist); g.addWidget(f, 1, 2)
        g.setRowStretch(1, 1)

    def _build_defects(self, w):
        g = QGridLayout(w); g.setContentsMargins(10, 10, 10, 10); g.setSpacing(10)
        f, l, _ = card('pareto', '📊'); self.p_par = mkplot(240)
        self.b_par = pg.BarGraphItem(x=[], height=[], width=0.6, brush=pg.mkBrush(ORANGE), pen=pg.mkPen(None)); self.p_par.addItem(self.b_par)
        pi = self.p_par.getPlotItem(); self.vb2 = pg.ViewBox(); pi.showAxis('right'); pi.scene().addItem(self.vb2)
        pi.getAxis('right').linkToView(self.vb2); self.vb2.setXLink(pi); self.vb2.setYRange(0, 105)
        self.c_par = pg.PlotDataItem([], [], pen=pg.mkPen(RED, width=3), symbol='o', symbolBrush=RED, symbolSize=8); self.vb2.addItem(self.c_par)

        def _upd():
            self.vb2.setGeometry(pi.vb.sceneBoundingRect()); self.vb2.linkedViewChanged(pi.vb, self.vb2.XAxis)
        pi.vb.sigResized.connect(_upd)
        self._ax(self.p_par, 'left', 'count'); self._ax(self.p_par, 'right', 'cum_pct')
        l.addWidget(self.p_par); self.l_par = QLabel(); self.l_par.setObjectName('Sub'); self.l_par.setWordWrap(True); l.addWidget(self.l_par)
        g.addWidget(f, 0, 0)
        f, l, _ = card('defect_share', '🍩'); self.d_def = Donut('kpi_defect'); l.addWidget(self.d_def); g.addWidget(f, 0, 1)
        f, l, _ = card('st_conf_time', '•'); self.p_conf = mkplot(240); self.p_conf.setYRange(0, 1.02)
        self.s_conf = pg.ScatterPlotItem(size=7, pen=pg.mkPen(None)); self.p_conf.addItem(self.s_conf)
        self.thr_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(ORANGE, width=2, style=Qt.PenStyle.DashLine)); self.p_conf.addItem(self.thr_line)
        self._ax(self.p_conf, 'bottom', 'st_items'); self._ax(self.p_conf, 'left', 'p_defect')
        l.addWidget(self.p_conf); g.addWidget(f, 1, 0)
        f, l, _ = card('st_classes', '📊'); self.p_cls = mkplot(240)
        self.b_cls = pg.BarGraphItem(x=list(range(5)), height=[0] * 5, width=0.6, brushes=[pg.mkBrush(c) for c in CLASS_COLORS.values()],
                                     pen=pg.mkPen(None))
        self.p_cls.addItem(self.b_cls); self._ax(self.p_cls, 'left', 'count'); l.addWidget(self.p_cls); g.addWidget(f, 1, 1)

    def _build_perf(self, w):
        g = QGridLayout(w); g.setContentsMargins(10, 10, 10, 10); g.setSpacing(10)
        f, l, _ = card('lat_hist', '⏱'); self.p_lh = mkplot()
        self.h_lat = self.p_lh.plot([0, 1], [0], stepMode='center', fillLevel=0, brush=pg.mkBrush(30, 136, 229, 120), pen=pg.mkPen(BLUE, width=2))
        self.lh_p95 = pg.InfiniteLine(angle=90, pen=pg.mkPen(RED, width=2, style=Qt.PenStyle.DashLine), label='p95',
                                      labelOpts={'color': RED, 'position': 0.9}); self.p_lh.addItem(self.lh_p95)
        self._ax(self.p_lh, 'bottom', 'ms'); self._ax(self.p_lh, 'left', 'count')
        l.addWidget(self.p_lh); g.addWidget(f, 0, 0)
        f, l, _ = card('lat_trend', '📈'); self.p_lt = mkplot(); self.p_lt.addLegend(offset=(10, 10))
        self.c_lat = self.p_lt.plot([], [], pen=pg.mkPen(BLUE, width=1.5), name='ms')
        self.c_latm = self.p_lt.plot([], [], pen=pg.mkPen(ORANGE, width=3), name='MA-10')
        self.lim_lat = pg.InfiniteLine(angle=0, pen=pg.mkPen(RED, width=1, style=Qt.PenStyle.DotLine)); self.p_lt.addItem(self.lim_lat)
        self._ax(self.p_lt, 'bottom', 'st_items'); self._ax(self.p_lt, 'left', 'ms')
        l.addWidget(self.p_lt); g.addWidget(f, 0, 1)
        f, l, _ = card('plc_cycle', '⚙'); self.p_cyc = mkplot()
        self.c_cyc = self.p_cyc.plot([], [], pen=pg.mkPen(TEAL, width=1.5))
        self._ax(self.p_cyc, 'bottom', 'scans'); self._ax(self.p_cyc, 'left', 'ms')
        l.addWidget(self.p_cyc); g.addWidget(f, 1, 0)
        f, l, _ = card('perf_table', '📋'); self.t_perf = table(12, 2); self.t_perf.horizontalHeader().setVisible(False)
        l.addWidget(self.t_perf); g.addWidget(f, 1, 1)

    def _build_model(self, w):
        g = QGridLayout(w); g.setContentsMargins(10, 10, 10, 10); g.setSpacing(10)
        f, l, _ = card('model_info', '🧠'); self.t_model = table(10, 2); self.t_model.horizontalHeader().setVisible(False)
        l.addWidget(self.t_model); g.addWidget(f, 0, 0)
        f, l, _ = card('test_vs_live', '⚖'); self.t_cmp = table(7, 4); l.addWidget(self.t_cmp)
        self.drift = QLabel(); self.drift.setObjectName('Big'); self.drift.setAlignment(Qt.AlignmentFlag.AlignCenter); l.addWidget(self.drift)
        self.drift_sub = QLabel(); self.drift_sub.setObjectName('Sub'); self.drift_sub.setWordWrap(True)
        self.drift_sub.setAlignment(Qt.AlignmentFlag.AlignCenter); l.addWidget(self.drift_sub)
        g.addWidget(f, 0, 1)
        f, l, _ = card('mlops_levels', '🔁')
        self.t_ml = table(3, 3); self.t_ml.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        l.addWidget(self.t_ml)
        h = L('mlops_hint', 'Sub'); h.setWordWrap(True); l.addWidget(h)
        g.addWidget(f, 1, 0, 1, 2)
        g.setRowStretch(0, 3); g.setRowStretch(1, 2)

    # ------------------------------------------------------------------ texts
    def retranslate(self):
        super().retranslate()
        for i, k in enumerate(self.tab_keys):
            self.tabs.setTabText(i, tr.t(k))
        names = self.ctrl.names; labels = [tr.cls(n) for n in names]
        self.p_cls.getAxis('bottom').setTicks([[(i, s) for i, s in enumerate(labels)]])
        self.cm.setHorizontalHeaderLabels(labels); self.cm.setVerticalHeaderLabels(labels)
        for pw, ax, key in self.axis_labels:
            pw.setLabel(ax, tr.t(key))
        self.pc.setHorizontalHeaderLabels([tr.t(k) for k in ('class', 'support', 'n_pred', 'precision_s', 'recall_s', 'f1_s', 'conf_mean')])
        self.wi.setHorizontalHeaderLabels([tr.t('metric'), tr.t('wi_now'), tr.t('wi_new')])
        self.t_cmp.setHorizontalHeaderLabels([tr.t('metric'), tr.t('test_set'), tr.t('live'), 'Δ'])
        self.t_ml.setHorizontalHeaderLabels([tr.t('ml_level'), tr.t('ml_what'), tr.t('ml_now')])
        if self.wi_spin.value() <= 0.0101:
            self.wi_spin.setValue(self.ctrl.threshold)
        self.refresh(force=True)

    # ------------------------------------------------------------------ what-if
    def _whatif_changed(self, v):
        self.wi_slider.blockSignals(True); self.wi_slider.setValue(int(round(v * 100))); self.wi_slider.blockSignals(False)
        self.refresh(force=True)

    def _wi_current(self):
        self.wi_spin.setValue(self.ctrl.threshold)

    def _wi_apply(self):
        self.ctrl.set_threshold(self.wi_spin.value())

    def reset(self):
        if QMessageBox.question(self, tr.t('reset_stats'), tr.t('confirm_reset')) == QMessageBox.StandardButton.Yes:
            self.ctrl.reset_stats()

    # ------------------------------------------------------------------ refresh
    def refresh(self, force=False):
        if not self.isVisible() and not force:
            return
        c = self.ctrl; st = c.stats; names = c.names
        s = st.summary(names)
        roc = st.roc(); pr = st.pr(); lat = st.latency(); oee = c.oee()
        self.oee_hist.append(oee['OEE'] * 100 if oee['OEE'] is not None else None)
        self.oee_hist = self.oee_hist[-120:]
        T = self.t
        T['kpi_total'].set(str(s['n'])); T['kpi_good'].set(str(s['good'])); T['kpi_defect'].set(str(s['bad']))
        T['kpi_rate'].set(fnum(s['rate'], 1, ' %')); T['kpi_rate'].set_spark(st.rolling_rate(20)[1])
        T['kpi_acc'].set(fpct(s['acc'])); T['recall_def'].set(fpct(s['recall'])); T['precision_def'].set(fpct(s['precision']))
        T['f1_def'].set(fpct(s['f1'])); T['specificity'].set(fpct(s['specificity']))
        T['auc'].set(fnum(roc[2], 3) if roc else '—')
        T['throughput'].set(fnum(s['throughput'], 1)); T['throughput'].set_spark(st.throughput_series(15)[1])
        T['lat_mean'].set(fnum(lat['mean'], 1, ' ms') if lat else '—')
        T['lat_mean'].set_spark([r['latency_ms'] for r in st.records[-80:]])
        T['lat_p95'].set(fnum(lat['p95'], 1, ' ms') if lat else '—')
        T['oee'].set(fpct(oee['OEE'])); T['oee'].set_spark(self.oee_hist)
        self.info.setText(tr.t('stats_info', n=s['n'], k=s['n_known'], t=fdur(time.time() - st.t0)))
        i = self.tabs.currentIndex()
        [self._r_overview, self._r_ai, self._r_defects, self._r_perf, self._r_model][i](s, roc, pr, lat, oee)

    def _r_overview(self, s, roc, pr, lat, oee):
        c = self.ctrl; st = c.stats
        self.g_oee.set(None if oee['OEE'] is None else oee['OEE'] * 100)
        self.g_a.set(None if oee['A'] is None else oee['A'] * 100)
        self.g_p.set(None if oee['P'] is None else oee['P'] * 100)
        self.g_q.set(None if oee['Q'] is None else oee['Q'] * 100)
        self.oee_info.setText(tr.t('oee_info', ideal=fnum(c.ideal_rate() * 60, 1), prod=fdur(oee['prod_s']), plan=fdur(oee['planned_s'])))
        d = c.states.dur
        self.d_states.set([(tr.t('state_' + k), round(d[k]), STATE_COLORS[k]) for k in STATES if d[k] >= 0.5])
        x, y = st.rolling_rate(20); self.c_roll.setData(x, y)
        x, y = st.cumulative_rate(); self.c_cum.setData(x, y)
        self.lim_line.setValue(float(c.settings['alarm_rate']))
        bx, by = st.throughput_series(15)
        self.b_thr.setOpts(x=bx, height=by, width=12)
        self.ideal_line.setValue(c.ideal_rate() * 60)
        rows = [('sess_time', fdur(time.time() - st.t0)), ('sess_prod', fdur(oee['prod_s'])),
                ('state_run', fdur(d['run'])), ('state_inspect', fdur(d['inspect'])), ('state_stop', fdur(d['stop'])),
                ('state_estop', fdur(d['estop'])), ('state_manual', fdur(d['manual'])),
                ('push_strokes', str(c.sim.push_strokes)), ('consec_now', str(s['consec'])),
                ('mtbd', fnum((time.time() - st.t0) / s['bad'], 1, ' s') if s['bad'] else '—')]
        for i, (k, v) in enumerate(rows):
            set_cell(self.t_sess, i, 0, tr.t(k), center=False, color=TH['sub']); set_cell(self.t_sess, i, 1, v, bold=True)
        self.timeline.update()

    def _r_ai(self, s, roc, pr, lat, oee):
        c = self.ctrl; st = c.stats; names = c.names
        cm = s['cm']; mx = max(1, cm.max())
        for i in range(len(names)):
            for j in range(len(names)):
                a = int(40 + 190 * cm[i, j] / mx) if cm[i, j] else 0
                bg = QColor(46, 157, 74, a) if i == j else QColor(211, 47, 47, a)
                set_cell(self.cm, i, j, str(cm[i, j]), bold=cm[i, j] > 0)
                self.cm.item(i, j).setBackground(bg)
        for i, pcr in enumerate(st.per_class(names)):
            vals = [tr.cls(pcr['cls']), str(pcr['support']), str(pcr['n_pred']), fpct(pcr['precision'], 0), fpct(pcr['recall'], 0),
                    fpct(pcr['f1'], 0), fpct(pcr['conf_mean'], 0)]
            for j, v in enumerate(vals):
                set_cell(self.pc, i, j, v, color=CLASS_COLORS.get(pcr['cls']) if j == 0 else None, bold=j == 0)
        mf = st.macro_f1(names)
        self.macro.setText(tr.t('macro_f1', f=fpct(mf), a=fpct(s['acc'])))
        # what-if
        th_now, th_new = c.threshold, self.wi_spin.value()
        a, b = st.binary_at(th_now), st.binary_at(th_new)
        rows = [('tp_lbl', 'tp', False), ('fp_lbl', 'fp', False), ('fn_lbl', 'fn', False), ('tn_lbl', 'tn', False),
                ('recall_def', 'recall', True), ('precision_def', 'precision', True), ('f1_def', 'f1', True), ('specificity', 'specificity', True)]
        for i, (lbl, k, pct) in enumerate(rows):
            va, vb = a[k], b[k]
            fa = fpct(va) if pct else str(va); fb = fpct(vb) if pct else str(vb)
            better = None
            if va is not None and vb is not None and va != vb:
                up_good = k not in ('fp', 'fn')
                better = (vb > va) == up_good
            set_cell(self.wi, i, 0, tr.t(lbl), center=False); set_cell(self.wi, i, 1, fa)
            set_cell(self.wi, i, 2, fb, color=(GREEN if better else RED) if better is not None else None, bold=True)
        # ROC / PR
        if roc:
            fpr, tpr, auc = roc
            self.c_roc.setData(fpr, tpr)
            pts = [{'pos': (a['fpr'] or 0, a['recall'] or 0), 'brush': pg.mkBrush(ORANGE), 'symbol': 'o'},
                   {'pos': (b['fpr'] or 0, b['recall'] or 0), 'brush': pg.mkBrush(BLUE), 'symbol': 'd'}]
            self.s_roc.setData(pts)
            ref = c.test_metrics().get('auc')
            self.l_roc.setText(tr.t('auc_txt', a=fnum(auc, 3), r=fnum(ref, 3)))
        else:
            self.c_roc.setData([], []); self.s_roc.setData([]); self.l_roc.setText(tr.t('need_both'))
        if pr:
            rec, prec, ap = pr
            self.c_pr.setData(rec, prec)
            self.s_pr.setData([{'pos': (a['recall'] or 0, a['precision'] or 0), 'brush': pg.mkBrush(ORANGE)},
                               {'pos': (b['recall'] or 0, b['precision'] or 0), 'brush': pg.mkBrush(BLUE), 'symbol': 'd'}])
            self.l_pr.setText(tr.t('ap_txt', a=fnum(ap, 3)))
        else:
            self.c_pr.setData([], []); self.s_pr.setData([]); self.l_pr.setText(tr.t('need_both'))
        k = st.known()
        edges = np.linspace(0, 1, 21)
        hg, _ = np.histogram([r['p_defect'] for r in k if r['truth'] == 'good'], edges)
        hb, _ = np.histogram([r['p_defect'] for r in k if r['truth'] != 'good'], edges)
        self.h_good.setData(edges, hg); self.h_bad.setData(edges, hb)
        self.h_thr.setValue(th_now); self.h_wi.setValue(th_new)

    def _r_defects(self, s, roc, pr, lat, oee):
        c = self.ctrl; st = c.stats; names = c.names
        items, cum = st.pareto(names)
        self.b_par.setOpts(x=list(range(len(items))), height=[v for _, v in items], width=0.6,
                           brushes=[pg.mkBrush(CLASS_COLORS.get(n, ORANGE)) for n, _ in items])
        self.c_par.setData(list(range(len(items))), cum)
        self.p_par.getAxis('bottom').setTicks([[(i, tr.cls(n)) for i, (n, _) in enumerate(items)]])
        top = [tr.cls(n) for (n, v), cp in zip(items, cum) if v > 0]
        k80 = next((i for i, cp in enumerate(cum) if cp >= 80), None)
        self.l_par.setText(tr.t('pareto_txt', c=', '.join(top[:k80 + 1]) if k80 is not None and top else '—'))
        self.d_def.set([(tr.cls(n), v, CLASS_COLORS.get(n, ORANGE)) for n, v in items if v > 0])
        last = st.records[-300:]
        if last:
            self.s_conf.setData([r['n'] for r in last], [r['p_defect'] for r in last],
                                brush=[pg.mkBrush(RED if r['decision'] == 'defect' else GREEN) for r in last],
                                symbol=['o' if (not r['truth'] or (r['decision'] == 'defect') == (r['truth'] != 'good')) else 'x' for r in last])
        else:
            self.s_conf.setData([], [])
        self.thr_line.setValue(c.threshold)
        self.b_cls.setOpts(height=[s['by_pred'][n] for n in names])

    def _r_perf(self, s, roc, pr, lat, oee):
        c = self.ctrl; st = c.stats
        L_ = np.array([r['latency_ms'] for r in st.records], dtype=float)
        if len(L_):
            hi = max(1.0, float(np.percentile(L_, 99)) * 1.2)
            edges = np.linspace(0, hi, 26); h, _ = np.histogram(np.clip(L_, 0, hi), edges)
            self.h_lat.setData(edges, h); self.lh_p95.setValue(lat['p95'])
            x = np.arange(1, len(L_) + 1); self.c_lat.setData(x, L_)
            ma = np.convolve(L_, np.ones(10) / 10, mode='valid') if len(L_) >= 10 else L_
            self.c_latm.setData(np.arange(len(L_) - len(ma) + 1, len(L_) + 1), ma)
        else:
            self.h_lat.setData([0, 1], [0]); self.c_lat.setData([], []); self.c_latm.setData([], [])
        self.lim_lat.setValue(float(c.settings['alarm_latency']))
        cyc = np.array(c.cycle_hist, dtype=float)
        self.c_cyc.setData(np.arange(len(cyc)), cyc)
        ref = c.test_metrics().get('latency')
        rows = [('lat_mean', fnum(lat['mean'], 2, ' ms') if lat else '—'), ('lat_p50', fnum(lat['p50'], 2, ' ms') if lat else '—'),
                ('lat_p95', fnum(lat['p95'], 2, ' ms') if lat else '—'), ('lat_max', fnum(lat['max'], 2, ' ms') if lat else '—'),
                ('lat_std', fnum(lat['std'], 2, ' ms') if lat else '—'), ('lat_ref', fnum(ref, 2, ' ms')),
                ('plc_scans', str(c.plc.scans)), ('plc_cyc_mean', fnum(float(cyc.mean()) if len(cyc) else None, 3, ' ms')),
                ('plc_cyc_max', fnum(float(cyc.max()) if len(cyc) else None, 3, ' ms')),
                ('tick_ms', f'{c.TICK_MS} ms'), ('ideal_rate', fnum(c.ideal_rate() * 60, 1, ' / min')),
                ('throughput', fnum(s['throughput'], 1, ' / min'))]
        for i, (k, v) in enumerate(rows):
            set_cell(self.t_perf, i, 0, tr.t(k), center=False, color=TH['sub']); set_cell(self.t_perf, i, 1, v, bold=True)

    def _r_model(self, s, roc, pr, lat, oee):
        c = self.ctrl; m = c.model
        meta = m.meta or {}
        info = [('m_name', str(meta.get('model', '—'))), ('m_created', str(meta.get('created', '—'))),
                ('m_torch', str(meta.get('torch_version', '—'))), ('m_input', f'{m.size}×{m.size} RGB' if m.ready else '—'),
                ('m_classes', ', '.join(tr.cls(n) for n in c.names)), ('m_demo', str(len(m.demo))),
                ('m_thr_def', fnum(m.default_threshold, 3) if m.ready else '—'), ('m_thr_now', fnum(c.threshold, 3)),
                ('m_norm', 'ImageNet' if meta.get('mean') and abs(meta['mean'][0] - 0.485) < 1e-3 else tr.t('m_norm_ds')),
                ('m_format', 'TorchScript (.pt)')]
        for i, (k, v) in enumerate(info):
            set_cell(self.t_model, i, 0, tr.t(k), center=False, color=TH['sub']); set_cell(self.t_model, i, 1, v, bold=True)
        ref = c.test_metrics(); b = c.stats.binary_at(c.threshold)
        live = {'acc': s['acc'], 'f1': c.stats.macro_f1(c.names), 'auc': roc[2] if roc else None, 'recall': b['recall'],
                'fpr': b['fpr'], 'precision': b['precision'], 'latency': lat['mean'] if lat else None}
        rows = [('kpi_acc', 'acc', True), ('macro_f1_s', 'f1', True), ('auc', 'auc', False), ('recall_def', 'recall', True),
                ('fpr_s', 'fpr', True), ('precision_def', 'precision', True), ('lat_mean', 'latency', False)]
        for i, (lbl, k, pct) in enumerate(rows):
            r, v = ref.get(k), live.get(k)
            f = (lambda x: fpct(x)) if pct else (lambda x: fnum(x, 3) if k != 'latency' else fnum(x, 1, ' ms'))
            delta, col = '—', None
            if r is not None and v is not None:
                dd = v - r
                delta = (f'{dd * 100:+.1f} pp' if pct else (f'{dd:+.3f}' if k != 'latency' else f'{dd:+.1f} ms')).replace('.', ',')
                worse = (dd > 0) if k in ('fpr', 'latency') else (dd < 0)
                col = RED if worse and abs(dd) > (0.1 if pct or k == 'auc' else 50) else (YELLOW if worse else GREEN)
            set_cell(self.t_cmp, i, 0, tr.t(lbl), center=False); set_cell(self.t_cmp, i, 1, f(r)); set_cell(self.t_cmp, i, 2, f(v), bold=True)
            set_cell(self.t_cmp, i, 3, delta, color=col, bold=True)
        a = c.alarms.get('drift')
        nk = b['tp'] + b['fn']
        if nk < 15:
            self.drift.setText('⏳ ' + tr.t('drift_wait')); self.drift.setStyleSheet(f'color:{TH["sub"]};')
            self.drift_sub.setText(tr.t('drift_need', n=nk))
        elif a and a['active']:
            self.drift.setText('⚠ ' + tr.t('drift_yes')); self.drift.setStyleSheet(f'color:{RED};')
            self.drift_sub.setText(tr.t('drift_yes_sub'))
        else:
            self.drift.setText('✔ ' + tr.t('drift_no')); self.drift.setStyleSheet(f'color:{GREEN};')
            self.drift_sub.setText(tr.t('drift_no_sub', m=fpct(float(c.settings['drift_margin']), 0)))
        ml = [('ml_sys', 'ml_sys_what', tr.t('ml_sys_now', l=fnum(lat['mean'] if lat else None, 1), p=fnum(lat['p95'] if lat else None, 1),
                                                c=fnum(float(np.mean(c.cycle_hist)) if c.cycle_hist else None, 3))),
              ('ml_model', 'ml_model_what', tr.t('ml_model_now', a=fpct(s['acc']), r=fpct(b['recall']), d=self.drift.text())),
              ('ml_bus', 'ml_bus_what', tr.t('ml_bus_now', q=fnum(s['rate'], 1), o=fpct(oee['OEE']), t=fnum(s['throughput'], 1)))]
        for i, (a1, a2, a3) in enumerate(ml):
            set_cell(self.t_ml, i, 0, tr.t(a1), bold=True, center=False); set_cell(self.t_ml, i, 1, tr.t(a2), center=False)
            set_cell(self.t_ml, i, 2, a3, center=False)

    # ------------------------------------------------------------------ export
    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, tr.t('export_csv'), f'rezultati_{datetime.datetime.now():%Y%m%d_%H%M}.csv', 'CSV (*.csv)')
        if path:
            self.ctrl.stats.export(path); self.ctrl.log('INFO', tr.t('log_export', f=path))

    def grab_tabs(self):
        """Screenshots of all dashboard tabs (for the report)."""
        was_visible = self.isVisible()
        if not was_visible:
            self.show()
        cur = self.tabs.currentIndex(); shots = []
        for i in range(self.tabs.count()):
            self.tabs.setCurrentIndex(i); self.refresh(force=True)
            for _ in range(3):
                QApplication.processEvents()
            shots.append((self.tabs.tabText(i), self.tabs.currentWidget().grab().toImage()))
        self.tabs.setCurrentIndex(cur)
        if not was_visible:
            self.hide()
        return shots

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, tr.t('export_png'), f'statistika_{datetime.datetime.now():%Y%m%d_%H%M}.png', 'PNG (*.png)')
        if path:
            self.grab().save(path); self.ctrl.log('INFO', tr.t('log_export', f=path))

    def export_report(self, kind, path=None):
        ext = 'pdf' if kind == 'pdf' else 'html'
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, tr.t('export_' + kind), f'atskaite_{datetime.datetime.now():%Y%m%d_%H%M}.{ext}',
                                                  f'{ext.upper()} (*.{ext})')
        if not path:
            return None
        shots = self.grab_tabs()
        if kind == 'html':
            imgs = []
            for title, img in shots:
                buf = QBuffer(); buf.open(QIODevice.OpenModeFlag.WriteOnly); img.save(buf, 'PNG')
                imgs.append((title, 'data:image/png;base64,' + base64.b64encode(bytes(buf.data())).decode()))
            html = build_report_html(self.ctrl, imgs, width=1100)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
        else:
            doc = QTextDocument()
            imgs = []
            for k, (title, img) in enumerate(shots):
                name = f'img{k}.png'
                doc.addResource(QTextDocument.ResourceType.ImageResource.value, QUrl(name), img)
                imgs.append((title, name))
            doc.setHtml(build_report_html(self.ctrl, imgs, width=640, pdf=True))
            w = QPdfWriter(path)
            w.setPageLayout(QPageLayout(QPageSize(QPageSize.PageSizeId.A4), QPageLayout.Orientation.Portrait, QMarginsF(12, 12, 12, 12)))
            w.setResolution(96); w.setTitle(tr.t('report_title'))
            doc.setPageSize(QSizeF(w.width(), w.height()))
            (getattr(doc, 'print', None) or getattr(doc, 'print_'))(w)
        self.ctrl.log('INFO', tr.t('log_export', f=path))
        return path


def build_report_html(ctrl, imgs, width=1100, pdf=False):
    """Session report (HTML; also used for PDF through QTextDocument)."""
    c = ctrl; st = c.stats; names = c.names
    s = st.summary(names); roc = st.roc(); pr = st.pr(); lat = st.latency(); oee = c.oee()
    ref = c.test_metrics(); meta = c.model.meta or {}
    E = lambda x: str(x).replace('&', '&amp;').replace('<', '&lt;')
    now = datetime.datetime.now()
    acc = '#C55A11'
    css = (f'body{{font-family:Segoe UI,Arial;color:#22303A;max-width:{width + 40}px;margin:20px auto;}}'
           f'h1{{color:{acc};margin-bottom:2px}} h2{{color:{acc};border-bottom:2px solid {acc};padding-bottom:3px;margin-top:22px}}'
           'table{border-collapse:collapse;margin:6px 0 10px 0} td,th{border:1px solid #C8D2DC;padding:5px 9px;font-size:10pt}'
           'th{background:#ECEFF3} .sub{color:#5F7686} .good{color:#2E9D4A;font-weight:700} .bad{color:#D32F2F;font-weight:700}')

    def kv(rows):
        return '<table>' + ''.join(f'<tr><th align="left">{E(k)}</th><td>{E(v)}</td></tr>' for k, v in rows) + '</table>'

    h = [f'<html><head><meta charset="utf-8"><title>{E(tr.t("report_title"))}</title><style>{css}</style></head><body>']
    h.append(f'<h1>{E(tr.t("report_title"))}</h1><div class="sub">{E(tr.t("app_title"))} · {now:%d.%m.%Y %H:%M:%S}</div>')
    h.append(f'<h2>1. {E(tr.t("rep_session"))}</h2>')
    h.append(kv([(tr.t('sess_time'), fdur(time.time() - st.t0)), (tr.t('mode'), tr.t('mode_' + c.mode)),
                 (tr.t('plc_source'), tr.t('plc_opc') if c.plc.kind == 'opc' else tr.t('plc_sim')),
                 (tr.t('m_name'), meta.get('model', '—')), (tr.t('threshold'), fnum(c.threshold, 3)),
                 (tr.t('ideal_rate'), fnum(c.ideal_rate() * 60, 1, ' / min'))]))
    h.append(f'<h2>2. {E(tr.t("rep_kpi"))}</h2>')
    h.append(kv([(tr.t('kpi_total'), s['n']), (tr.t('kpi_good'), s['good']), (tr.t('kpi_defect'), s['bad']),
                 (tr.t('kpi_rate'), fnum(s['rate'], 1, ' %')), (tr.t('throughput'), fnum(s['throughput'], 1)),
                 (tr.t('oee'), fpct(oee['OEE'])), (tr.t('availability'), fpct(oee['A'])),
                 (tr.t('performance'), fpct(oee['P'])), (tr.t('quality'), fpct(oee['Q'])),
                 (tr.t('push_strokes'), c.sim.push_strokes)]))
    h.append(f'<h2>3. {E(tr.t("rep_ai"))}</h2>')
    rows = [('kpi_acc', s['acc'], ref.get('acc'), True), ('recall_def', s['recall'], ref.get('recall'), True),
            ('precision_def', s['precision'], ref.get('precision'), True), ('f1_def', s['f1'], None, True),
            ('specificity', s['specificity'], None, True), ('auc', roc[2] if roc else None, ref.get('auc'), False),
            ('ap', pr[2] if pr else None, None, False), ('lat_mean', lat['mean'] if lat else None, ref.get('latency'), False),
            ('lat_p95', lat['p95'] if lat else None, None, False)]
    h.append(f'<table><tr><th>{E(tr.t("metric"))}</th><th>{E(tr.t("live"))}</th><th>{E(tr.t("test_set"))}</th></tr>')
    for k, v, r, pct in rows:
        f = fpct if pct else (lambda x: fnum(x, 3))
        h.append(f'<tr><td>{E(tr.t(k))}</td><td align="center"><b>{E(f(v))}</b></td><td align="center">{E(f(r))}</td></tr>')
    h.append('</table>')
    h.append(f'<p>TP = {s["tp"]} · FP = {s["fp"]} · FN = {s["fn"]} · TN = {s["tn"]}</p>')
    # confusion matrix
    lab = [tr.cls(n) for n in names]
    h.append(f'<p><b>{E(tr.t("st_cm"))}</b></p><table><tr><th></th>' + ''.join(f'<th>{E(x)}</th>' for x in lab) + '</tr>')
    for i, n in enumerate(names):
        h.append(f'<tr><th align="left">{E(lab[i])}</th>' + ''.join(
            f'<td align="center" style="background:{"#D7F0DC" if i == j and s["cm"][i, j] else ("#F8D7D7" if s["cm"][i, j] else "#FFFFFF")}">{s["cm"][i, j]}</td>'
            for j in range(len(names))) + '</tr>')
    h.append('</table>')
    h.append(f'<p><b>{E(tr.t("per_class"))}</b></p><table><tr>' + ''.join(
        f'<th>{E(tr.t(k))}</th>' for k in ('class', 'support', 'n_pred', 'precision_s', 'recall_s', 'f1_s')) + '</tr>')
    for p in st.per_class(names):
        h.append(f'<tr><td>{E(tr.cls(p["cls"]))}</td><td align="center">{p["support"]}</td><td align="center">{p["n_pred"]}</td>'
                 f'<td align="center">{fpct(p["precision"])}</td><td align="center">{fpct(p["recall"])}</td><td align="center">{fpct(p["f1"])}</td></tr>')
    h.append('</table>')
    h.append(f'<h2>4. {E(tr.t("pareto"))}</h2>')
    items, cum = st.pareto(names)
    h.append('<table><tr>' + ''.join(f'<th>{E(x)}</th>' for x in (tr.t('class'), tr.t('count'), tr.t('cum_pct'))) + '</tr>')
    for (n, v), cp in zip(items, cum):
        h.append(f'<tr><td>{E(tr.cls(n))}</td><td align="center">{v}</td><td align="center">{fnum(cp, 1, " %")}</td></tr>')
    h.append('</table>')
    h.append(f'<h2>5. {E(tr.t("win_alarms"))}</h2>')
    hist = c.alarm_hist[-30:]
    if hist:
        h.append('<table><tr>' + ''.join(f'<th>{E(x)}</th>' for x in (tr.t('col_time'), tr.t('col_msg'), tr.t('al_event'))) + '</tr>')
        for t, key, kw, ev, lvl in reversed(hist):
            cls = 'bad' if ev == 'ON' and lvl == 'ALARM' else ''
            h.append(f'<tr><td>{t}</td><td class="{cls}">{E(tr.t("al_" + key, **kw))}</td><td align="center">{ev}</td></tr>')
        h.append('</table>')
    else:
        h.append(f'<p class="good">{E(tr.t("no_alarms"))}</p>')
    h.append(f'<h2>6. {E(tr.t("rep_charts"))}</h2>')
    for title, src in imgs:
        h.append(f'<p><b>{E(title)}</b></p><p><img src="{src}" width="{width}"></p>')
    h.append(f'<p class="sub">{E(tr.t("rep_footer"))}</p></body></html>')
    return '\n'.join(h)
