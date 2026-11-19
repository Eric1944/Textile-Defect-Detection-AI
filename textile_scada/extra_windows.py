"""Additional windows: alarm management (ISA-18.2 style) and batch inspection of a folder."""
import csv, datetime, time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QPixmap, QIcon
from PyQt6.QtWidgets import (QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QFileDialog, QProgressBar, QCheckBox, QSplitter, QWidget)

from .i18n import tr
from .core import collect_images
from .theme import TH, GREEN, RED, ORANGE, BLUE, YELLOW, GRAY, CLASS_COLORS
from .widgets import BaseWindow, card, KpiTile, Donut, L, B, fnum, fpct, fdur, PIX, page


def _table(cols):
    t = QTableWidget(0, cols); t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.verticalHeader().setVisible(False); t.setAlternatingRowColors(True)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    return t


# =====================================================================
# Alarms
# =====================================================================
class AlarmWindow(BaseWindow):
    title_key = 'win_alarms'

    def __init__(self, ctrl):
        super().__init__(ctrl); self.resize(1100, 720)
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(10)
        top = QHBoxLayout()
        self.k_active = KpiTile('al_active', RED); self.k_unack = KpiTile('al_unacked', ORANGE); self.k_total = KpiTile('al_total')
        for k in (self.k_active, self.k_unack, self.k_total):
            top.addWidget(k)
        root.addLayout(top)
        f, l, _ = card('al_list', '🚨')
        self.t = _table(5); self.t.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        l.addWidget(self.t)
        r = QHBoxLayout(); r.addStretch()
        r.addWidget(B('ack_sel', None, self.ack_sel)); r.addWidget(B('ack_all', 'Accent', lambda: ctrl.ack()))
        l.addLayout(r)
        root.addWidget(f, 3)
        f, l, _ = card('al_hist', '🕘')
        self.h = _table(4); self.h.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        l.addWidget(self.h)
        root.addWidget(f, 2)
        f, l, _ = card('al_limits', '⚙')
        self.lim = L('al_limits_txt', 'Sub'); self.lim.setWordWrap(True); l.addWidget(self.lim)
        root.addWidget(f)
        ctrl.alarms_changed.connect(self.reload)
        self.retranslate()

    def retranslate(self):
        super().retranslate()
        self.t.setHorizontalHeaderLabels([tr.t('col_time'), tr.t('al_level'), tr.t('col_msg'), tr.t('al_state'), tr.t('al_ack')])
        self.h.setHorizontalHeaderLabels([tr.t('col_time'), tr.t('al_level'), tr.t('col_msg'), tr.t('al_event')])
        self.reload()

    def reload(self):
        c = self.ctrl; s = c.settings
        al = c.visible_alarms()
        self.t.setRowCount(len(al))
        for i, a in enumerate(al):
            col = RED if a['level'] == 'ALARM' else BLUE
            vals = [a['time'], 'ALARM' if a['level'] == 'ALARM' else 'INFO', tr.t('al_' + a['key'], **a['kw']),
                    tr.t('al_on') if a['active'] else tr.t('al_off'), '✔' if a['acked'] else '✖']
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v); it.setData(Qt.ItemDataRole.UserRole, a['key'])
                it.setForeground(QColor(col if a['active'] else TH['sub']))
                if a['active'] and not a['acked'] and a['level'] == 'ALARM':
                    it.setBackground(QColor(211, 47, 47, 45))
                self.t.setItem(i, j, it)
        hist = list(reversed(c.alarm_hist[-400:]))
        self.h.setRowCount(len(hist))
        for i, (t, key, kw, ev, lvl) in enumerate(hist):
            col = {'ON': RED if lvl == 'ALARM' else BLUE, 'OFF': GREEN, 'ACK': ORANGE}.get(ev, TH['text'])
            for j, v in enumerate((t, lvl, tr.t('al_' + key, **kw), ev)):
                it = QTableWidgetItem(v); it.setForeground(QColor(col)); self.h.setItem(i, j, it)
        self.k_active.set(str(sum(a['active'] for a in al)))
        self.k_unack.set(str(sum(not a['acked'] for a in al)))
        self.k_total.set(str(sum(1 for h in c.alarm_hist if h[3] == 'ON')))
        self.lim.setText(tr.t('al_limits_txt', r=fnum(float(s['alarm_rate']), 0), k=int(s['alarm_consec']),
                              ms=fnum(float(s['alarm_latency']), 0), d=fnum(float(s['drift_margin']) * 100, 0)))

    def ack_sel(self):
        keys = {self.t.item(i.row(), 0).data(Qt.ItemDataRole.UserRole) for i in self.t.selectedIndexes()}
        if keys:
            self.ctrl.ack(keys)


# =====================================================================
# Batch inspection
# =====================================================================
class BatchWindow(BaseWindow):
    title_key = 'win_batch'

    def __init__(self, ctrl):
        super().__init__(ctrl); self.resize(1320, 820)
        self.rows = []; self.folder = None; self.running = False; self.t_start = 0
        root = QVBoxLayout(self); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(10)
        f, l, _ = card('batch_src', '📁')
        r = QHBoxLayout()
        self.lbl_folder = L('batch_none', 'Sub'); r.addWidget(self.lbl_folder, 1)
        self.chk_rec = QCheckBox(tr.t('batch_recursive')); self.chk_rec.setProperty('tkey', 'batch_recursive'); self.chk_rec.setChecked(True)
        r.addWidget(self.chk_rec)
        r.addWidget(B('batch_choose', None, self.choose)); r.addWidget(B('batch_demo', None, self.use_demo))
        self.b_run = B('batch_run', 'Accent', self.run); self.b_cancel = B('batch_cancel', None, ctrl.cancel_batch)
        r.addWidget(self.b_run); r.addWidget(self.b_cancel); l.addLayout(r)
        self.prog = QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0); l.addWidget(self.prog)
        self.hint = L('batch_hint', 'Sub'); self.hint.setWordWrap(True); l.addWidget(self.hint)
        root.addWidget(f)
        kp = QHBoxLayout(); self.k = {}
        for k, col in [('kpi_total', None), ('kpi_good', GREEN), ('kpi_defect', RED), ('kpi_rate', ORANGE),
                       ('kpi_acc', BLUE), ('recall_def', BLUE), ('lat_mean', None), ('batch_speed', None)]:
            self.k[k] = KpiTile(k, col); kp.addWidget(self.k[k])
        root.addLayout(kp)
        body = QHBoxLayout(); root.addLayout(body, 1)
        f, l, _ = card('batch_results', '📋')
        self.t = _table(7); self.t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.t.setIconSize(QSize(34, 34)); self.t.itemSelectionChanged.connect(self.on_sel)
        l.addWidget(self.t)
        r = QHBoxLayout(); r.addStretch(); r.addWidget(B('export_csv', None, self.export)); l.addLayout(r)
        body.addWidget(f, 3)
        right = QVBoxLayout(); body.addLayout(right, 1)
        f, l, _ = card('current_image', '🖼')
        self.prev = QLabel(); self.prev.setFixedSize(240, 240); self.prev.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prev.setStyleSheet('background:#0B1117; border-radius:8px;')
        l.addWidget(self.prev, alignment=Qt.AlignmentFlag.AlignCenter)
        self.prev_txt = QLabel('—'); self.prev_txt.setWordWrap(True); self.prev_txt.setAlignment(Qt.AlignmentFlag.AlignCenter); l.addWidget(self.prev_txt)
        self.b_insp = B('batch_to_inspect', None, self.to_inspect); l.addWidget(self.b_insp)
        right.addWidget(f)
        f, l, _ = card('st_classes', '🍩'); self.donut = Donut('kpi_total'); l.addWidget(self.donut); right.addWidget(f, 1)
        ctrl.bworker.progress.connect(self.on_progress)
        ctrl.bworker.done.connect(self.on_done)
        self.retranslate(); self._buttons()

    def retranslate(self):
        super().retranslate()
        self.t.setHorizontalHeaderLabels([tr.t('col_file'), tr.t('class'), tr.t('confidence'), tr.t('p_defect'),
                                          tr.t('decision'), tr.t('truth'), tr.t('col_ok')])
        if self.folder:
            self.lbl_folder.setText(self.folder)
        self.summary()

    def _buttons(self):
        self.b_run.setEnabled(bool(self.folder) and not self.running)
        self.b_cancel.setEnabled(self.running)

    def choose(self):
        d = QFileDialog.getExistingDirectory(self, tr.t('batch_choose'), self.folder or '')
        if d:
            self.folder = d; self.lbl_folder.setText(d); self._buttons()

    def use_demo(self):
        if self.ctrl.model.ready:
            self.folder = str(Path(self.ctrl.settings['model_dir']) / 'demo_images'); self.lbl_folder.setText(self.folder); self._buttons()

    def run(self, jobs=None):
        if jobs is None or isinstance(jobs, bool):
            jobs = collect_images(self.folder, self.ctrl.names, self.chk_rec.isChecked())
        if not jobs:
            self.hint.setText(tr.t('batch_empty')); return
        self.rows = []; self.t.setRowCount(0)
        if self.ctrl.start_batch(jobs):
            self.running = True; self.t_start = time.perf_counter()
            self.prog.setRange(0, len(jobs)); self.prog.setValue(0); self._buttons()

    def on_progress(self, i, n, res):
        self.prog.setValue(i)
        self.rows.append(res)
        r = self.t.rowCount(); self.t.insertRow(r)
        if 'error' in res:
            self.t.setItem(r, 0, QTableWidgetItem(Path(res['path']).name)); self.t.setItem(r, 1, QTableWidgetItem(res['error'][:60]))
            return
        defect = res['defect']; truth = res.get('truth')
        ok = None if not truth else ((truth != 'good') == defect)
        vals = [Path(res['path']).name, tr.cls(res['name']), fpct(res['confidence']), fnum(res['p_defect'], 3),
                tr.t('short_defect' if defect else 'short_good'), tr.cls(truth) if truth else '—',
                '—' if ok is None else ('✔' if ok else '✖')]
        for j, v in enumerate(vals):
            it = QTableWidgetItem(v)
            if j == 0:
                it.setIcon(QIcon(PIX.get(res['path']))); it.setData(Qt.ItemDataRole.UserRole, r)
            if j == 1:
                it.setForeground(QColor(CLASS_COLORS.get(res['name'], TH['text'])))
            if j == 4:
                it.setForeground(QColor(RED if defect else GREEN))
            if j == 6 and ok is not None:
                it.setForeground(QColor(GREEN if ok else RED))
            self.t.setItem(r, j, it)
        self.t.setRowHeight(r, 38)
        if i % 10 == 0 or i == n:
            self.summary()
            self.t.scrollToBottom()

    def on_done(self, sec):
        self.running = False; self._buttons(); self.summary()
        good = [r for r in self.rows if 'error' not in r]
        self.ctrl.log('AI', tr.t('log_batch_done', n=len(good), s=fnum(sec, 1)))

    def summary(self):
        R = [r for r in self.rows if 'error' not in r]
        n = len(R); bad = sum(r['defect'] for r in R)
        known = [r for r in R if r.get('truth')]
        acc = sum(r['name'] == r['truth'] for r in known) / len(known) if known else None
        td = [r for r in known if r['truth'] != 'good']
        rec = sum(r['defect'] for r in td) / len(td) if td else None
        lat = float(np.mean([r['latency_ms'] for r in R])) if R else None
        el = time.perf_counter() - self.t_start if self.t_start else 0
        self.k['kpi_total'].set(str(n)); self.k['kpi_good'].set(str(n - bad)); self.k['kpi_defect'].set(str(bad))
        self.k['kpi_rate'].set(fnum(bad / n * 100 if n else None, 1, ' %')); self.k['kpi_acc'].set(fpct(acc))
        self.k['recall_def'].set(fpct(rec)); self.k['lat_mean'].set(fnum(lat, 1, ' ms'))
        self.k['batch_speed'].set(fnum(n / el if el > 0 and n else None, 1, ' /s'))
        cnt = {}
        for r in R:
            cnt[r['name']] = cnt.get(r['name'], 0) + 1
        self.donut.set([(tr.cls(k), v, CLASS_COLORS.get(k, ORANGE)) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])])

    def _sel_row(self):
        rows = {i.row() for i in self.t.selectedIndexes()}
        if not rows:
            return None
        r = min(rows)
        return self.rows[r] if r < len(self.rows) else None

    def on_sel(self):
        res = self._sel_row()
        if not res or 'error' in res:
            return
        self.prev.setPixmap(QPixmap(res['path']).scaled(240, 240, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
        col = RED if res['defect'] else GREEN
        self.prev_txt.setText(f'<b style="color:{col}">{tr.cls(res["name"])}</b> · {fpct(res["confidence"])}<br>'
                              f'P(def) {fnum(res["p_defect"], 3)} · {fnum(res["latency_ms"], 1, " ms")}')

    def to_inspect(self):
        res = self._sel_row()
        if res and 'error' not in res:
            self.ctrl.inspect_file(res['path'])

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, tr.t('export_csv'), f'partija_{datetime.datetime.now():%Y%m%d_%H%M}.csv', 'CSV (*.csv)')
        if not path:
            return
        names = self.ctrl.names
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(['file', 'pred', 'confidence', 'p_defect', 'decision', 'truth', 'latency_ms'] + [f'p_{n}' for n in names])
            for r in self.rows:
                if 'error' in r:
                    continue
                w.writerow([r['path'], r['name'], f'{r["confidence"]:.4f}'.replace('.', ','), f'{r["p_defect"]:.4f}'.replace('.', ','),
                            'defect' if r['defect'] else 'good', r.get('truth') or '', f'{r["latency_ms"]:.2f}'.replace('.', ',')]
                           + [f'{v:.4f}'.replace('.', ',') for v in r['probs']])
        self.ctrl.log('INFO', tr.t('log_export', f=path))
