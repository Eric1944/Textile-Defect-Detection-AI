"""Custom widgets drawn with QPainter (lecture: QPainter, paintEvent, update()) + small UI helpers."""
import math, time

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (QPainter, QPen, QBrush, QColor, QFont, QPixmap, QImage, QLinearGradient, QRadialGradient,
                         QPainterPath, QPolygonF)
from PyQt6.QtWidgets import QWidget, QFrame, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy, QPushButton

from .core import ProcessSim
from .i18n import tr
from .theme import TH, ORANGE, GREEN, RED, BLUE, YELLOW, GRAY, PURPLE, CLASS_COLORS, STATE_COLORS

STATUS_COLOR = {'new': GRAY, 'wait': YELLOW, 'good': GREEN, 'reject_pending': RED, 'reject_drop': RED, 'manual_wait': BLUE}


# =====================================================================
# helpers shared by all windows
# =====================================================================
def L(key, obj=None, **kw):
    w = QLabel(tr.t(key, **kw)); w.setProperty('tkey', key)
    if obj: w.setObjectName(obj)
    return w


def B(key, obj=None, slot=None, tip=None):
    w = QPushButton(tr.t(key)); w.setProperty('tkey', key)
    if obj: w.setObjectName(obj)
    if slot: w.clicked.connect(slot)
    if tip: w.setProperty('tipkey', tip); w.setToolTip(tr.t(tip))
    return w


def retranslate_tree(root):
    for w in [root] + root.findChildren(QWidget):
        key = w.property('tkey')
        if key and hasattr(w, 'setText'):
            w.setText(tr.t(key))
        tip = w.property('tipkey')
        if tip:
            w.setToolTip(tr.t(tip))


def page(widget):
    widget.setObjectName('Page'); widget.setAutoFillBackground(True)


def fnum(v, d=1, unit=''):
    return '—' if v is None else (f'{v:.{d}f}'.replace('.', ',') + unit)


def fpct(v, d=1):
    """v in 0..1 → '93,4 %'"""
    return '—' if v is None else f'{v * 100:.{d}f} %'.replace('.', ',')


def fdur(sec):
    if sec is None:
        return '—'
    sec = int(sec); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def card(title_key=None, icon=''):
    """Rounded panel with an optional accent title; returns (frame, layout, title_label)."""
    f = QFrame(); f.setObjectName('Card')
    lay = QVBoxLayout(f); lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(8)
    lbl = None
    if title_key:
        lbl = QLabel((icon + '  ' if icon else '') + tr.t(title_key)); lbl.setObjectName('CardTitle')
        if not icon:
            lbl.setProperty('tkey', title_key)
        else:
            lbl.setProperty('ckey', title_key); lbl.setProperty('cicon', icon)
        lay.addWidget(lbl)
    return f, lay, lbl


def retranslate_cards(root):
    for w in root.findChildren(QLabel):
        k = w.property('ckey')
        if k:
            w.setText(w.property('cicon') + '  ' + tr.t(k))


def qc(col, alpha=None):
    c = QColor(col)
    if alpha is not None:
        c.setAlpha(alpha)
    return c


# =====================================================================
# Small indicators
# =====================================================================
class Lamp(QWidget):
    clicked = pyqtSignal()

    def __init__(self, color=GREEN, size=26, parent=None):
        super().__init__(parent)
        self.color, self.on, self.size_ = QColor(color), False, size
        self.setFixedSize(size + 8, size + 8)

    def set(self, on):
        if on != self.on:
            self.on = on; self.update()

    def mousePressEvent(self, e):
        self.clicked.emit()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(4, 4, self.size_, self.size_)
        if self.on:
            glow = QRadialGradient(r.center(), self.size_ * 0.75)
            glow.setColorAt(0, qc(self.color, 120)); glow.setColorAt(1, qc(self.color, 0))
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(glow)); p.drawEllipse(r.adjusted(-4, -4, 4, 4))
            g = QRadialGradient(r.center() - QPointF(self.size_ * .15, self.size_ * .15), self.size_ / 1.6)
            g.setColorAt(0, self.color.lighter(180)); g.setColorAt(1, self.color)
            p.setBrush(QBrush(g))
        else:
            p.setBrush(QBrush(QColor(TH['muted'])))
        p.setPen(QPen(QColor(TH['border2']), 1.5)); p.drawEllipse(r); p.end()


class Spark(QWidget):
    """Tiny line chart inside a KPI tile."""

    def __init__(self, color):
        super().__init__(); self.color = QColor(color); self.data = []; self.setFixedHeight(26)

    def set(self, data):
        self.data = [float(v) for v in list(data)[-80:] if v is not None]; self.update()

    def paintEvent(self, e):
        if len(self.data) < 2:
            return
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        lo, hi = min(self.data), max(self.data)
        rng = (hi - lo) or 1.0
        pts = [QPointF(i * (W - 2) / (len(self.data) - 1) + 1, H - 3 - (v - lo) / rng * (H - 6)) for i, v in enumerate(self.data)]
        path = QPainterPath(pts[0])
        for q in pts[1:]:
            path.lineTo(q)
        area = QPainterPath(path); area.lineTo(pts[-1].x(), H); area.lineTo(pts[0].x(), H); area.closeSubpath()
        g = QLinearGradient(0, 0, 0, H); g.setColorAt(0, qc(self.color, 90)); g.setColorAt(1, qc(self.color, 0))
        p.fillPath(area, QBrush(g))
        p.setPen(QPen(self.color, 1.8)); p.drawPath(path)
        p.setBrush(self.color); p.drawEllipse(pts[-1], 2.5, 2.5)
        p.end()


class KpiTile(QFrame):
    def __init__(self, title_key, color=None, spark=False, icon=''):
        super().__init__(); self.setObjectName('Kpi')
        self.accent = QColor(color or TH['accent']); self.color = color
        lay = QVBoxLayout(self); lay.setContentsMargins(14, 9, 12, 8); lay.setSpacing(1)
        self.icon = icon
        self.title = QLabel(); self.title.setObjectName('KpiTitle'); self.tkey = title_key
        self.value = QLabel('—'); self.value.setObjectName('KpiValue')
        if color:
            self.value.setStyleSheet(f'color: {color};')
        lay.addWidget(self.title); lay.addWidget(self.value)
        self.spark = Spark(color or ORANGE) if spark else None
        if self.spark:
            lay.addWidget(self.spark)
        self.retranslate()

    def retranslate(self):
        self.title.setText((self.icon + ' ' if self.icon else '') + tr.t(self.tkey))

    def set(self, text, color=None):
        self.value.setText(text)
        if color and color != self.color:
            self.color = color; self.accent = QColor(color); self.value.setStyleSheet(f'color: {color};'); self.update()

    def set_spark(self, data):
        if self.spark:
            self.spark.set(data)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(self.accent)
        p.drawRoundedRect(QRectF(0, 10, 4, self.height() - 20), 2, 2); p.end()


class ProbBars(QWidget):
    """Horizontal bars with class probabilities."""

    def __init__(self):
        super().__init__(); self.items = []; self.setMinimumHeight(150)

    def set(self, items):
        self.items = items; self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.items:
            p.end(); return
        h = self.height() / len(self.items); lw = 130
        for k, (name, v, col) in enumerate(self.items):
            y = k * h + 4; bh = h - 8
            p.setPen(QColor(TH['text'])); p.setFont(QFont('Segoe UI', 9))
            p.drawText(QRectF(0, y, lw - 8, bh), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, name)
            full = self.width() - lw - 64
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(TH['header'])); p.drawRoundedRect(QRectF(lw, y, full, bh), 5, 5)
            g = QLinearGradient(lw, 0, lw + full, 0); g.setColorAt(0, QColor(col).darker(115)); g.setColorAt(1, QColor(col).lighter(125))
            p.setBrush(QBrush(g)); p.drawRoundedRect(QRectF(lw, y, max(3, full * v), bh), 5, 5)
            p.setPen(QColor(TH['text'])); p.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
            p.drawText(QRectF(lw + full + 6, y, 60, bh), Qt.AlignmentFlag.AlignVCenter, f'{v * 100:.1f} %'.replace('.', ','))
        p.end()


# =====================================================================
# Dashboard elements
# =====================================================================
class Gauge(QWidget):
    """Speedometer-type gauge (270° arc) with colour zones and a needle."""

    def __init__(self, title_key, vmax=100.0, unit='%', zones=None, decimals=1, invert=False):
        super().__init__()
        self.tkey, self.vmax, self.unit, self.dec = title_key, vmax, unit, decimals
        # zones: list of (upper fraction, colour); default: higher = better
        self.zones = zones or ([(0.5, RED), (0.75, YELLOW), (1.0, GREEN)] if not invert else [(0.25, GREEN), (0.5, YELLOW), (1.0, RED)])
        self.value = None; self.shown = 0.0
        self.setMinimumSize(170, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set(self, v):
        self.value = v
        target = 0.0 if v is None else max(0.0, min(1.0, v / self.vmax))
        self.shown += (target - self.shown) * 0.35          # smooth needle movement
        self.update()

    def _zone_color(self, f):
        for up, c in self.zones:
            if f <= up + 1e-9:
                return c
        return self.zones[-1][1]

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        d = min(W - 16, (H - 30) * 1.15)
        rect = QRectF((W - d) / 2, 8, d, d)
        c = rect.center(); R = d / 2
        thick = max(8, d * 0.09)
        # background arc
        p.setPen(QPen(QColor(TH['header']), thick, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        inner = rect.adjusted(thick, thick, -thick, -thick)
        p.drawArc(inner, 225 * 16, -270 * 16)
        # zone ticks (thin outer arc)
        lo = 0.0
        for up, col in self.zones:
            p.setPen(QPen(qc(col, 160), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            p.drawArc(rect.adjusted(3, 3, -3, -3), int((225 - 270 * lo) * 16), int(-270 * (up - lo) * 16))
            lo = up
        # value arc
        f = self.shown
        col = QColor(self._zone_color(f))
        if f > 0.002:
            p.setPen(QPen(col, thick, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(inner, 225 * 16, int(-270 * f * 16))
        # scale labels
        p.setPen(QColor(TH['sub'])); p.setFont(QFont('Segoe UI', 7))
        for k in range(1, 5):
            fr = k / 5; ang = math.radians(225 - 270 * fr)
            rr = R - thick * 2.3
            pt = QPointF(c.x() + rr * math.cos(ang), c.y() - rr * math.sin(ang))
            val = self.vmax * fr
            p.drawText(QRectF(pt.x() - 16, pt.y() - 7, 32, 14), Qt.AlignmentFlag.AlignCenter, f'{val:.0f}')
        # needle
        ang = math.radians(225 - 270 * f)
        tip = QPointF(c.x() + (R - thick * 1.2) * math.cos(ang), c.y() - (R - thick * 1.2) * math.sin(ang))
        p.setPen(QPen(QColor(TH['text']), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)); p.drawLine(c, tip)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(col); p.drawEllipse(c, 6, 6)
        # value
        p.setPen(QColor(TH['text'])); p.setFont(QFont('Segoe UI', max(10, int(d * 0.11)), QFont.Weight.Black))
        txt = '—' if self.value is None else (f'{self.value:.{self.dec}f}'.replace('.', ',') + (' ' + self.unit if self.unit else ''))
        p.drawText(QRectF(rect.left(), c.y() + R * 0.28, d, R * 0.45), Qt.AlignmentFlag.AlignCenter, txt)
        p.setPen(QColor(TH['sub'])); p.setFont(QFont('Segoe UI', 9, QFont.Weight.DemiBold))
        title = p.fontMetrics().elidedText(tr.t(self.tkey), Qt.TextElideMode.ElideRight, W - 4)
        p.drawText(QRectF(0, H - 22, W, 20), Qt.AlignmentFlag.AlignCenter, title)
        p.end()


class Donut(QWidget):
    """Donut chart with legend."""

    def __init__(self, center_key=None):
        super().__init__(); self.items = []; self.center_key = center_key
        self.setMinimumSize(260, 170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set(self, items):
        self.items = [(n, float(v), c) for n, v, c in items]; self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        tot = sum(v for _, v, _ in self.items)
        p.setFont(QFont('Segoe UI', 9))
        fm = p.fontMetrics()
        lw = max([fm.horizontalAdvance(f'{n}:  {int(v)}  (100,0 %)') for n, v, _ in self.items] + [80]) + 30
        d = max(70.0, min(H - 16, W - lw - 20))
        rect = QRectF(10, (H - d) / 2, d, d)
        if tot <= 0:
            p.setPen(QPen(QColor(TH['header']), d * 0.18)); p.drawEllipse(rect.adjusted(d * .09, d * .09, -d * .09, -d * .09))
        else:
            a0 = 90 * 16
            for n, v, col in self.items:
                span = -int(round(v / tot * 360 * 16))
                p.setPen(QPen(QColor(TH['card']), 2)); p.setBrush(QColor(col)); p.drawPie(rect, a0, span); a0 += span
        hole = rect.adjusted(d * .2, d * .2, -d * .2, -d * .2)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(TH['card'])); p.drawEllipse(hole)
        p.setPen(QColor(TH['text'])); p.setFont(QFont('Segoe UI', max(10, int(d * 0.13)), QFont.Weight.Black))
        p.drawText(hole.adjusted(0, -d * .06, 0, -d * .06), Qt.AlignmentFlag.AlignCenter, str(int(tot)))
        if self.center_key:
            p.setPen(QColor(TH['sub'])); p.setFont(QFont('Segoe UI', 8))
            p.drawText(hole.adjusted(0, d * .16, 0, d * .16), Qt.AlignmentFlag.AlignCenter, tr.t(self.center_key))
        # legend
        lx = rect.right() + 16; n = max(1, len(self.items)); lh = min(24, (H - 10) / n)
        y = (H - lh * n) / 2
        for name, v, col in self.items:
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(col)); p.drawRoundedRect(QRectF(lx, y + lh / 2 - 6, 12, 12), 3, 3)
            p.setPen(QColor(TH['text'])); p.setFont(QFont('Segoe UI', 9))
            pct = f'{v / tot * 100:.1f} %'.replace('.', ',') if tot else '0 %'
            txt = fm.elidedText(f'{name}:  {int(v)}  ({pct})', Qt.TextElideMode.ElideRight, int(W - lx - 20))
            p.drawText(QRectF(lx + 18, y, W - lx - 18, lh), Qt.AlignmentFlag.AlignVCenter, txt)
            y += lh
        p.end()


class StateTimeline(QWidget):
    """Gantt-like strip of line states over the last `span` seconds."""

    def __init__(self, ctrl, span=300):
        super().__init__(); self.ctrl = ctrl; self.span = span; self.setMinimumHeight(84)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        now = time.time(); t0 = now - self.span
        bar = QRectF(4, 4, W - 8, max(16, H - 56))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(TH['header'])); p.drawRoundedRect(bar, 6, 6)
        for st, a, b in self.ctrl.states.segments:
            if b < t0:
                continue
            x0 = bar.left() + max(0, (a - t0) / self.span) * bar.width()
            x1 = bar.left() + min(1, (b - t0) / self.span) * bar.width()
            p.setBrush(QColor(STATE_COLORS.get(st, GRAY))); p.drawRect(QRectF(x0, bar.top(), max(1.0, x1 - x0), bar.height()))
        p.setPen(QColor(TH['sub'])); p.setFont(QFont('Segoe UI', 7))
        for k in range(6):
            x = bar.left() + k / 5 * bar.width()
            lab = tr.t('now') if k == 5 else f'-{round(self.span * (1 - k / 5))} s'
            al = Qt.AlignmentFlag.AlignLeft if k == 0 else (Qt.AlignmentFlag.AlignRight if k == 5 else Qt.AlignmentFlag.AlignHCenter)
            rx = x if k == 0 else (x - 60 if k == 5 else x - 30)
            p.drawText(QRectF(rx, bar.bottom() + 1, 60, 12), al | Qt.AlignmentFlag.AlignVCenter, lab)
        x = 6; y = bar.bottom() + 16
        p.setFont(QFont('Segoe UI', 8))
        for st in ('run', 'inspect', 'stop', 'estop', 'manual'):
            txt = tr.t('state_' + st); tw = p.fontMetrics().horizontalAdvance(txt)
            if x + 18 + tw > W:
                x = 6; y += 17
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(STATE_COLORS[st])); p.drawRoundedRect(QRectF(x, y + 3, 10, 10), 2, 2)
            p.setPen(QColor(TH['text']))
            p.drawText(QRectF(x + 14, y, tw + 4, 16), Qt.AlignmentFlag.AlignVCenter, txt)
            x += 26 + tw
        p.end()


class SignalTrace(QWidget):
    """Logic analyser (timing diagram) of PLC signals — shows the ImageRequest → ResultReady handshake."""
    SIG = [('%I0.3', 'sensor', YELLOW), ('DB.Req', 'ImageRequest', BLUE), ('DB.Ready', 'ResultReady', PURPLE),
           ('%Q0.0', 'motor', GREEN), ('%Q0.1', 'pusher', ORANGE), ('%Q0.2', 'green', GREEN), ('%Q0.3', 'red', RED), ('%Q0.4', 'alarm', RED)]

    def __init__(self, ctrl, span=12.0):
        super().__init__(); self.ctrl = ctrl; self.span = span; self.setMinimumHeight(220)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor(TH['card2']))
        lw = 70; n = len(self.SIG); rh = (H - 18) / n
        now = time.time(); t0 = now - self.span
        tr_ = list(self.ctrl.trace)
        p.setFont(QFont('Consolas', 8, QFont.Weight.Bold))
        for k, (lab, key, col) in enumerate(self.SIG):
            y = k * rh + 2
            p.setPen(QColor(TH['sub'])); p.drawText(QRectF(4, y, lw - 8, rh), Qt.AlignmentFlag.AlignVCenter, lab)
            p.setPen(QPen(QColor(TH['grid']), 1)); p.drawLine(QPointF(lw, y + rh - 2), QPointF(W - 4, y + rh - 2))
            if not tr_:
                continue
            path = QPainterPath(); first = True
            hi_y, lo_y = y + rh * 0.2, y + rh - 4
            prev = None
            for t, sig in tr_:
                if t < t0:
                    continue
                x = lw + (t - t0) / self.span * (W - lw - 6)
                yy = hi_y if sig.get(key) else lo_y
                if first:
                    path.moveTo(x, yy); first = False
                else:
                    if prev is not None and prev != yy:
                        path.lineTo(x, prev)
                    path.lineTo(x, yy)
                prev = yy
            p.setPen(QPen(QColor(col), 2)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawPath(path)
        p.setPen(QColor(TH['sub'])); p.setFont(QFont('Segoe UI', 7))
        for k in range(7):
            x = lw + k / 6 * (W - lw - 6)
            p.drawText(QRectF(x - 25, H - 14, 50, 12), Qt.AlignmentFlag.AlignCenter, f'-{self.span * (1 - k / 6):.0f} s' if k < 6 else tr.t('now'))
        p.end()


# =====================================================================
# Pixmaps
# =====================================================================
class _PixCache:
    def __init__(self):
        self.cache = {}

    def get(self, path):
        pm = self.cache.get(path)
        if pm is None:
            pm = QPixmap(path)
            if len(self.cache) > 800:
                self.cache.clear()
            self.cache[path] = pm
        return pm


PIX = _PixCache()


def _jet(v):
    """Simple 'jet' colour map, v in 0..1 → RGB uint8."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def heat_pixmap(path, heat, size=260, alpha=0.5):
    """Original image + colour heat map of the occlusion sensitivity (where the model 'looks')."""
    img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    h = Image.fromarray((np.asarray(heat, dtype=np.float32) * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    hv = np.asarray(h, dtype=np.float32) / 255.0
    col = _jet(hv).astype(np.float32)
    a = (alpha * np.clip(hv * 1.4, 0.15, 1.0))[..., None]
    out = (np.asarray(img, dtype=np.float32) * (1 - a) + col * a).astype(np.uint8)
    out = np.ascontiguousarray(out)
    qi = QImage(out.data, size, size, 3 * size, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qi)


# =====================================================================
# Process mimics
# =====================================================================
class ConveyorView(QWidget):
    """Process mimic: conveyor, photo sensor, camera with AI, pusher, reject chute, lamp tower."""

    def __init__(self, ctrl, compact=False):
        super().__init__(); self.ctrl = ctrl
        self.setMinimumSize(640 if compact else 760, 300 if compact else 330)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def bx(self, x):
        x0, x1 = 50, self.width() - 150
        return x0 + x * (x1 - x0)

    def paintEvent(self, e):
        c = self.ctrl; sim, plc = c.sim, c.plc
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        g = QLinearGradient(0, 0, 0, H); g.setColorAt(0, QColor(TH['canvas_top'])); g.setColorAt(1, QColor(TH['canvas_bot']))
        p.fillRect(self.rect(), QBrush(g))
        # floor grid
        p.setPen(QPen(qc(TH['grid'], 160), 1))
        for gx in range(0, W, 40):
            p.drawLine(QPointF(gx, H - 16), QPointF(gx + 20, H))
        belt_y = H * 0.56; belt_h = 26; x0, x1 = self.bx(0), self.bx(1)
        cam_x = self.bx(ProcessSim.CAMERA_X); item = 58
        # frame legs
        p.setPen(QPen(QColor(TH['belt']), 5))
        for lx in (x0 + 20, (x0 + x1) / 2, x1 - 20):
            p.drawLine(QPointF(lx, belt_y + belt_h), QPointF(lx, H - 18))
        # reject chute
        chute = QPolygonF([QPointF(cam_x - 40, belt_y + belt_h + 2), QPointF(cam_x + 40, belt_y + belt_h + 2),
                           QPointF(cam_x + 60, H - 40), QPointF(cam_x - 60, H - 40)])
        p.setPen(QPen(QColor('#B71C1C'), 1.5)); p.setBrush(QColor(211, 47, 47, 45)); p.drawPolygon(chute)
        p.setPen(QColor(RED)); p.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        p.drawText(QRectF(cam_x - 90, H - 38, 180, 20), Qt.AlignmentFlag.AlignCenter, tr.t('bin_bad') + f': {sim.n_bad}')
        # belt
        bg = QLinearGradient(0, belt_y, 0, belt_y + belt_h)
        bg.setColorAt(0, QColor(TH['belt']).lighter(125)); bg.setColorAt(1, QColor(TH['belt']).darker(130))
        p.setPen(QPen(QColor('#263238'), 2)); p.setBrush(QBrush(bg))
        p.drawRoundedRect(QRectF(x0 - 14, belt_y, x1 - x0 + 28, belt_h), 13, 13)
        p.setPen(QPen(QColor(TH['belt_line']), 2))
        off = (sim.travel * (x1 - x0)) % 24
        xx = x0 + off
        while xx < x1:
            p.drawLine(QPointF(xx, belt_y + 3), QPointF(xx - 6, belt_y + belt_h - 3)); xx += 24
        # rotating rollers
        ang = sim.travel * 40
        for rx in (x0, x1):
            ctr = QPointF(rx, belt_y + belt_h / 2)
            p.setBrush(QColor('#455A64')); p.setPen(QPen(QColor('#1C262B'), 1.5)); p.drawEllipse(ctr, 13, 13)
            p.setPen(QPen(QColor('#B0BEC5'), 2))
            for k in range(3):
                a = ang + k * 2.094
                p.drawLine(ctr, QPointF(ctr.x() + 10 * math.cos(a), ctr.y() + 10 * math.sin(a)))
        # good bin
        p.setPen(QPen(QColor('#1B5E20'), 1.5)); p.setBrush(QColor(46, 157, 74, 45))
        p.drawRoundedRect(QRectF(x1 + 22, belt_y - 10, 110, 110), 8, 8)
        p.setPen(QColor(GREEN)); p.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        p.drawText(QRectF(x1 + 12, belt_y + 104, 130, 20), Qt.AlignmentFlag.AlignCenter, f'{tr.t("bin_good")}: {sim.n_good}')
        for k, it in enumerate(sim.bin_good[-9:]):
            pm = PIX.get(it.path)
            p.drawPixmap(QRectF(x1 + 28 + (k % 3) * 33, belt_y - 4 + (k // 3) * 33, 29, 29), pm, QRectF(pm.rect()))
        # photo sensor
        sx = cam_x - 34
        p.setPen(QPen(QColor('#37474F'), 1)); p.setBrush(QColor('#263238'))
        p.drawRect(QRectF(sx - 6, belt_y - item - 30, 12, 16))
        p.setPen(QPen(QColor(RED) if plc.I['sensor'] else qc(TH['sub'], 140), 2, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(sx, belt_y - item - 14), QPointF(sx, belt_y))
        # items
        for it in sim.items:
            ix = self.bx(it.x) - item / 2; iy = belt_y - item - 2
            if it.status == 'reject_drop':
                iy += it.drop * (H - 40 - belt_y); p.setOpacity(max(0.2, 1 - it.drop * 0.6))
            pm = PIX.get(it.path)
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(0, 0, 0, 60)); p.drawRoundedRect(QRectF(ix + 3, iy + 4, item, item), 4, 4)
            p.drawPixmap(QRectF(ix, iy, item, item), pm, QRectF(pm.rect()))
            p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(QColor(STATUS_COLOR.get(it.status, GRAY)), 3))
            p.drawRoundedRect(QRectF(ix, iy, item, item), 4, 4)
            p.setOpacity(1.0)
            if it.result is not None and it.status != 'reject_drop':
                p.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold)); p.setPen(QColor(STATUS_COLOR.get(it.status, TH['text'])))
                p.drawText(QRectF(ix - 25, iy - 18, item + 50, 16), Qt.AlignmentFlag.AlignCenter,
                           f'{tr.cls(it.result["name"])} {it.result["confidence"] * 100:.0f}%')
            p.setPen(QColor(TH['sub'])); p.setFont(QFont('Segoe UI', 7))
            p.drawText(QRectF(ix, iy + item + 1, item, 12), Qt.AlignmentFlag.AlignCenter, f'#{it.id}')
        # camera + flash
        analysing = plc.DB['ImageRequest'] or c.pending
        if analysing:
            cone = QPolygonF([QPointF(cam_x - 8, 64), QPointF(cam_x + 8, 64), QPointF(cam_x + 42, belt_y - 2), QPointF(cam_x - 42, belt_y - 2)])
            cg = QLinearGradient(0, 64, 0, belt_y); cg.setColorAt(0, QColor(255, 235, 59, 150)); cg.setColorAt(1, QColor(255, 235, 59, 30))
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(cg)); p.drawPolygon(cone)
        p.setPen(QPen(QColor('#102027'), 1.5)); p.setBrush(QColor('#37474F'))
        p.drawRoundedRect(QRectF(cam_x - 38, 18, 76, 42), 7, 7)
        p.setBrush(QColor(BLUE) if analysing else QColor('#90A4AE')); p.drawEllipse(QPointF(cam_x, 60), 9, 9)
        p.setPen(QColor('white')); p.setFont(QFont('Segoe UI', 8, QFont.Weight.Bold))
        p.drawText(QRectF(cam_x - 38, 18, 76, 30), Qt.AlignmentFlag.AlignCenter, 'AI CAM')
        # pusher
        ext = 1.0 if plc.Q['pusher'] else 0.0
        p.setPen(QPen(QColor('#E65100'), 1.5)); p.setBrush(QColor('#FFB300'))
        p.drawRect(QRectF(cam_x + 44, belt_y - 66, 30, 22))
        p.setBrush(QColor('#FFCA28')); p.drawRect(QRectF(cam_x + 36 - ext * 30, belt_y - 60, 10 + ext * 30, 10))
        p.setPen(QColor('#FF9800')); p.setFont(QFont('Segoe UI', 8))
        p.drawText(QRectF(cam_x + 40, belt_y - 86, 140, 16), Qt.AlignmentFlag.AlignLeft, tr.t('pusher') + ' %Q0.1')
        # lamp tower
        tx = W - 26
        p.setPen(QPen(QColor('#455A64'), 3)); p.drawLine(QPointF(tx, 30), QPointF(tx, 130))
        blink = int(time.time() * 3) % 2 == 0
        for k, (col, on) in enumerate([(RED, plc.Q['red'] or (plc.Q['alarm'] and blink)), (YELLOW, analysing), (GREEN, plc.Q['green'] or plc.Q['motor'])]):
            r = QRectF(tx - 13, 18 + k * 28, 26, 26)
            if on:
                gl = QRadialGradient(r.center(), 22); gl.setColorAt(0, qc(col, 150)); gl.setColorAt(1, qc(col, 0))
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(gl)); p.drawEllipse(r.adjusted(-10, -10, 10, 10))
            p.setPen(QPen(QColor('#263238'), 1)); p.setBrush(QColor(col) if on else QColor(TH['muted']))
            p.drawRoundedRect(r, 5, 5)
        # captions
        p.setPen(QColor(TH['text'])); p.setFont(QFont('Segoe UI', 9))
        motor_txt = f'{tr.t("conveyor")} %Q0.0: ' + ('ON' if plc.Q['motor'] else 'OFF') + \
                    f'   ·   v = {float(c.settings["speed"]):.2f}'.replace('.', ',')
        p.drawText(QRectF(x0, belt_y + belt_h + 6, 360, 18), Qt.AlignmentFlag.AlignLeft, motor_txt)
        p.drawText(QRectF(sx - 60, belt_y - item - 50, 120, 16), Qt.AlignmentFlag.AlignCenter, '%I0.3')
        if plc.Q['alarm']:
            p.setPen(QColor(RED)); p.setFont(QFont('Segoe UI', 18, QFont.Weight.Black))
            if blink:
                p.drawText(QRectF(0, 70, W, 40), Qt.AlignmentFlag.AlignCenter, '⚠ ' + tr.t('st_estop') + ' ⚠')
        p.end()


class BinView(QWidget):
    """Sorting window: two bins with thumbnails of the latest items."""

    def __init__(self, ctrl):
        super().__init__(); self.ctrl = ctrl; self.setMinimumSize(620, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, e):
        sim = self.ctrl.sim
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(self.rect(), QColor(TH['card2']))
        for k, (items, n, key, col) in enumerate([(sim.bin_good, sim.n_good, 'bin_good', GREEN), (sim.bin_bad, sim.n_bad, 'bin_bad', RED)]):
            bx = 16 + k * (W / 2); bw = W / 2 - 32
            path = QPainterPath(); path.addRoundedRect(QRectF(bx, 40, bw, H - 56), 12, 12)
            p.setPen(QPen(QColor(col), 2)); p.setBrush(qc(col, 28)); p.drawPath(path)
            p.setPen(QColor(col)); p.setFont(QFont('Segoe UI', 12, QFont.Weight.Bold))
            p.drawText(QRectF(bx, 8, bw, 28), Qt.AlignmentFlag.AlignCenter, f'{tr.t(key)}: {n}')
            cell = 46; cols = max(1, int((bw - 16) // cell))
            for j, it in enumerate(reversed(items[-cols * 6:])):
                r, cc = divmod(j, cols)
                rect = QRectF(bx + 10 + cc * cell, 52 + r * cell, cell - 6, cell - 6)
                if rect.bottom() > H - 20:
                    break
                pm = PIX.get(it.path); p.drawPixmap(rect, pm, QRectF(pm.rect()))
                if it.result is not None and it.truth:
                    ok = (it.result['name'] == 'good') == (it.truth == 'good')
                    p.setPen(QPen(QColor(GREEN if ok else '#FF6F00'), 2 if ok else 3)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRect(rect)
        p.end()


# =====================================================================
# Base class for all secondary windows (persistent windows, show / hide)
# =====================================================================
class BaseWindow(QWidget):
    title_key = ''
    icon = ''

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl; page(self)
        self.setWindowTitle(tr.t(self.title_key))
        ctrl.state_changed.connect(self.retranslate)

    def retranslate(self):
        self.setWindowTitle(tr.t(self.title_key)); retranslate_tree(self); retranslate_cards(self)
        for k in self.findChildren(KpiTile):
            k.retranslate()
        self.update()
        for w in self.findChildren(QWidget):
            w.update()

    def toggle(self):
        if self.isVisible():
            self.hide()
        else:
            self.show(); self.raise_(); self.activateWindow()
