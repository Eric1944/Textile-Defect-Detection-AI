"""Tekstila kvalitātes kontroles sistēma — palaišana / запуск / launch:  python main.py"""
import sys

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QLinearGradient, QColor, QFont, QPen
from PyQt6.QtWidgets import QApplication, QSplashScreen

from textile_scada.core import Controller, load_settings
from textile_scada.i18n import tr
from textile_scada.theme import TH, build_qss


def splash_pixmap():
    pm = QPixmap(720, 380); pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    g = QLinearGradient(0, 0, 720, 380); g.setColorAt(0, QColor('#0E141B')); g.setColorAt(1, QColor('#24384A'))
    p.setBrush(g); p.setPen(Qt.PenStyle.NoPen); p.drawRoundedRect(QRectF(0, 0, 720, 380), 22, 22)
    p.setPen(QPen(QColor('#FF8A3D'), 3))
    for k in range(14):                               # stylised fabric threads
        p.drawLine(470 + k * 16, 40, 420 + k * 16, 340)
    p.setPen(QColor('#FF8A3D')); p.setFont(QFont('Segoe UI', 13, QFont.Weight.Bold)); p.drawText(40, 70, 'RTU · Datorvadība ražošanas sistēmās')
    p.setPen(QColor('white')); p.setFont(QFont('Segoe UI', 26, QFont.Weight.Black))
    p.drawText(QRectF(40, 90, 460, 130), Qt.TextFlag.TextWordWrap, tr.t('app_title'))
    p.setPen(QColor('#A9BAC9')); p.setFont(QFont('Segoe UI', 11)); p.drawText(QRectF(40, 230, 440, 60), Qt.TextFlag.TextWordWrap, tr.t('app_sub'))
    p.end()
    return pm


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    s = load_settings(); tr.lang = s.get('lang', 'lv'); TH.set(s.get('theme', 'dark'))
    app.setStyleSheet(build_qss())
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True)
    sp = QSplashScreen(splash_pixmap()); sp.show()
    sp.showMessage('   PyTorch / TorchScript …', Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, QColor('white'))
    app.processEvents()
    ctrl = Controller()
    sp.showMessage('   UI …', Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft, QColor('white')); app.processEvents()
    from textile_scada.windows import MainWindow
    win = MainWindow(ctrl)
    win.show(); sp.finish(win)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
