"""Colour themes (dark SCADA / light) and the Qt style sheet built from them."""
import weakref

ORANGE = '#E07020'
GREEN = '#2E9D4A'
RED = '#D32F2F'
BLUE = '#1E88E5'
YELLOW = '#F9A825'
GRAY = '#90A4AE'
PURPLE = '#8E24AA'
BROWN = '#8D6E63'
TEAL = '#00897B'

CLASS_COLORS = {'good': GREEN, 'hole': PURPLE, 'objects': BLUE, 'oil spot': BROWN, 'thread error': ORANGE}
STATE_COLORS = {'run': GREEN, 'inspect': YELLOW, 'stop': GRAY, 'estop': RED, 'manual': BLUE}

PALETTES = {
    'dark': dict(bg='#0E141B', card='#16202A', card2='#1B2733', border='#253341', border2='#33475A', text='#E4EAF0',
                 sub='#8DA2B5', accent='#FF8A3D', input='#101921', header='#1E2B38', btn='#1B2733', btn_hover='#24384A',
                 btn_press='#2C4459', muted='#34495C', grid='#22313F', alt='#19242F', plot_bg='#16202A', plot_fg='#A9BAC9',
                 canvas_top='#1A2633', canvas_bot='#0F171F', belt='#4E6272', belt_line='#3B4C5A', status='#080C10'),
    'light': dict(bg='#F2F4F7', card='#FFFFFF', card2='#F8FAFC', border='#DDE3EA', border2='#C8D2DC', text='#22303A',
                  sub='#5F7686', accent='#C55A11', input='#FFFFFF', header='#ECEFF3', btn='#FFFFFF', btn_hover='#FFF4EB',
                  btn_press='#FDE3CF', muted='#CFD8DC', grid='#ECEFF1', alt='#F7F9FB', plot_bg='#FFFFFF', plot_fg='#37474F',
                  canvas_top='#FAFBFC', canvas_bot='#E6ECF1', belt='#78909C', belt_line='#5F7885', status='#263238'),
}


class Theme:
    def __init__(self, name='dark'):
        self.set(name)

    def set(self, name):
        self.name = name if name in PALETTES else 'dark'
        self.c = PALETTES[self.name]

    def __getitem__(self, k):
        return self.c[k]

    @property
    def dark(self):
        return self.name == 'dark'


TH = Theme('dark')

# ---------------------------------------------------------------- pyqtgraph plots follow the theme
_plots = weakref.WeakSet()


def style_plot(pw, grid=(True, True)):
    _plots.add(pw)
    pw._grid = grid
    _apply_plot(pw)
    return pw


def _apply_plot(pw):
    import pyqtgraph as pg
    pw.setBackground(TH['plot_bg'])
    for ax in ('left', 'bottom', 'right', 'top'):
        a = pw.getAxis(ax)
        a.setPen(pg.mkPen(TH['plot_fg'])); a.setTextPen(pg.mkPen(TH['plot_fg']))
    pw.showGrid(x=pw._grid[0], y=pw._grid[1], alpha=0.25)
    leg = pw.getPlotItem().legend
    if leg is not None:
        leg.setBrush(TH['card2']); leg.setPen(TH['border2']); leg.setLabelTextColor(TH['text'])


def restyle_plots():
    for pw in list(_plots):
        try:
            _apply_plot(pw)
        except RuntimeError:          # widget already deleted
            pass


def build_qss():
    c = TH.c
    return f"""
QWidget {{ font-family: 'Segoe UI', 'Inter', 'DejaVu Sans', Arial; font-size: 10pt; color: {c['text']}; }}
QMainWindow, QWidget#Page, QDialog {{ background: {c['bg']}; }}
QLabel {{ background: transparent; }}
QFrame#Card {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 12px; }}
QFrame#Kpi {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 12px; }}
QLabel#Title {{ font-size: 19pt; font-weight: 800; color: {c['text']}; }}
QLabel#Sub {{ color: {c['sub']}; }}
QLabel#CardTitle {{ font-size: 11pt; font-weight: 700; color: {c['accent']}; }}
QLabel#KpiTitle {{ color: {c['sub']}; font-size: 8.5pt; font-weight: 600; }}
QLabel#KpiValue {{ font-size: 17pt; font-weight: 800; }}
QLabel#Big {{ font-size: 22pt; font-weight: 800; }}
QLabel#Mono {{ font-family: Consolas, 'DejaVu Sans Mono', monospace; font-weight: 700; }}
QPushButton {{ background: {c['btn']}; border: 1px solid {c['border2']}; border-radius: 8px; padding: 8px 14px; color: {c['text']}; }}
QPushButton:hover {{ border-color: {c['accent']}; background: {c['btn_hover']}; }}
QPushButton:pressed {{ background: {c['btn_press']}; }}
QPushButton:disabled {{ color: {c['muted']}; background: {c['card2']}; border-color: {c['border']}; }}
QPushButton#Nav {{ text-align: left; padding: 11px 16px; font-size: 10.5pt; font-weight: 600; }}
QPushButton#Nav:checked {{ border: 2px solid {c['accent']}; background: {c['btn_hover']}; }}
QPushButton#Start {{ background: {GREEN}; color: white; font-weight: 800; border: none; font-size: 12pt; }}
QPushButton#Start:hover {{ background: #37B257; }}
QPushButton#Stop {{ background: #EF6C00; color: white; font-weight: 800; border: none; font-size: 12pt; }}
QPushButton#Stop:hover {{ background: #FB8C00; }}
QPushButton#EStop {{ background: {RED}; color: white; font-weight: 900; border: 3px solid #FFEB3B; border-radius: 12px; font-size: 12pt; }}
QPushButton#EStop:checked {{ background: #7F0000; }}
QPushButton#Accent {{ background: {c['accent']}; color: white; font-weight: 700; border: none; }}
QPushButton#Accent:hover {{ background: #FF9D5C; }}
QPushButton#Tool {{ padding: 6px 10px; font-size: 13pt; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {c['input']}; border: 1px solid {c['border2']}; border-radius: 6px; padding: 5px 8px; color: {c['text']}; }}
QComboBox:hover, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {c['accent']}; }}
QComboBox QAbstractItemView {{ background: {c['input']}; color: {c['text']}; selection-background-color: {c['accent']}; border: 1px solid {c['border2']}; }}
QTableWidget {{ background: {c['card']}; alternate-background-color: {c['alt']}; border: 1px solid {c['border']}; border-radius: 8px;
               gridline-color: {c['grid']}; selection-background-color: {c['accent']}; selection-color: white; }}
QHeaderView::section {{ background: {c['header']}; color: {c['text']}; border: none; padding: 6px; font-weight: 700; }}
QTableCornerButton::section {{ background: {c['header']}; border: none; }}
QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 10px; background: {c['bg']}; top: -1px; }}
QTabBar::tab {{ background: {c['header']}; color: {c['sub']}; padding: 9px 18px; border-top-left-radius: 8px;
               border-top-right-radius: 8px; margin-right: 3px; font-weight: 600; }}
QTabBar::tab:selected {{ background: {c['card']}; color: {c['accent']}; font-weight: 800; }}
QTabBar::tab:hover {{ color: {c['text']}; }}
QProgressBar {{ border: 1px solid {c['border2']}; border-radius: 7px; text-align: center; background: {c['header']}; height: 18px; color: {c['text']}; font-weight: 700; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 6px; }}
QProgressBar#Order::chunk {{ background: {GREEN}; border-radius: 6px; }}
QStatusBar {{ background: {c['status']}; color: white; }}
QStatusBar QLabel {{ color: #E0E6EB; }}
QMenuBar {{ background: {c['card']}; color: {c['text']}; border-bottom: 1px solid {c['border']}; }}
QMenuBar::item:selected {{ background: {c['header']}; }}
QMenu {{ background: {c['card']}; color: {c['text']}; border: 1px solid {c['border2']}; }}
QMenu::item {{ padding: 6px 24px; }}
QMenu::item:selected {{ background: {c['accent']}; color: white; }}
QListWidget {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 8px; }}
QListWidget::item:selected {{ background: {c['btn_hover']}; color: {c['text']}; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c['border2']}; border-radius: 5px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {c['border2']}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QSlider::groove:horizontal {{ height: 6px; background: {c['header']}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; border: 3px solid {c['accent']}; width: 12px; margin: -6px 0; border-radius: 9px; }}
QCheckBox {{ spacing: 8px; }}
QFrame#Alarm {{ background: {RED}; border-radius: 10px; }}
QFrame#Alarm QLabel {{ color: white; font-weight: 800; font-size: 11pt; }}
QFrame#Alarm QPushButton {{ background: white; color: {RED}; font-weight: 800; border: none; }}
QToolTip {{ background: {c['card']}; color: {c['text']}; border: 1px solid {c['border2']}; padding: 4px; }}
QSplitter::handle {{ background: {c['border']}; }}
"""
