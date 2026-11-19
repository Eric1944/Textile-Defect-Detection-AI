"""Core of the application: model, PLC (simulated / OPC UA), process simulation, statistics,
OEE / state tracking, alarms, production order, batch inspection and the controller."""
import csv, json, os, random, time, warnings, datetime
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from .i18n import tr

APP_DIR = Path(__file__).resolve().parent.parent
SETTINGS_FILE = APP_DIR / 'settings.json'
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

DEFAULTS = {
    'lang': 'lv',
    'theme': 'dark',
    'model_dir': str(APP_DIR / 'model'),
    'threshold': None,            # None → value from model_meta.json
    'speed': 0.10,                # belt length per second (0..1 = whole belt)
    'spacing': 0.24,              # distance between items on the belt
    'opc_endpoint': 'opc.tcp://192.168.0.1:4840',
    'opc_ns': 3,
    'opc_db': 'DB_Vision',
    # alarms
    'alarm_rate': 50.0,           # % defects in the last 20 items
    'alarm_consec': 5,            # defects in a row
    'alarm_latency': 500.0,       # ms
    'drift_margin': 0.15,         # live recall below test recall − margin → drift
    # production order
    'order_target': 50,
    'order_autostop': True,
}


def load_settings():
    s = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            s.update(json.loads(SETTINGS_FILE.read_text(encoding='utf-8')))
        except Exception:
            pass
    return s


def save_settings(s):
    try:
        SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass


def _f(v):
    """Metric value from model_meta.json → float or None (values can be numbers or strings)."""
    try:
        return float(str(v).replace(',', '.'))
    except Exception:
        return None


def meta_metric(d, *words):
    """Finds a metric in a dict by words in the key, e.g. meta_metric(tm, 'acc')."""
    for k, v in (d or {}).items():
        kl = k.lower()
        if all(w in kl for w in words):
            return _f(v)
    return None


# =====================================================================
# AI model
# =====================================================================
class ModelService:
    """Loads the TorchScript model exported from Colab and classifies images (same code as in the notebook)."""

    def __init__(self):
        self.model = None
        self.meta = None
        self.demo = []          # [(path, true_label)]
        self.names = []

    @property
    def ready(self):
        return self.model is not None

    def load(self, folder):
        import torch
        folder = Path(folder)
        self.meta = json.loads((folder / 'model_meta.json').read_text(encoding='utf-8'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model = torch.jit.load(str(folder / 'textile_model_ts.pt'), map_location='cpu').eval()
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.size = int(self.meta['img_size'])
        self.mean = np.array(self.meta['mean'], dtype=np.float32)
        self.std = np.array(self.meta['std'], dtype=np.float32)
        self.names = list(self.meta['class_names'])
        self.good = int(self.meta['good_class_index'])
        self.default_threshold = float(self.meta['defect_threshold'])
        self.demo = []
        demo_dir = folder / 'demo_images'
        lab = demo_dir / 'labels.csv'
        if lab.exists():
            with open(lab, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    self.demo.append((str(demo_dir / row['file']), row['label']))
        elif demo_dir.exists():
            self.demo = [(str(p), None) for p in sorted(demo_dir.glob('*.png'))]
        self.classify_array(np.zeros((self.size, self.size, 3), dtype=np.uint8), 0.5)   # warm-up

    # -- preprocessing identical to the notebook --
    def _prep(self, rgb):
        img = Image.fromarray(rgb).convert('RGB').resize((self.size, self.size), Image.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)

    def _forward(self, xb):
        import torch
        with torch.no_grad():
            return torch.softmax(self.model(torch.from_numpy(np.ascontiguousarray(xb))), 1).numpy()

    def decide(self, p, threshold):
        p_def = float(1.0 - p[self.good])
        if p_def >= threshold:
            q = np.array(p, dtype=float).copy(); q[self.good] = -1
            return int(q.argmax()), True, p_def
        return self.good, False, p_def

    def classify_array(self, rgb, threshold):
        t0 = time.perf_counter()
        p = self._forward(self._prep(rgb)[None])[0]
        cls, defect, p_def = self.decide(p, threshold)
        return {'probs': [float(v) for v in p], 'cls': cls, 'name': self.names[cls], 'confidence': float(p[cls]),
                'p_defect': p_def, 'defect': defect, 'threshold': threshold,
                'latency_ms': (time.perf_counter() - t0) * 1000}

    def classify(self, path, threshold):
        rgb = np.asarray(Image.open(path).convert('RGB'))
        return self.classify_array(rgb, threshold)

    def explain(self, path, threshold, grid=8):
        """Occlusion sensitivity map: every cell of a grid×grid net is covered by the mean colour and
        the drop of the probability of the predicted class is measured. Big drop = the model looks there."""
        t0 = time.perf_counter()
        rgb = np.asarray(Image.open(path).convert('RGB'))
        x = self._prep(rgb)
        p = self._forward(x[None])[0]
        cls, defect, p_def = self.decide(p, threshold)
        S = self.size; step = S / grid
        batch = []
        for gy in range(grid):
            for gx in range(grid):
                xo = x.copy()
                y0, y1, x0, x1 = int(gy * step), int((gy + 1) * step), int(gx * step), int((gx + 1) * step)
                xo[:, y0:y1, x0:x1] = 0.0          # 0 in normalised space = mean colour
                batch.append(xo)
        batch = np.stack(batch)
        out = []
        try:
            for i in range(0, len(batch), 16):
                out.append(self._forward(batch[i:i + 16]))
            out = np.concatenate(out)
        except Exception:                     # model traced for batch = 1 only
            out = np.concatenate([self._forward(b[None]) for b in batch])
        heat = (p[cls] - out[:, cls]).reshape(grid, grid)
        heat = np.clip(heat, 0, None)
        if heat.max() > 1e-9:
            heat = heat / heat.max()
        return {'heat': heat.tolist(), 'cls': cls, 'name': self.names[cls], 'p_defect': p_def, 'defect': defect,
                'confidence': float(p[cls]), 'path': path, 'grid': grid, 'ms': (time.perf_counter() - t0) * 1000}


class InferenceWorker(QObject):
    """Runs the model in a separate QThread so the windows never freeze (lecture: QThread + signals)."""
    finished = pyqtSignal(dict)
    explained = pyqtSignal(dict)

    def __init__(self, service):
        super().__init__()
        self.service = service

    @pyqtSlot(dict)
    def run(self, job):
        try:
            res = self.service.classify(job['path'], job['threshold'])
            res.update(job)
        except Exception as e:  # noqa
            res = dict(job, error=str(e))
        self.finished.emit(res)

    @pyqtSlot(dict)
    def explain(self, job):
        try:
            res = self.service.explain(job['path'], job['threshold'])
        except Exception as e:  # noqa
            res = dict(job, error=str(e))
        self.explained.emit(res)


class BatchWorker(QObject):
    """Batch inspection of a whole folder in its own thread."""
    progress = pyqtSignal(int, int, dict)
    done = pyqtSignal(float)

    def __init__(self, service):
        super().__init__()
        self.service = service
        self.cancel = False

    @pyqtSlot(list, float)
    def run(self, jobs, threshold):
        self.cancel = False
        t0 = time.perf_counter()
        n = len(jobs)
        for i, (path, truth) in enumerate(jobs):
            if self.cancel:
                break
            try:
                res = self.service.classify(path, threshold)
            except Exception as e:  # noqa
                res = {'error': str(e)}
            res.update(path=path, truth=truth)
            self.progress.emit(i + 1, n, res)
        self.done.emit(time.perf_counter() - t0)


def collect_images(folder, names, recursive=True):
    """Returns [(path, truth)]: labels.csv if present, otherwise the parent folder name if it is a class."""
    folder = Path(folder)
    lab = folder / 'labels.csv'
    if lab.exists():
        try:
            with open(lab, encoding='utf-8') as f:
                rows = [(str(folder / r['file']), r.get('label') or None) for r in csv.DictReader(f)]
            return [r for r in rows if Path(r[0]).exists()]
        except Exception:
            pass
    it = folder.rglob('*') if recursive else folder.glob('*')
    out = []
    for p in sorted(it):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            parent = p.parent.name
            out.append((str(p), parent if parent in names else None))
    return out


# =====================================================================
# Process simulation: conveyor with fabric samples
# =====================================================================
class Item:
    __slots__ = ('id', 'x', 'drop', 'path', 'truth', 'status', 'result')

    def __init__(self, id_, x, path, truth):
        self.id, self.x, self.drop, self.path, self.truth = id_, x, 0.0, path, truth
        self.status = 'new'        # new → wait → good | reject_pending → reject_drop ; manual_wait
        self.result = None


class ProcessSim:
    CAMERA_X = 0.42
    END_X = 1.06

    def __init__(self):
        self.items, self.pool, self.next_id = [], [], 1
        self.travel = 0.0
        self.bin_good, self.bin_bad = [], []
        self.n_good = self.n_bad = 0
        self.push_strokes = 0
        self._push_prev = False
        self._idx = 0

    def set_pool(self, demo):
        self.pool = list(demo)
        random.shuffle(self.pool)
        self._idx = 0

    def _spawn(self):
        if not self.pool:
            return
        path, truth = self.pool[self._idx % len(self.pool)]
        self._idx += 1
        self.items.append(Item(self.next_id, -0.06, path, truth))
        self.next_id += 1

    def reset(self):
        self.items.clear(); self.bin_good.clear(); self.bin_bad.clear()
        self.travel = 0.0; self.push_strokes = 0; self.n_good = self.n_bad = 0

    def step(self, dt, motor, pusher, speed, spacing):
        moved = speed * dt if motor else 0.0
        if pusher and not self._push_prev:
            self.push_strokes += 1
        self._push_prev = pusher
        for it in self.items:
            if it.status == 'reject_drop':
                it.drop += dt * 2.2
            else:
                it.x += moved
            if pusher and it.status == 'reject_pending' and abs(it.x - self.CAMERA_X) < 0.05:
                it.status = 'reject_drop'
        self.travel += moved
        newest = min((it.x for it in self.items if it.status != 'reject_drop'), default=None)   # last spawned item
        if motor and (newest is None or newest - (-0.06) >= spacing):
            self._spawn()
        keep = []
        for it in self.items:
            if it.status == 'reject_drop' and it.drop >= 1.0:
                self.bin_bad.append(it); self.n_bad += 1
            elif it.x > self.END_X:
                self.bin_good.append(it); self.n_good += 1
            else:
                keep.append(it)
        self.items = keep
        self.bin_good = self.bin_good[-60:]
        self.bin_bad = self.bin_bad[-60:]

    def sensor(self):
        return any(it.status == 'new' and abs(it.x - self.CAMERA_X) < 0.012 for it in self.items)

    def item_at_camera(self):
        cands = [it for it in self.items if it.status != 'reject_drop' and abs(it.x - self.CAMERA_X) < 0.04]
        return min(cands, key=lambda it: abs(it.x - self.CAMERA_X)) if cands else None

    def get(self, id_):
        return next((it for it in self.items if it.id == id_), None)


# =====================================================================
# PLC: simulated S7-1500 with the same LAD logic as in TIA Portal
# =====================================================================
class SimPLC:
    kind = 'sim'

    def __init__(self):
        self.I = {'start': False, 'stop_nc': True, 'estop_nc': True, 'sensor': False}
        self.Q = {'motor': False, 'pusher': False, 'green': False, 'red': False, 'alarm': False}
        self.DB = {'ImageRequest': False, 'ResultReady': False, 'Defect': False, 'DefectClass': 0,
                   'Confidence': 0.0, 'GoodCount': 0, 'BadCount': 0}
        self.run = False
        self.manual = False
        self.jog = False
        self.force = {'motor': False, 'pusher': False, 'green': False, 'red': False}
        self._start = self._stop = False
        self._prev_sensor = False
        self.t_push = self.t_green = self.t_red = 0.0
        self.last_active = {k: -10.0 for k in range(1, 6)}
        self.cycle_ms = 0.0
        self.scans = 0
        self.t_ready = 0.0              # time of the last ResultReady (for the logic analyser)
        self.connected = True
        self.error = None

    # operator panel
    def press_start(self): self._start = True
    def press_stop(self): self._stop = True
    def set_estop(self, pressed): self.I['estop_nc'] = not pressed

    def write_result(self, defect, cls, conf):
        self.DB.update(Defect=bool(defect), DefectClass=int(cls), Confidence=float(conf), ResultReady=True)

    def manual_reject(self):
        self.t_push = self.t_red = 1.0; self.DB['BadCount'] += 1; self.last_active[3] = time.time()

    def manual_accept(self):
        self.t_green = 1.0; self.DB['GoodCount'] += 1; self.last_active[4] = time.time()

    def scan(self, dt, sensor):
        t0 = time.perf_counter(); now = time.time()
        I, Q, DB = self.I, self.Q, self.DB
        I['sensor'] = sensor
        I['start'] = self._start
        I['stop_nc'] = not self._stop
        self._start = self._stop = False
        # NW1: start / stop with priority of stop (R after S)
        if I['start'] and I['stop_nc'] and I['estop_nc']:
            self.run = True; self.last_active[1] = now
        if (not I['stop_nc'] or not I['estop_nc']) and self.run:
            self.run = False; self.last_active[1] = now
        if not self.manual:
            # NW2: rising edge of the sensor → image request (belt stops)
            if sensor and not self._prev_sensor and self.run:
                DB['ImageRequest'] = True; self.last_active[2] = now
            # NW3 / NW4: result from the AI
            if DB['ResultReady']:
                self.t_ready = now
                if DB['Defect']:
                    self.t_push = self.t_red = 1.0; DB['BadCount'] += 1; self.last_active[3] = now
                else:
                    self.t_green = 1.0; DB['GoodCount'] += 1; self.last_active[4] = now
                # NW5: acknowledge
                DB['ImageRequest'] = False; DB['ResultReady'] = False; self.last_active[5] = now
        self._prev_sensor = sensor
        self.t_push = max(0.0, self.t_push - dt)
        self.t_green = max(0.0, self.t_green - dt)
        self.t_red = max(0.0, self.t_red - dt)
        if self.manual:
            Q['motor'] = (self.jog or self.force['motor']) and I['estop_nc']
            Q['pusher'] = (self.t_push > 0 or self.force['pusher']) and I['estop_nc']
            Q['green'] = self.t_green > 0 or self.force['green']
            Q['red'] = self.t_red > 0 or self.force['red']
        else:
            Q['motor'] = self.run and not DB['ImageRequest'] and self.t_push <= 0
            Q['pusher'] = self.t_push > 0
            Q['green'] = self.t_green > 0
            Q['red'] = self.t_red > 0
        Q['alarm'] = not I['estop_nc']
        self.scans += 1
        self.cycle_ms = (time.perf_counter() - t0) * 1000

    def close(self):
        pass


class OpcUaPLC(SimPLC):
    """Real S7-1500 (or PLCSIM Advanced) through OPC UA. The PLC runs the LAD logic itself;
    the application writes operator buttons, the simulated sensor and AI results into DB_Vision
    and reads back the outputs and counters."""
    kind = 'opc'
    BOOL_IN = ['HmiStart', 'HmiStop', 'HmiEStop', 'PartSensor']
    READ = ['ImageRequest', 'ResultReady', 'GoodCount', 'BadCount', 'Motor', 'Pusher', 'LampGreen', 'LampRed', 'Alarm']

    def __init__(self, endpoint, ns, db):
        super().__init__()
        from asyncua.sync import Client
        from asyncua import ua
        self.ua = ua
        self.endpoint = endpoint
        self.client = Client(endpoint, timeout=3)
        try:
            self.client.connect()
        except Exception:
            self._stop_loop()
            raise
        node = lambda name: self.client.get_node(f'ns={ns};s="{db}"."{name}"')
        self.nodes = {n: node(n) for n in self.BOOL_IN + self.READ + ['Defect', 'DefectClass', 'Confidence']}
        try:
            for n in self.READ:                 # fail early if the DB layout does not match
                self.nodes[n].read_value()
        except Exception:
            self.close()
            raise
        self._estop = False

    def set_estop(self, pressed):
        super().set_estop(pressed); self._estop = pressed

    def _w(self, name, value, vtype):
        self.nodes[name].write_value(self.ua.DataValue(self.ua.Variant(value, vtype)))

    def write_result(self, defect, cls, conf):
        V = self.ua.VariantType
        self._w('Defect', bool(defect), V.Boolean)
        self._w('DefectClass', int(cls), V.Int16)
        self._w('Confidence', float(conf), V.Float)
        self._w('ResultReady', True, V.Boolean)
        self.DB.update(Defect=bool(defect), DefectClass=int(cls), Confidence=float(conf))

    def manual_reject(self): self.write_result(True, 1, 1.0)
    def manual_accept(self): self.write_result(False, 0, 1.0)

    def scan(self, dt, sensor):
        t0 = time.perf_counter(); now = time.time()
        V = self.ua.VariantType
        self._w('HmiStart', self._start, V.Boolean)
        self._w('HmiStop', self._stop, V.Boolean)
        self._w('HmiEStop', self._estop, V.Boolean)
        self._w('PartSensor', bool(sensor), V.Boolean)
        self._start = self._stop = False
        vals = {n: self.nodes[n].read_value() for n in self.READ}
        prev_req, prev_good, prev_bad = self.DB['ImageRequest'], self.DB['GoodCount'], self.DB['BadCount']
        self.DB.update(ImageRequest=bool(vals['ImageRequest']), ResultReady=bool(vals['ResultReady']),
                       GoodCount=int(vals['GoodCount']), BadCount=int(vals['BadCount']))
        self.Q.update(motor=bool(vals['Motor']), pusher=bool(vals['Pusher']), green=bool(vals['LampGreen']),
                      red=bool(vals['LampRed']), alarm=bool(vals['Alarm']))
        self.I.update(sensor=bool(sensor), estop_nc=not self._estop)
        self.run = self.Q['motor'] or self.DB['ImageRequest'] or self.Q['pusher']
        if self.DB['ResultReady']: self.t_ready = now
        if self.DB['ImageRequest'] and not prev_req: self.last_active[2] = now
        if self.DB['BadCount'] != prev_bad: self.last_active[3] = now
        if self.DB['GoodCount'] != prev_good: self.last_active[4] = now
        self.scans += 1
        self.cycle_ms = (time.perf_counter() - t0) * 1000

    def _stop_loop(self):
        try:
            self.client.tloop.stop()
        except Exception:
            pass

    def close(self):
        try:
            self.client.disconnect()
        except Exception:
            pass
        self._stop_loop()


# =====================================================================
# Statistics
# =====================================================================
def _auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)) if len(x) > 1 else None


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.records = []
        self.t0 = time.time()

    def add(self, res, decision_defect, truth, source):
        r = {'n': len(self.records) + 1, 'time': datetime.datetime.now().strftime('%H:%M:%S'), 't': time.time() - self.t0,
             'pred': res['name'], 'p_defect': res['p_defect'], 'confidence': res['confidence'],
             'decision': 'defect' if decision_defect else 'good', 'truth': truth or '',
             'latency_ms': res['latency_ms'], 'source': source, 'path': res.get('path', '')}
        self.records.append(r)
        return r

    def known(self):
        return [r for r in self.records if r['truth']]

    @staticmethod
    def _bin(tp, fp, fn, tn):
        rec = tp / (tp + fn) if tp + fn else None
        prec = tp / (tp + fp) if tp + fp else None
        f1 = 2 * prec * rec / (prec + rec) if rec is not None and prec is not None and prec + rec > 0 else None
        spec = tn / (tn + fp) if tn + fp else None
        fpr = fp / (fp + tn) if fp + tn else None
        n = tp + fp + fn + tn
        acc = (tp + tn) / n if n else None
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'recall': rec, 'precision': prec, 'f1': f1,
                'specificity': spec, 'fpr': fpr, 'bin_acc': acc}

    def binary_at(self, th, good='good'):
        """What-if: binary metrics if the threshold were `th` (from the stored P(defect))."""
        tp = fp = fn = tn = 0
        for r in self.known():
            pd, td = r['p_defect'] >= th, r['truth'] != good
            tp += pd and td; fp += pd and not td; fn += (not pd) and td; tn += (not pd) and not td
        return self._bin(tp, fp, fn, tn)

    def _scores(self, good='good'):
        k = self.known()
        y = np.array([r['truth'] != good for r in k], dtype=bool)
        s = np.array([r['p_defect'] for r in k], dtype=float)
        return y, s

    def roc(self, good='good'):
        y, s = self._scores(good)
        P, N = int(y.sum()), int((~y).sum())
        if P == 0 or N == 0:
            return None
        o = np.argsort(-s, kind='mergesort'); ys = y[o]
        tpr = np.concatenate([[0], np.cumsum(ys) / P]); fpr = np.concatenate([[0], np.cumsum(~ys) / N])
        return fpr, tpr, _auc(fpr, tpr)

    def pr(self, good='good'):
        y, s = self._scores(good)
        P = int(y.sum())
        if P == 0 or len(y) == 0:
            return None
        o = np.argsort(-s, kind='mergesort'); ys = y[o]
        ctp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
        prec = ctp / k; rec = ctp / P
        ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
        return np.concatenate([[0], rec]), np.concatenate([[1], prec]), ap

    def per_class(self, names):
        k = self.known(); out = []
        for c in names:
            tp = sum(r['pred'] == c and r['truth'] == c for r in k)
            fp = sum(r['pred'] == c and r['truth'] != c for r in k)
            fn = sum(r['pred'] != c and r['truth'] == c for r in k)
            sup = sum(r['truth'] == c for r in k)
            p = tp / (tp + fp) if tp + fp else None
            rc = tp / (tp + fn) if tp + fn else None
            f1 = 2 * p * rc / (p + rc) if p is not None and rc is not None and p + rc > 0 else None
            conf = [r['confidence'] for r in self.records if r['pred'] == c]
            out.append({'cls': c, 'precision': p, 'recall': rc, 'f1': f1, 'support': sup,
                        'n_pred': sum(r['pred'] == c for r in self.records),
                        'conf_mean': float(np.mean(conf)) if conf else None})
        return out

    def macro_f1(self, names):
        f = [c['f1'] for c in self.per_class(names) if c['support'] > 0 and c['f1'] is not None]
        return float(np.mean(f)) if f else None

    def latency(self):
        a = np.array([r['latency_ms'] for r in self.records[-1000:]], dtype=float)
        if not len(a):
            return None
        return {'mean': float(a.mean()), 'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
                'max': float(a.max()), 'min': float(a.min()), 'std': float(a.std()), 'n': int(len(a))}

    def rolling_rate(self, w=20):
        d = np.array([r['decision'] == 'defect' for r in self.records], dtype=float)
        if not len(d):
            return np.array([]), np.array([])
        c = np.cumsum(np.concatenate([[0], d]))
        idx = np.arange(1, len(d) + 1); lo = np.maximum(0, idx - w)
        return idx, (c[idx] - c[lo]) / (idx - lo) * 100

    def cumulative_rate(self):
        d = np.array([r['decision'] == 'defect' for r in self.records], dtype=float)
        if not len(d):
            return np.array([]), np.array([])
        idx = np.arange(1, len(d) + 1)
        return idx, np.cumsum(d) / idx * 100

    def throughput_series(self, bin_s=15):
        now = time.time() - self.t0
        nb = max(1, int(np.ceil(now / bin_s)))
        cnt = np.zeros(nb)
        for r in self.records:
            cnt[min(nb - 1, int(r['t'] // bin_s))] += 1
        return (np.arange(nb) + 0.5) * bin_s, cnt * 60.0 / bin_s

    def consecutive_defects(self):
        k = 0
        for r in reversed(self.records):
            if r['decision'] != 'defect':
                break
            k += 1
        return k

    def pareto(self, names, good='good'):
        cnt = {n: sum(r['decision'] == 'defect' and r['pred'] == n for r in self.records) for n in names if n != good}
        items = sorted(cnt.items(), key=lambda kv: -kv[1])
        tot = sum(cnt.values())
        cum, acc = [], 0
        for _, v in items:
            acc += v; cum.append(acc / tot * 100 if tot else 0)
        return items, cum

    def summary(self, names, good_name='good'):
        n = len(self.records)
        good = sum(r['decision'] == 'good' for r in self.records)
        bad = n - good
        known = self.known()
        acc = (sum(r['pred'] == r['truth'] for r in known) / len(known)) if known else None
        tp = sum(r['decision'] == 'defect' and r['truth'] != good_name for r in known)
        fp = sum(r['decision'] == 'defect' and r['truth'] == good_name for r in known)
        fn = sum(r['decision'] == 'good' and r['truth'] != good_name for r in known)
        tn = sum(r['decision'] == 'good' and r['truth'] == good_name for r in known)
        b = self._bin(tp, fp, fn, tn)
        lat = float(np.mean([r['latency_ms'] for r in self.records[-50:]])) if n else None
        now = time.time() - self.t0
        thr = sum(1 for r in self.records if now - r['t'] <= 60) * (60 / min(60, max(now, 1)))
        by_pred = {k: sum(r['pred'] == k for r in self.records) for k in names}
        cm = np.zeros((len(names), len(names)), dtype=int)
        for r in known:
            if r['truth'] in names and r['pred'] in names:
                cm[names.index(r['truth']), names.index(r['pred'])] += 1
        _, rr = self.rolling_rate(20)
        out = {'n': n, 'good': good, 'bad': bad, 'rate': (bad / n * 100) if n else 0.0, 'acc': acc, 'n_known': len(known),
               'latency': lat, 'throughput': thr, 'by_pred': by_pred, 'cm': cm,
               'rolling': float(rr[-1]) if len(rr) else None, 'consec': self.consecutive_defects()}
        out.update(b)
        return out

    def export(self, path):
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=['n', 'time', 'pred', 'p_defect', 'confidence', 'decision', 'truth', 'latency_ms', 'source', 'path'],
                               extrasaction='ignore', delimiter=';')
            w.writeheader()
            for r in self.records:
                rr = dict(r)
                for k in ('p_defect', 'confidence', 'latency_ms'):
                    rr[k] = f'{r[k]:.4f}'.replace('.', ',')
                w.writerow(rr)


# =====================================================================
# Line state tracking → availability / OEE / timeline
# =====================================================================
STATES = ('run', 'inspect', 'stop', 'estop', 'manual')


class StateTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = time.time()
        self.dur = {k: 0.0 for k in STATES}
        self.segments = []            # [state, t_start, t_end] (absolute time)
        self.started = False
        self.state = 'stop'

    def update(self, state, dt):
        now = time.time()
        self.state = state
        if self.started or state != 'stop':
            self.dur[state] += dt
        if self.segments and self.segments[-1][0] == state and now - self.segments[-1][2] < 1.0:
            self.segments[-1][2] = now
        else:
            self.segments.append([state, now - dt, now])
        if len(self.segments) > 4000:
            self.segments = self.segments[-3000:]

    def oee(self, n_auto, good_auto, ideal_per_s):
        d = self.dur
        prod = d['run'] + d['inspect']
        planned = prod + d['stop'] + d['estop']
        A = prod / planned if planned > 1e-6 else None
        P = min(1.0, n_auto / (prod * ideal_per_s)) if prod > 1 and ideal_per_s > 0 else None
        Q = good_auto / n_auto if n_auto else None
        O = A * P * Q if None not in (A, P, Q) else None
        return {'A': A, 'P': P, 'Q': Q, 'OEE': O, 'prod_s': prod, 'planned_s': planned}


# =====================================================================
# Controller — connects everything (one object shared by all windows)
# =====================================================================
class Controller(QObject):
    tick = pyqtSignal()
    result = pyqtSignal(dict)
    explained = pyqtSignal(dict)
    logged = pyqtSignal(str, str, str)       # time, type, message
    state_changed = pyqtSignal()             # mode / plc / language / theme changes
    alarms_changed = pyqtSignal()
    order_changed = pyqtSignal()
    request_inference = pyqtSignal(dict)
    request_explain = pyqtSignal(dict)
    request_batch = pyqtSignal(list, float)
    error = pyqtSignal(str)

    TICK_MS = 50
    TRACE_SIGNALS = ('sensor', 'ImageRequest', 'ResultReady', 'motor', 'pusher', 'green', 'red', 'alarm')

    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        tr.lang = self.settings['lang']
        self.model = ModelService()
        self.sim = ProcessSim()
        self.plc = SimPLC()
        self.stats = Stats()
        self.states = StateTracker()
        self.events = []
        self.mode = 'auto'
        self.last_result = None
        self.last_image = None
        self.pending = False
        self._prev_req = False
        self.trace = deque(maxlen=400)            # logic analyser: (t, {signal: bool})
        self.cycle_hist = deque(maxlen=1200)      # PLC scan time samples
        # alarms (ISA-18.2 style: active / unacknowledged)
        self.alarms = {}
        self.alarm_hist = []
        # production order
        self.order_active = False
        self.order_base = 0
        self.order_t0 = None
        # inference thread
        self.thread = QThread()
        self.worker = InferenceWorker(self.model)
        self.worker.moveToThread(self.thread)
        self.request_inference.connect(self.worker.run)
        self.request_explain.connect(self.worker.explain)
        self.worker.finished.connect(self.on_inference)
        self.worker.explained.connect(self.explained)
        self.thread.start()
        # batch thread
        self.bthread = QThread()
        self.bworker = BatchWorker(self.model)
        self.bworker.moveToThread(self.bthread)
        self.request_batch.connect(self.bworker.run)
        self.bthread.start()
        # cyclic timer = "PLC scan cycle + process simulation"
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._t_last = time.perf_counter()
        self.load_model(self.settings['model_dir'])
        self.timer.start(self.TICK_MS)

    # ---------- helpers ----------
    @property
    def threshold(self):
        th = self.settings.get('threshold')
        if th is None and self.model.ready:
            th = self.model.default_threshold
        return float(th if th is not None else 0.5)

    @property
    def names(self):
        return self.model.names if self.model.ready else ['good', 'hole', 'objects', 'oil spot', 'thread error']

    def ideal_rate(self):
        """Ideal number of items per second = belt speed / spacing (no inspection stops)."""
        return float(self.settings['speed']) / max(0.05, float(self.settings['spacing']))

    def log(self, typ, msg):
        t = datetime.datetime.now().strftime('%H:%M:%S')
        self.events.append((t, typ, msg))
        self.events = self.events[-3000:]
        self.logged.emit(t, typ, msg)

    def line_state(self):
        plc = self.plc
        if plc.Q['alarm']:
            return 'estop'
        if self.mode == 'manual':
            return 'manual'
        if plc.DB['ImageRequest'] or self.pending:
            return 'inspect'
        if plc.run:
            return 'run'
        return 'stop'

    def oee(self):
        auto = [r for r in self.stats.records if r['source'] == 'auto']
        return self.states.oee(len(auto), sum(r['decision'] == 'good' for r in auto), self.ideal_rate())

    def test_metrics(self):
        """Metrics of the model on the test set (from model_meta.json) — reference for drift monitoring."""
        if not self.model.ready:
            return {}
        m = self.model.meta
        tm, tb = m.get('test_metrics', {}) or {}, m.get('test_binary', {}) or {}
        return {'acc': meta_metric(tm, 'acc') or meta_metric(tb, 'acc'),
                'f1': meta_metric(tm, 'f1'), 'auc': meta_metric(tm, 'auc') or meta_metric(tb, 'auc'),
                'recall': meta_metric(tb, 'tpr') or meta_metric(tb, 'recall'),
                'fpr': meta_metric(tb, 'fpr'), 'precision': meta_metric(tb, 'prec'),
                'latency': _f(m.get('latency_ms'))}

    # ---------- model ----------
    def load_model(self, folder):
        try:
            self.model.load(folder)
            self.sim.set_pool(self.model.demo)
            self.log('INFO', tr.t('log_model', m=self.model.meta.get('model', '?'), s=self.model.size, t=self.threshold))
            return True
        except Exception as e:
            self.model.model = None
            self.log('ALARM', tr.t('log_model_err', e=e))
            return False

    def set_threshold(self, th):
        md = self.model.default_threshold if self.model.ready else None
        self.settings['threshold'] = None if md is not None and abs(th - md) < 1e-6 else float(th)
        save_settings(self.settings)
        self.log('INFO', tr.t('log_threshold', t=self.threshold))
        self.state_changed.emit()

    def set_lang(self, lang):
        tr.lang = lang; self.settings['lang'] = lang; save_settings(self.settings)
        self.log('INFO', tr.t('log_lang')); self.state_changed.emit()

    def set_theme(self, name):
        self.settings['theme'] = name; save_settings(self.settings); self.state_changed.emit()

    def set_mode(self, mode):
        self.mode = mode
        self.plc.manual = (mode == 'manual') and self.plc.kind == 'sim'
        if mode == 'manual':
            self.plc.DB['ImageRequest'] = False
        self.log('INFO', tr.t('log_mode', m=tr.t('mode_' + mode)))
        self.state_changed.emit()

    def use_opc(self, enable):
        old = self.plc
        if enable:
            try:
                new = OpcUaPLC(self.settings['opc_endpoint'], int(self.settings['opc_ns']), self.settings['opc_db'])
            except ImportError:
                self.log('ALARM', tr.t('log_opc_err', e=tr.t('s_opc_missing'))); self.error.emit(tr.t('s_opc_missing'))
                self.raise_event('opc', e=tr.t('s_opc_missing')); self.state_changed.emit(); return False
            except Exception as e:
                self.log('ALARM', tr.t('log_opc_err', e=e)); self.error.emit(tr.t('log_opc_err', e=e))
                self.raise_event('opc', e=str(e)[:80]); self.state_changed.emit(); return False
            self.plc = new
            self.log('PLC', tr.t('log_opc_ok', u=self.settings['opc_endpoint']))
        else:
            self.plc = SimPLC()
            self.plc.manual = self.mode == 'manual'
        old.close()
        self.state_changed.emit()
        return True

    # ---------- alarms ----------
    def _alarm(self, key, active, level='ALARM', **kw):
        a = self.alarms.get(key)
        now = datetime.datetime.now().strftime('%H:%M:%S')
        if active and (a is None or not a['active']):
            self.alarms[key] = {'key': key, 'kw': kw, 'time': now, 'active': True, 'acked': False, 'level': level}
            self.alarm_hist.append((now, key, kw, 'ON', level))
            self.log('ALARM' if level == 'ALARM' else 'INFO', tr.t('al_' + key, **kw))
            self.alarms_changed.emit()
        elif active and a is not None:
            a['kw'] = kw
        elif not active and a is not None and a['active']:
            a['active'] = False
            self.alarm_hist.append((now, key, a['kw'], 'OFF', a['level']))
            if a['acked']:
                del self.alarms[key]
            self.alarms_changed.emit()
        self.alarm_hist = self.alarm_hist[-1000:]

    def raise_event(self, key, level='ALARM', **kw):
        """One-shot alarm (comes and goes, stays in the list until acknowledged)."""
        self._alarm(key, True, level, **kw); self._alarm(key, False)

    def ack(self, keys=None):
        now = datetime.datetime.now().strftime('%H:%M:%S')
        for k in list(self.alarms):
            if keys is None or k in keys:
                a = self.alarms[k]
                if not a['acked']:
                    a['acked'] = True; self.alarm_hist.append((now, k, a['kw'], 'ACK', a['level']))
                if not a['active']:
                    del self.alarms[k]
        self.log('INFO', tr.t('log_ack')); self.alarms_changed.emit()

    def visible_alarms(self):
        return sorted(self.alarms.values(), key=lambda a: a['time'], reverse=True)

    def unacked(self):
        return [a for a in self.alarms.values() if not a['acked'] and a['level'] == 'ALARM']

    def _check_alarms(self):
        s = self.settings; st = self.stats
        n = len(st.records)
        _, rr = st.rolling_rate(20)
        rate = float(rr[-1]) if len(rr) else 0.0
        self._alarm('rate', n >= 10 and rate >= float(s['alarm_rate']), r=rate, lim=float(s['alarm_rate']))
        c = st.consecutive_defects()
        self._alarm('consec', c >= int(s['alarm_consec']), k=c)
        lat = st.records[-1]['latency_ms'] if n else 0
        self._alarm('latency', lat > float(s['alarm_latency']), ms=lat, lim=float(s['alarm_latency']))
        ref = self.test_metrics().get('recall')
        b = st.binary_at(self.threshold)
        live = b['recall']
        ok_n = (b['tp'] + b['fn']) >= 15
        self._alarm('drift', bool(ref and live is not None and ok_n and live < ref - float(s['drift_margin'])),
                    live=live or 0.0, ref=ref or 0.0)

    # ---------- production order ----------
    def order_start(self, target=None):
        if target is not None:
            self.settings['order_target'] = int(target); save_settings(self.settings)
        self.order_active = True
        self.order_base = sum(r['decision'] == 'good' for r in self.stats.records)
        self.order_bad_base = sum(r['decision'] == 'defect' for r in self.stats.records)
        self.order_t0 = time.time()
        self.log('INFO', tr.t('log_order_start', n=int(self.settings['order_target'])))
        self.order_changed.emit()

    def order_cancel(self):
        self.order_active = False; self.log('INFO', tr.t('log_order_cancel')); self.order_changed.emit()

    def order_progress(self):
        target = int(self.settings['order_target'])
        if not self.order_active and self.order_t0 is None:
            return 0, target, 0, None
        good = sum(r['decision'] == 'good' for r in self.stats.records) - self.order_base
        bad = sum(r['decision'] == 'defect' for r in self.stats.records) - getattr(self, 'order_bad_base', 0)
        good = max(0, good)
        eta = None
        if self.order_active and good > 0 and self.order_t0:
            el = time.time() - self.order_t0
            eta = max(0.0, (target - good) * el / good)
        return min(good, target), target, max(0, bad), eta

    def _check_order(self):
        if not self.order_active:
            return
        done, target, _, _ = self.order_progress()
        if done >= target:
            self.order_active = False
            if self.settings.get('order_autostop', True):
                self.plc.press_stop()
            self.log('INFO', tr.t('log_order_done', n=target))
            self.raise_event('order', level='INFO', n=target)
        self.order_changed.emit()

    # ---------- operator commands ----------
    def cmd_start(self):
        self.plc.press_start(); self.states.started = True; self.log('INFO', tr.t('log_start'))

    def cmd_stop(self):
        self.plc.press_stop(); self.log('INFO', tr.t('log_stop'))

    def cmd_estop(self, pressed):
        self.plc.set_estop(pressed)
        self.log('ALARM' if pressed else 'INFO', tr.t('log_estop' if pressed else 'log_estop_reset'))
        self.state_changed.emit()

    def set_jog(self, on):
        self.plc.jog = on

    def manual_inspect(self):
        if not self.model.ready:
            self.error.emit(tr.t('no_model')); return
        it = self.sim.item_at_camera()
        if it is None:
            self.log('INFO', tr.t('no_item')); return
        it.status = 'wait'
        self.pending = True
        self.request_inference.emit({'path': it.path, 'threshold': self.threshold, 'item': it.id, 'truth': it.truth, 'source': 'manual'})

    def manual_decide(self, defect):
        it = self.sim.item_at_camera()
        if it is None or it.result is None:
            self.log('INFO', tr.t('no_item')); return
        it.status = 'reject_pending' if defect else 'good'
        (self.plc.manual_reject if defect else self.plc.manual_accept)()
        self.stats.add(it.result, defect, it.truth, 'operator')
        self.log('AI', tr.t('log_manual_dec', n=it.id, dec=tr.t('dec_defect' if defect else 'dec_good')))
        self.result.emit(dict(it.result, item=it.id, truth=it.truth, source='operator', decision_defect=defect))
        self._check_alarms(); self._check_order()

    def inspect_file(self, path):
        if not self.model.ready:
            self.error.emit(tr.t('no_model')); return
        self.request_inference.emit({'path': path, 'threshold': self.threshold, 'item': None, 'truth': None, 'source': 'file'})

    def explain(self, path):
        if not self.model.ready:
            self.error.emit(tr.t('no_model')); return
        self.request_explain.emit({'path': path, 'threshold': self.threshold})

    def start_batch(self, jobs):
        if not self.model.ready:
            self.error.emit(tr.t('no_model')); return False
        self.log('INFO', tr.t('log_batch', n=len(jobs)))
        self.request_batch.emit(jobs, self.threshold)
        return True

    def cancel_batch(self):
        self.bworker.cancel = True

    def reset_stats(self):
        self.stats.reset(); self.sim.reset(); self.states.reset(); self.trace.clear(); self.cycle_hist.clear()
        self.order_active = False; self.order_t0 = None
        for k in ('rate', 'consec', 'latency', 'drift'):
            self._alarm(k, False)
        self.log('INFO', tr.t('log_reset'))
        self.state_changed.emit(); self.order_changed.emit()

    # ---------- cycle ----------
    def _tick(self):
        now = time.perf_counter(); dt = min(0.2, now - self._t_last); self._t_last = now
        sensor = self.sim.sensor()
        try:
            self.plc.scan(dt, sensor)
        except Exception as e:      # OPC UA connection lost
            self.log('ALARM', tr.t('log_opc_err', e=e)); self.error.emit(tr.t('log_opc_err', e=e))
            self.raise_event('opc', e=str(e)[:80])
            self.use_opc(False); return
        req = self.plc.DB['ImageRequest']
        if self.mode == 'auto' and req and not self._prev_req and not self.pending:
            self.log('PLC', tr.t('log_request'))
            it = self.sim.item_at_camera()
            if it is not None and self.model.ready:
                it.status = 'wait'; self.pending = True
                self.request_inference.emit({'path': it.path, 'threshold': self.threshold, 'item': it.id, 'truth': it.truth, 'source': 'auto'})
            else:
                self.plc.write_result(False, 0, 0.0)
        self._prev_req = req
        self.sim.step(dt, self.plc.Q['motor'], self.plc.Q['pusher'], float(self.settings['speed']), float(self.settings['spacing']))
        self.states.update(self.line_state(), dt)
        P = self.plc
        self.trace.append((time.time(), {'sensor': P.I['sensor'], 'ImageRequest': P.DB['ImageRequest'], 'ResultReady': P.DB['ResultReady'] or time.time() - P.t_ready < 0.25,
                                         'motor': P.Q['motor'], 'pusher': P.Q['pusher'], 'green': P.Q['green'],
                                         'red': P.Q['red'], 'alarm': P.Q['alarm']}))
        self.cycle_hist.append(P.cycle_ms)
        self._alarm('estop', P.Q['alarm'])
        self.tick.emit()

    @pyqtSlot(dict)
    def on_inference(self, res):
        if res.get('source') in ('auto', 'manual'):
            self.pending = False
        if 'error' in res:
            self.log('ALARM', res['error']); return
        self.last_result = res
        self.last_image = res['path']
        src = res['source']
        it = self.sim.get(res['item']) if res.get('item') else None
        if src == 'auto':
            if it is not None:
                it.result = res
                it.status = 'reject_pending' if res['defect'] else 'good'
            try:
                self.plc.write_result(res['defect'], res['cls'], res['confidence'])
            except Exception as e:
                self.log('ALARM', tr.t('log_opc_err', e=e)); self.raise_event('opc', e=str(e)[:80]); self.use_opc(False)
            rec = self.stats.add(res, res['defect'], res.get('truth'), 'auto')
            self.log('AI', tr.t('log_result', n=rec['n'], cls=tr.cls(res['name']), conf=res['confidence'], p=res['p_defect'],
                                dec=tr.t('dec_defect' if res['defect'] else 'dec_good')))
            self._check_alarms(); self._check_order()
        elif src == 'manual':
            if it is not None:
                it.result = res; it.status = 'manual_wait'
            self.log('AI', tr.t('log_result', n=res['item'], cls=tr.cls(res['name']), conf=res['confidence'], p=res['p_defect'],
                                dec=tr.t('dec_defect' if res['defect'] else 'dec_good')) + ' (?)')
        else:
            self.log('AI', tr.t('log_file', f=Path(res['path']).name, cls=tr.cls(res['name']), conf=res['confidence']))
        self.result.emit(res)

    def shutdown(self):
        self.timer.stop()
        self.bworker.cancel = True
        self.plc.close()
        self.thread.quit(); self.thread.wait(2000)
        self.bthread.quit(); self.bthread.wait(3000)
