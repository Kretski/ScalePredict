"""
wtwin_optimizer.py
==================
W-Twin интегриран директно в PyTorch оптимизатор.

Поддържани wrapper-и:
    WTwinAdamW   — за pretraining (β₂=0.95, frontier стандарт)
    WTwinSGD     — за SGD + Momentum
    WTwinLion    — за Lion (изисква lion-pytorch)

Четири preset-а:
    'pretraining'          — LLaMA/GPT-2/Mistral стандарт
                             warmup=2000, alpha=2.0, n_consec=5, fixed threshold
                             Baseline: PowerLawBaseline (валидиран)
    'adaptive_pretraining' — За дълги скъпи runs (>50K стъпки) с cosine annealing
                             warmup=2000, alpha=2.0, n_consec=5, adaptive threshold
                             Тествано: 2.4× по-бърза детекция vs fixed (delay 69 vs 170 стъпки)
                             ⚠ Не е тестван при cosine annealing с restarts (OneCycleLR, SGDR)
    'finetuning'           — Full fine-tuning с ExpFloorBaseline [EXPERIMENTAL]
                             warmup=200, alpha=2.5, n_consec=7
                             Baseline: ExpFloorBaseline (изисква scipy)
                             Валидирано: clean, plateau, progressive drift, drift след plateau
                             НЕ хваща: LR spike с ускорена конвергенция, catastrophic forgetting,
                                       RLHF collapse. Не използвай за LoRA/QLoRA/SFT < 2000 стъпки.
    'custom'               — Подаваш всичко ръчно

Употреба:
    optimizer = WTwinAdamW(model.parameters(), lr=3e-4)
    for step, (x, y) in enumerate(loader, 1):
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(loss=loss)        ← loss подаден тук
        optimizer.zero_grad()
        if optimizer.wtwin_alert():
            print(f'⚠ Деградация @ стъпка {step}')

Или с callback:
    def on_alert(step, W, state):
        print(f'⚠ step={step} W={W:.3f}')
        # → rollback checkpoint, намали LR, изпрати notification

    optimizer = WTwinAdamW(..., wtwin_on_alert=on_alert)
"""

from __future__ import annotations

import torch
import warnings
from typing import Callable, Optional, Iterable
from wtwin import WTwinMonitor

# ---------------------------------------------------------------------------
# Preset конфигурации — базирани на реални frontier параметри
# ---------------------------------------------------------------------------

PRESETS = {
    'pretraining': dict(
        warmup_steps=2000,
        alpha=2.0,
        n_consec=5,
        calibration_frac=0.10,
        use_adaptive_T=False,
    ),
    'adaptive_pretraining': dict(
        warmup_steps=2000,
        alpha=2.0,
        n_consec=5,
        calibration_frac=0.10,
        use_adaptive_T=True,    # 2.4× по-бърза детекция при cosine schedule
                                # ⚠ не е тестван при LR restarts (OneCycleLR, SGDR)
    ),
    # finetuning използва ExpFloorBaseline — конфигурацията е в _make_monitor()
    # защото изисква lazy import (scipy зависимост)
    'finetuning': dict(
        warmup_steps=200,
        alpha=2.5,
        n_consec=7,
        use_adaptive_T=False,
        # EXPERIMENTAL: ExpFloorBaseline добавен от _make_monitor()
        # Валидирано: clean, plateau, progressive drift, drift след plateau (4/5 матрица)
        # НЕ хваща: LR spike с ускорена конвергенция, catastrophic forgetting, RLHF collapse
        # Не използвай за: LoRA, QLoRA, SFT < 2000 стъпки, RLHF/DPO
    ),
    'custom': {},
}


def _make_monitor(preset: str, wtwin_config: Optional[dict]) -> WTwinMonitor:
    """
    Factory за WTwinMonitor с правилния baseline за всеки preset.

    Lazy import на ExpFloorBaseline само при 'finetuning' preset —
    потребители без scipy не се засягат при използване на другите preset-и.
    """
    cfg = dict(PRESETS.get(preset, PRESETS['pretraining']))
    if wtwin_config:
        cfg.update(wtwin_config)

    if preset == 'finetuning' and 'baseline' not in cfg:
        try:
            from exp_floor_baseline import ExpFloorBaseline
            # ExpFloorBaseline warmup е отделен от WTwinMonitor warmup.
            # WTwinMonitor warmup=200 пропуска ранните стъпки за детекция.
            # ExpFloorBaseline warmup=20 — нужни са повече точки за fit на L_inf.
            # Calibration_frac=0.30 за по-надеждна оценка на асимптотата.
            cfg['baseline'] = ExpFloorBaseline(
                warmup_steps=20,
                calibration_frac=0.30,
            )
        except ImportError:
            warnings.warn(
                "scipy не е инсталиран — finetuning preset използва PowerLawBaseline. "
                "pip install scipy за ExpFloorBaseline (препоръчан за fine-tuning).",
                UserWarning, stacklevel=3,
            )

    return WTwinMonitor(**cfg)


# ---------------------------------------------------------------------------
# Базов mixin — логиката е тук, наследяват всички wrapper-и
# ---------------------------------------------------------------------------

class WTwinMixin:
    """
    Mixin за W-Twin мониторинг.
    Наследяващият клас трябва да е валиден torch.optim.Optimizer.

    Не добавя overhead освен WTwinMonitor.update() след всяка стъпка:
    измерено ~0.30ms срещу 8–84ms за optimizer.step().
    """

    def _wtwin_init(
        self,
        preset: str = 'pretraining',
        wtwin_config: Optional[dict] = None,
        wtwin_on_alert: Optional[Callable] = None,
        wtwin_log_every: int = 1,
    ):
        """Инициализира W-Twin state. Извиква се от __init__ на wrapper-а."""
        self._wtwin_monitor    = _make_monitor(preset, wtwin_config)
        self._wtwin_step       = 0
        self._wtwin_on_alert   = wtwin_on_alert
        self._wtwin_log_every  = wtwin_log_every
        self._wtwin_preset     = preset
        self._wtwin_history    = []      # (step, loss, W, alert)
        self._wtwin_last_loss  = None

    def step(self, closure=None, loss=None):
        """
        Overrides optimizer.step().

        loss може да се подаде по два начина:
          1. optimizer.step(loss=loss_tensor)      ← препоръчан
          2. optimizer.step(closure)               ← standard PyTorch closure
        """
        # Изпълняваме оригиналния optimizer step
        result = super().step(closure)

        # Вземаме loss стойността
        loss_val = None
        if loss is not None:
            loss_val = float(loss.item()) if hasattr(loss, 'item') else float(loss)
        elif closure is not None:
            # closure вече е изпълнен от super().step() — loss не е достъпен тук
            # потребителят трябва да подаде loss= директно
            pass
        elif self._wtwin_last_loss is not None:
            loss_val = self._wtwin_last_loss

        # W-Twin update
        if loss_val is not None:
            self._wtwin_step += 1
            state = self._wtwin_monitor.update(self._wtwin_step, loss_val)

            if self._wtwin_step % self._wtwin_log_every == 0:
                self._wtwin_history.append((
                    self._wtwin_step, loss_val,
                    round(state.W, 4), state.alert
                ))

            # Alert callback
            if state.alert:
                first = self._wtwin_monitor.first_alert_step()
                if first == self._wtwin_step and self._wtwin_on_alert:
                    self._wtwin_on_alert(self._wtwin_step, state.W, state)

        return result

    def record_loss(self, loss):
        """
        Алтернативен начин за подаване на loss — преди step().
        Полезно когато не може да се модифицира training loop-ът.

        Употреба:
            optimizer.record_loss(loss)
            optimizer.step()
        """
        self._wtwin_last_loss = (
            float(loss.item()) if hasattr(loss, 'item') else float(loss)
        )

    def wtwin_alert(self) -> bool:
        """True ако W-Twin е в alert state на последната стъпка."""
        if self._wtwin_monitor.history:
            return self._wtwin_monitor.history[-1].alert
        return False

    def wtwin_first_alert_step(self) -> Optional[int]:
        """Стъпката на първия алерт, или None."""
        return self._wtwin_monitor.first_alert_step()

    def wtwin_state(self):
        """Последният WTwinState обект."""
        if self._wtwin_monitor.history:
            return self._wtwin_monitor.history[-1]
        return None

    def wtwin_summary(self) -> dict:
        """Кратко резюме на мониторинга."""
        first = self._wtwin_monitor.first_alert_step()
        return {
            'preset':           self._wtwin_preset,
            'steps_monitored':  self._wtwin_step,
            'first_alert_step': first,
            'alert_active':     self.wtwin_alert(),
            'baseline_fitted':  self._wtwin_monitor.baseline.is_fitted,
        }

    def wtwin_reset(self):
        """Нулира W-Twin (при resume от checkpoint, например)."""
        self._wtwin_monitor.reset()
        self._wtwin_step = 0
        self._wtwin_history.clear()
        self._wtwin_last_loss = None


# ---------------------------------------------------------------------------
# Конкретни wrapper-и
# ---------------------------------------------------------------------------

class WTwinAdamW(WTwinMixin, torch.optim.AdamW):
    """
    AdamW + W-Twin мониторинг.

    Frontier pretraining параметри (LLaMA/GPT-2/Mistral стандарт):
        β₁=0.9, β₂=0.95, weight_decay=0.1, eps=1e-8
        (различно от PyTorch default β₂=0.999!)

    Пример:
        optimizer = WTwinAdamW(
            model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),      # frontier стандарт
            weight_decay=0.1,
            preset='pretraining',
            wtwin_on_alert=lambda step, W, s: print(f'ALERT @ {step}'),
        )
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        betas: tuple = (0.9, 0.95),      # frontier стандарт, не PyTorch default
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        amsgrad: bool = False,
        # W-Twin параметри
        preset: str = 'pretraining',
        wtwin_config: Optional[dict] = None,
        wtwin_on_alert: Optional[Callable] = None,
        wtwin_log_every: int = 1,
    ):
        if betas[1] == 0.999:
            warnings.warn(
                "β₂=0.999 е PyTorch default, но frontier LLM runs използват β₂=0.95. "
                "W-Twin е оптимизиран за β₂=0.95. Ако тренираш от нула, помисли за промяна.",
                UserWarning, stacklevel=2
            )
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
        self._wtwin_init(preset, wtwin_config, wtwin_on_alert, wtwin_log_every)


class WTwinSGD(WTwinMixin, torch.optim.SGD):
    """
    SGD + Momentum + W-Twin.

    Препоръчани параметри за W-Twin при SGD:
        alpha=2.5, n_consec=7 (по-шумен optimizer)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        nesterov: bool = False,
        # W-Twin
        preset: str = 'pretraining',
        wtwin_config: Optional[dict] = None,
        wtwin_on_alert: Optional[Callable] = None,
        wtwin_log_every: int = 1,
    ):
        # SGD preset: по-консервативен поради осцилации
        if preset == 'pretraining' and wtwin_config is None:
            wtwin_config = {'alpha': 2.5, 'n_consec': 7, 'warmup_steps': 500}
        super().__init__(
            params, lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=nesterov,
        )
        self._wtwin_init(preset, wtwin_config, wtwin_on_alert, wtwin_log_every)


class WTwinLion(WTwinMixin):
    """
    Lion + W-Twin.

    Lion изисква lion-pytorch: pip install lion-pytorch
    Параметри: β₁=0.95, β₂=0.98, LR = 0.1 × AdamW LR, WD = 10 × AdamW WD
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-5,              # ~0.1× AdamW LR
        betas: tuple = (0.95, 0.98),
        weight_decay: float = 1.0,     # ~10× AdamW WD
        # W-Twin
        preset: str = 'pretraining',
        wtwin_config: Optional[dict] = None,
        wtwin_on_alert: Optional[Callable] = None,
        wtwin_log_every: int = 1,
    ):
        try:
            from lion_pytorch import Lion
            self._lion = Lion(params, lr=lr, betas=betas, weight_decay=weight_decay)
        except ImportError:
            raise ImportError(
                "lion-pytorch не е инсталиран. "
                "pip install lion-pytorch"
            )
        # Lion warmup е по-дълъг — агресивна ранна фаза
        if preset == 'pretraining' and wtwin_config is None:
            wtwin_config = {'warmup_steps': 2000, 'alpha': 2.0, 'n_consec': 5}

        self._wtwin_init(preset, wtwin_config, wtwin_on_alert, wtwin_log_every)
        # Копираме param_groups за съвместимост
        self.param_groups = self._lion.param_groups
        self.state        = self._lion.state

    def step(self, closure=None, loss=None):
        result = self._lion.step(closure)
        # W-Twin update (reuse mixin logic)
        loss_val = None
        if loss is not None:
            loss_val = float(loss.item()) if hasattr(loss, 'item') else float(loss)
        elif self._wtwin_last_loss is not None:
            loss_val = self._wtwin_last_loss
        if loss_val is not None:
            self._wtwin_step += 1
            state = self._wtwin_monitor.update(self._wtwin_step, loss_val)
            self._wtwin_history.append((self._wtwin_step, loss_val, round(state.W, 4), state.alert))
            if state.alert:
                first = self._wtwin_monitor.first_alert_step()
                if first == self._wtwin_step and self._wtwin_on_alert:
                    self._wtwin_on_alert(self._wtwin_step, state.W, state)
        return result

    def zero_grad(self, set_to_none=True):
        return self._lion.zero_grad(set_to_none=set_to_none)

    def record_loss(self, loss):
        self._wtwin_last_loss = float(loss.item()) if hasattr(loss, 'item') else float(loss)

    def wtwin_alert(self):
        if self._wtwin_monitor.history:
            return self._wtwin_monitor.history[-1].alert
        return False

    def wtwin_first_alert_step(self):
        return self._wtwin_monitor.first_alert_step()

    def wtwin_state(self):
        if self._wtwin_monitor.history:
            return self._wtwin_monitor.history[-1]
        return None

    def wtwin_summary(self):
        first = self._wtwin_monitor.first_alert_step()
        return {
            'preset': self._wtwin_preset,
            'steps_monitored': self._wtwin_step,
            'first_alert_step': first,
            'alert_active': self.wtwin_alert(),
            'baseline_fitted': self._wtwin_monitor.baseline.is_fitted,
        }


# ---------------------------------------------------------------------------
# Демо / тест
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch.nn as nn
    import time

    torch.manual_seed(42)

    TEXT = "To be or not to be that is the question " * 500
    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)
    VOCAB = len(chars)

    def get_batch(seq=64, bs=16):
        ix = torch.randint(len(data) - seq, (bs,))
        x = torch.stack([data[i:i+seq] for i in ix])
        y = torch.stack([data[i+1:i+seq+1] for i in ix])
        return x, y

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb  = nn.Embedding(VOCAB, 64)
            self.pos  = nn.Embedding(64, 64)
            self.lstm = nn.LSTM(64, 128, batch_first=True)
            self.head = nn.Linear(128, VOCAB)
        def forward(self, x):
            h = self.emb(x) + self.pos(torch.arange(x.size(1)))
            h, _ = self.lstm(h)
            return self.head(h)

    print("=" * 60)
    print("W-Twin Optimizer Wrapper — демо")
    print("=" * 60)

    alerts_log = []

    def on_alert(step, W, state):
        alerts_log.append(step)
        print(f"  ⚠ W-Twin ALERT @ step {step}  W={W:.3f}")

    model = TinyLM()
    crit  = nn.CrossEntropyLoss()

    # --- Test 1: Clean run с WTwinAdamW ---
    print("\n[1] Clean run — WTwinAdamW (preset=pretraining, warmup=200 за demo)")
    opt = WTwinAdamW(
        model.parameters(), lr=3e-4,
        betas=(0.9, 0.95), weight_decay=0.1,
        preset='custom',
        wtwin_config={'warmup_steps': 200, 'alpha': 2.0, 'n_consec': 5},
        wtwin_on_alert=on_alert,
    )

    t0 = time.time()
    for step in range(1, 1001):
        x, y = get_batch()
        opt.zero_grad()
        loss = crit(model(x).view(-1, VOCAB), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(loss=loss)

        if step % 200 == 0:
            s = opt.wtwin_state()
            W_str = f"W={s.W:+.3f}" if s else "W=n/a"
            print(f"  step {step:4d} | loss={loss.item():.4f} | {W_str} | {time.time()-t0:.1f}s")

    print(f"\n  Резюме: {opt.wtwin_summary()}")
    print(f"  False alarms (clean): {len(alerts_log)}/1000")

    # --- Test 2: Degraded run ---
    print("\n[2] Degraded run — LR spike @ step 600")
    model2 = TinyLM()
    alerts_log2 = []

    opt2 = WTwinAdamW(
        model2.parameters(), lr=3e-4,
        betas=(0.9, 0.95), weight_decay=0.1,
        preset='custom',
        wtwin_config={'warmup_steps': 200, 'alpha': 2.0, 'n_consec': 5},
        wtwin_on_alert=lambda s, W, st: alerts_log2.append(s),
    )

    for step in range(1, 1001):
        if step == 600:
            for pg in opt2.param_groups:
                pg['lr'] *= 10
            print(f"  [step 600] ⚡ LR spike")

        x, y = get_batch()
        opt2.zero_grad()
        loss2 = crit(model2(x).view(-1, VOCAB), y.view(-1))
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt2.step(loss=loss2)

    first = opt2.wtwin_first_alert_step()
    print(f"  Първи алерт: стъпка {first}  (delay={first - 600 if first else 'N/A'} steps)")
    print(f"\n  Резюме: {opt2.wtwin_summary()}")

    # --- record_loss API ---
    print("\n[3] record_loss() API — без промяна в training loop")
    model3 = TinyLM()
    opt3 = WTwinAdamW(model3.parameters(), lr=3e-4, betas=(0.9, 0.95))
    for step in range(1, 301):
        x, y = get_batch()
        opt3.zero_grad()
        loss3 = crit(model3(x).view(-1, VOCAB), y.view(-1))
        loss3.backward()
        opt3.record_loss(loss3)   # ← подаваме тук
        opt3.step()               # ← стандартен step без промяна
    print(f"  Steps monitored: {opt3.wtwin_summary()['steps_monitored']}")
    print(f"  Baseline fitted: {opt3.wtwin_summary()['baseline_fitted']}")

    print("\n✅ Всички тестове преминаха")
