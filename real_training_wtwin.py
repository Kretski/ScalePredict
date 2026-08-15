"""
Реален тренировъчен run — char-level language model върху Shakespeare.
Логва loss на всяка стъпка → dense log за W-Twin.
Два режима:
  - clean:      нормална тренировка (тест за false alarms)
  - degraded:   инжектираме LR spike след стъпка 600 (тест за detection)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from wtwin import WTwinMonitor

torch.manual_seed(42)
DEVICE = 'cpu'

# ---------------------------------------------------------------------------
# Данни — Shakespeare subset (генерирани за да не зависим от мрежата)
# ---------------------------------------------------------------------------

TEXT = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil
Must give us pause. There's the respect
That makes calamity of so long life.
""" * 80  # ~9KB реален текст

# Char-level tokenizer
chars = sorted(set(TEXT))
stoi  = {c: i for i, c in enumerate(chars)}
itos  = {i: c for i, c in enumerate(chars)}
VOCAB = len(chars)
data  = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)

def get_batch(seq_len=64, batch_size=16):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x  = torch.stack([data[i:i+seq_len]   for i in ix])
    y  = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

# ---------------------------------------------------------------------------
# Малък char-level Transformer (~200K params)
# ---------------------------------------------------------------------------

class SmallLM(nn.Module):
    def __init__(self, vocab, d=64, heads=4, layers=2, ctx=64):
        super().__init__()
        self.embed  = nn.Embedding(vocab, d)
        self.pos    = nn.Embedding(ctx, d)
        enc_layer   = nn.TransformerEncoderLayer(d, heads, d*4,
                                                  dropout=0.1,
                                                  batch_first=True,
                                                  norm_first=True)
        self.tf     = nn.TransformerEncoder(enc_layer, layers)
        self.head   = nn.Linear(d, vocab)
        self.ctx    = ctx

    def forward(self, x):
        B, T  = x.shape
        pos   = torch.arange(T, device=x.device)
        h     = self.embed(x) + self.pos(pos)
        mask  = nn.Transformer.generate_square_subsequent_mask(T)
        h     = self.tf(h, mask=mask, is_causal=True)
        return self.head(h)

# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------

def train(mode='clean', n_steps=1000, log_every=1):
    model = SmallLM(VOCAB).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    base_lr = 3e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                   weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()

    monitor = WTwinMonitor(
        warmup_steps=100,
        alpha=2.0,
        n_consec=5,
        calibration_frac=0.10,
    )

    log = []
    degrade_step = 600

    print(f"\n{'='*60}")
    print(f"Mode: {mode.upper()}  |  Steps: {n_steps}  |  Params: {n_params:,}")
    if mode == 'degraded':
        print(f"Degradation: LR x10 spike @ step {degrade_step} (simulates misconfiguration)")
    print(f"{'='*60}")

    t0 = time.time()
    model.train()

    for step in range(1, n_steps + 1):
        # Инжектираме деградация: LR spike + label noise
        if mode == 'degraded' and step == degrade_step:
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * 10   # LR spike
            print(f"  [step {step}] ⚡ LR spike injected")

        if mode == 'degraded' and step > degrade_step:
            # Постепенно нарастващ label noise — simulates data corruption
            noise_prob = min(0.4, (step - degrade_step) / 800)
        else:
            noise_prob = 0.0

        x, y = get_batch()

        # Label noise
        if noise_prob > 0:
            mask = torch.rand_like(y, dtype=torch.float) < noise_prob
            y[mask] = torch.randint(0, VOCAB, (mask.sum().item(),))

        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits.view(-1, VOCAB), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        state    = monitor.update(step, loss_val)

        if step % log_every == 0:
            log.append({
                'step': step,
                'loss': round(loss_val, 6),
                'W':    round(state.W, 4),
                'D':    round(state.D, 4),
                'alert': state.alert,
            })

        if state.alert and step == monitor.first_alert_step():
            delay = step - degrade_step if mode == 'degraded' else None
            print(f"  ⚠ W-Twin ALERT @ step {step}  W={state.W:.3f}"
                  + (f"  delay={delay} steps" if delay else ""))

        if step % 200 == 0:
            elapsed = time.time() - t0
            print(f"  step {step:4d} | loss={loss_val:.4f} | W={state.W:+.3f} | {elapsed:.1f}s")

    first_alert = monitor.first_alert_step()
    print(f"\nResult: first_alert={first_alert}"
          + (f"  (delay={first_alert - degrade_step} steps after injection)"
             if first_alert and mode == 'degraded' else ""))

    return log, first_alert, monitor

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    N_STEPS = 1000

    print("Реален char-level LM тренировъчен run")
    print(f"Vocab: {VOCAB} chars  |  Device: {DEVICE}")

    # Run 1: Clean
    log_clean, alert_clean, mon_clean = train('clean', N_STEPS)

    # Run 2: Degraded
    log_deg, alert_deg, mon_deg = train('degraded', N_STEPS)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ФИНАЛЕН РЕЗУЛТАТ — реални данни")
    print(f"{'='*60}")
    print(f"Clean run   : alert={alert_clean}  "
          + ("✅ без алерт (правилно)" if not alert_clean else "❌ false alarm"))
    print(f"Degraded run: alert={alert_deg}  "
          + (f"✅ хванато (delay={alert_deg-600} steps)" if alert_deg else "❌ пропуснато"))

    print(f"\nW-Twin calibration_frac=0.10, warmup=100, alpha=2.0, n_consec=5")
    print(f"Модел: char-LM, AdamW + CosineScheduler, ~реален run")

    # Запазване
    results = {
        'clean':    {'log': log_clean[:50], 'first_alert': alert_clean},
        'degraded': {'log': log_deg[:50],   'first_alert': alert_deg,
                     'inject_step': 600},
    }
    with open('/mnt/user-data/outputs/real_training_wtwin.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved → real_training_wtwin.json")
