"""
wtwin_trainer_callback.py
==========================
W-Twin интеграция с HuggingFace Trainer.

Два файла в един:
  1. WTwinCallback — drop-in callback за всеки Trainer
  2. Demo — реален small GPT-2 fine-tune на Shakespeare

Употреба:
    from wtwin_trainer_callback import WTwinCallback
    trainer = Trainer(
        model=model,
        callbacks=[WTwinCallback(preset='finetuning')]
    )

Presets:
    'pretraining'          — дълги runs от нулата
    'adaptive_pretraining' — cosine schedule, >50K стъпки
    'finetuning'           — SFT/LoRA (experimental)
    'custom'               — подаваш wtwin_config ръчно
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from transformers import TrainerCallback, TrainerState, TrainerControl

try:
    from wtwin import WTwinMonitor
    WTWIN_OK = True
except ImportError:
    print("❌ wtwin не е инсталиран.")
    print("   pip install git+https://github.com/Kretski/WTwin.git")
    WTWIN_OK = False

# ---------------------------------------------------------------------------
# Preset конфигурации (същите като wtwin_optimizer.py)
# ---------------------------------------------------------------------------

PRESETS = {
    'pretraining': dict(
        warmup_steps=2000, alpha=2.0, n_consec=7,
        calibration_frac=0.10, use_adaptive_T=False,
    ),
    'adaptive_pretraining': dict(
        warmup_steps=2000, alpha=2.0, n_consec=7,
        calibration_frac=0.10, use_adaptive_T=True,
    ),
    'finetuning': dict(
        warmup_steps=200, alpha=2.5, n_consec=7,
        calibration_frac=0.10, use_adaptive_T=False,
    ),
    'custom': {},
}


# ---------------------------------------------------------------------------
# WTwinCallback
# ---------------------------------------------------------------------------

class WTwinCallback(TrainerCallback):
    """
    W-Twin drop-in callback за HuggingFace Trainer.

    Следи train loss на всяка logging стъпка.
    Алармира при progressive trajectory drift.

    Параметри:
        preset       — 'pretraining' | 'adaptive_pretraining' |
                       'finetuning' | 'custom'
        wtwin_config — допълнителни параметри (override на preset)
        on_alert     — callback при алерт: fn(step, W, state)
                       По подразбиране: принтира предупреждение
        verbose      — принтира W trajectory на всяка logging стъпка
    """

    def __init__(
        self,
        preset='pretraining',
        wtwin_config=None,
        on_alert=None,
        verbose=True,
    ):
        if not WTWIN_OK:
            raise ImportError("wtwin не е инсталиран.")

        cfg = dict(PRESETS.get(preset, PRESETS['pretraining']))
        if wtwin_config:
            cfg.update(wtwin_config)

        self.monitor     = WTwinMonitor(**cfg)
        self.preset      = preset
        self.on_alert    = on_alert or self._default_alert
        self.verbose     = verbose
        self._step       = 0
        self._first_alert = None

    @staticmethod
    def _default_alert(step, W, state):
        print(f"\n  ⚠ W-Twin ALERT @ step {step}  W={W:.3f}")
        print(f"    Progressive training degradation detected.")
        print(f"    Consider: checkpoint rollback or early stopping.\n")

    def on_log(self, args, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        """Извиква се при всяко logging събитие."""
        if not logs:
            return

        loss = logs.get('loss') or logs.get('train_loss') if logs else None
        if loss is None:
            return

        self._step += 1
        wt_state = self.monitor.update(self._step, float(loss))

        if self.verbose and self._step % 10 == 0:
            fitted = self.monitor.baseline.is_fitted
            w_str  = f"W={wt_state.W:+.3f}" if fitted else "W=calibrating"
            print(f"  [W-Twin] step={state.global_step}  "
                  f"loss={loss:.4f}  {w_str}")

        if wt_state.alert and self._first_alert is None:
            self._first_alert = state.global_step
            self.on_alert(state.global_step, wt_state.W, wt_state)

    def on_train_end(self, args, state, control, **kwargs):
        """Резюме в края на тренировката."""
        first = self.monitor.first_alert_step()
        print()
        print("=" * 55)
        print("W-Twin Training Summary")
        print("=" * 55)
        print(f"  Preset:          {self.preset}")
        print(f"  Steps monitored: {self._step}")
        print(f"  Baseline fitted: {self.monitor.baseline.is_fitted}")
        print(f"  First alert:     {first if first else '— (чисто convergence)'}")
        if first:
            print(f"  ⚠ Progressive drift detected at step {first}")
            print(f"    Recommend: review checkpoint before step {first}")
        else:
            print(f"  ✅ Clean training trajectory throughout")
        print("=" * 55)

    def wtwin_summary(self):
        return {
            'preset':           self.preset,
            'steps_monitored':  self._step,
            'first_alert':      self.monitor.first_alert_step(),
            'baseline_fitted':  self.monitor.baseline.is_fitted,
        }


# ---------------------------------------------------------------------------
# Demo — GPT-2 small на Shakespeare subset
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        Trainer, TrainingArguments, DataCollatorForLanguageModeling
    )
    from datasets import Dataset

    print("=" * 55)
    print("W-Twin HuggingFace Trainer Demo")
    print("Model: GPT-2 small | Data: Shakespeare")
    print("=" * 55)
    print()

    # Данни
    TEXT = """To be, or not to be, that is the question:
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die to sleep,
No more and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to tis a consummation
Devoutly to be wished. To die to sleep.
For who would bear the whips and scorns of time,
The oppressors wrong, the proud mans contumely,
The pangs of despised love, the laws delay,
The insolence of office, and the spurns
That patient merit of the unworthy takes.""" * 30

    print("Зареждаме GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    tokens    = tokenizer(TEXT, return_tensors='pt', truncation=False)
    input_ids = tokens['input_ids'][0]

    # Правим dataset от chunks
    chunk_size = 128
    chunks     = [
        input_ids[i:i+chunk_size].tolist()
        for i in range(0, len(input_ids) - chunk_size, chunk_size)
    ]
    dataset = Dataset.from_dict({'input_ids': chunks})
    print(f"Dataset: {len(dataset)} chunks × {chunk_size} tokens")

    # Model
    print("Зареждаме GPT-2 small (~117M params)...")
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    # Data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # W-Twin callback с кратък warmup за demo
    wtwin_cb = WTwinCallback(
        preset='custom',
        wtwin_config=dict(
            warmup_steps = 2,
            alpha        = 2.0,
            n_consec     = 7,
            calibration_frac = 0.15,
        ),
        verbose=True,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir          = './wtwin_demo_output',
        num_train_epochs=8,
        per_device_train_batch_size = 4,
        logging_steps       = 1,
        save_steps          = 9999,
        learning_rate       = 2e-5,
        warmup_steps        = 5,
        report_to           = 'none',
        use_cpu=True,
        dataloader_num_workers = 0,
    )

    # Trainer
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = dataset,
        data_collator   = collator,
        callbacks       = [wtwin_cb],
    )

    print()
    print("Стартираме training с W-Twin мониторинг...")
    print("-" * 55)

    trainer.train()

    print()
    print("Summary:", wtwin_cb.wtwin_summary())
