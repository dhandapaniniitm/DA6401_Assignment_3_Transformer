# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

**Course No:** DA6401\
**Course Name:** Introduction to DeepLearning\
**Author:** Dhandapani N\
**Roll No:** BT25S018\
**Email Id:** bt25s018@smail.iitm.ac.in

---

## Links

**Link to WandB Report:**
[WandB Experiment Report](https://wandb.ai/bt25s018-iitm/DA6401_Assignment3/reports/DA6401-Assignment_3_Implementing-a-Transformer-for-Machine-Translation--VmlldzoxNjkyMjA5Nw?accessToken=29jtimxmclx0dtorkreew71s9ib2gchksr10yxn752tvlab048g5vd7yy0utog6a)

**Link to GitHub Repository:**
[Project GitHub Repository](https://github.com/dhandapaniniitm/DA6401_Assignment_3_Transformer.git)

---

##  Overview

This project implements the Transformer architecture (“Attention Is All You Need”) from scratch using PyTorch.

The system performs Neural Machine Translation (NMT) from German → English using the Multi30k dataset.

The implementation includes:

* Encoder–Decoder Transformer
* Multi-Head Attention
* Label Smoothing Loss
* Noam Learning Rate Scheduler
* Greedy Decoding
* BLEU Evaluation
* Weights & Biases (W&B) experiment tracking

---

##  Project Structure

```
.
├── dataset.py
├── lr_scheduler.py
├── model.py
├── README.md
├── requirements.txt
└── train.py
```

---

##  Training Pipeline (`train.py`)

### Core Features

**1. Training & Evaluation Loop**

* Forward pass with masks
* Label smoothing loss
* Backpropagation + gradient clipping
* Learning rate scheduling (Noam or fixed)

**2. Prediction Confidence**

* Logs softmax probability of correct token
* Helps analyze calibration and training stability
* Batch-wise metric (naturally noisy)

**3. Greedy Decoding**

* Autoregressive generation
* Starts from `<sos>`
* Stops at `<eos>` or max length

**4. BLEU Evaluation**

* Corpus-level BLEU score (0–100)
* Ignores special tokens
* Uses greedy decoding outputs

**5. Checkpointing**
Saves:

* Model weights
* Optimizer state
* Scheduler state
* Vocabulary
* Epoch number

---

##  Experiment Tracking

Logged via:

* Weights & Biases (W&B)
* CSV logs (`logs/<experiment_name>.csv`)

Tracked metrics:

* Train loss
* Validation loss
* BLEU score
* Learning rate
* Prediction confidence
* Test BLEU

---

##  Autograder Requirements

Ensure the following:

* `greedy_decode()` matches required signature
* `evaluate_bleu()` returns corpus BLEU (0–100)
* `Transformer.__init__()` loads:

  * Tokenizer
  * Vocabulary
  * Model weights

Must implement:

```python
model.infer(german_sentence)
```

Expected usage:

```python
model = Transformer().to(device)
model.eval()
model.infer(sentence)
```

---

##  Training Configuration

| Parameter       | Description             | Example    |
| --------------- | ----------------------- | ---------- |
| epochs          | Number of epochs        | 200        |
| batch_size      | Batch size              | 32         |
| device          | Compute device          | cuda       |
| scheduler_type  | LR scheduler            | noam       |
| lr              | Learning rate           | 1.0        |
| warmup_steps    | Noam warmup steps       | 4000       |
| use_scaling     | Attention scaling       | 1          |
| pos_encoding    | Positional encoding     | sinusoidal |
| label_smoothing | Smoothing factor        | 0.2        |
| dropout         | Dropout rate            | 0.3        |
| d_model         | Model dimension         | 256        |
| d_ff            | Feedforward dimension   | 2048       |
| N               | Transformer layers      | 3          |
| num_heads       | Attention heads         | 8          |
| num_workers     | DataLoader workers      | 10         |
| patience        | Early stopping patience | 20         |
| wandb_api_key   | W&B API key             | API-KEY    |
| wandb_task_name | Run name                | Train_v2_4 |

---

##  Sample Training Command

```bash
python train.py \
--epochs 200 \
--batch_size 32 \
--device cuda \
--scheduler_type noam \
--lr 1.0 \
--warmup_steps 4000 \
--use_scaling 1 \
--pos_encoding sinusoidal \
--label_smoothing 0.2 \
--dropout 0.3 \
--d_model 256 \
--d_ff 2048 \
--N 3 \
--num_heads 8 \
--num_workers 10 \
--patience 20 \
--wandb_api_key "API-KEY" \
--wandb_task_name "Train_v2_4"
```

---

##  Summary

This project implements a full Transformer-based NMT system from scratch with:

* End-to-end training pipeline
* BLEU evaluation
* Greedy decoding inference
* W&B experiment tracking
* Robust checkpointing

Fully compatible with autograder requirements.

---

## Contact

For questions or issues, please contact the email id given above.
