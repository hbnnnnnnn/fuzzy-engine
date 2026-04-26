# RAG Patch — Revision Notes

Review of [`src/rag_patch_training/`](src/rag_patch_training/) and its interaction with [`src/Nexus-Gen/`](src/Nexus-Gen/). Scope: verify the approach, diagnose the reported "gate too low, training collapsed" observation, and propose concrete improvements.

---

## 1. Is the approach legit, correct, and workable?

### What the code actually does

Two-stream flow-matching training on top of a frozen Nexus-GenV2 FLUX DiT:

| Stream | Conditioning | Weights | Output |
|---|---|---|---|
| 1 (frozen) | full prompt (CLIP + T5, 512 tokens) | LoRA **disabled** | $e_0$ |
| 2 (trainable) | 1-token style context from VGG-19 stats | LoRA **enabled** | $S_\varphi$ |

Fusion and loss:

$$
v_\text{pred} \;=\; e_0 \;+\; g \cdot S_\varphi, \qquad
\mathcal{L} \;=\; \mathbb{E}\big[w(t)\cdot \|v_\text{pred} - v^*\|_2^2\big], \qquad
v^* \;=\; \varepsilon - z_0
$$

with gate $g$ zero-initialised (ControlNet-style), and BSMNTW timestep weights $w(t)$.

### What's correct

- Flow-matching target $v^* = \varepsilon - z_0$ is the right target for FLUX (not $\varepsilon$-prediction). See [model.py:558-560](src/rag_patch_training/model.py#L558-L560).
- Zero-init gate gives exact baseline parity at $t=0$: $v_\text{pred}\big|_{g=0} = e_0$.
- In-place LoRA with `disable_adapter()` for Stream 1 is a valid memory-saving trick; the two streams share weights and differ only by the LoRA delta.
- Mean + std pooling of VGG-$\{1,2,3,4\}_1$ features is a standard cheap style descriptor (AdaIN-style), dimension $(64+128+256+512)\times 2 = 1920$.

### The structural problem that makes this uphill

Expand the loss around Stream 1:

$$
\mathcal{L}
= \|\,e_0 - v^*\,\|^2
\;+\; 2g\,\langle S_\varphi,\; e_0 - v^*\rangle
\;+\; g^2 \|S_\varphi\|^2
$$

The gate gradient is

$$
\frac{\partial \mathcal{L}}{\partial g}
\;=\; 2\,\langle S_\varphi,\; e_0 - v^*\rangle \;+\; 2g\,\|S_\varphi\|^2.
$$

Now observe:

- Stream 1 already receives the **ground-truth conditioning** (the prompt that describes the target image), so $e_0 \approx v^*$ in expectation — the base is essentially unbiased for the supervision signal.
- Stream 2 receives **strictly less information** (prompt dropped, replaced by 1 style token).
- Therefore $\mathbb{E}[\langle S_\varphi, e_0 - v^*\rangle] \approx 0$; the linear term is noise.
- Only the quadratic term $g^2\|S_\varphi\|^2$ has a consistent sign, and it is **minimized at $g = 0$**.

**Conclusion.** The objective itself pushes $g\to 0$. There is no "slot" in this loss for $S_\varphi$ to be helpful, because the base stream already has the richer conditioning. This is not a bug; it is the Bayes-optimal answer given the training signal.

Mathematical claim, loosely: if Stream 1 is $\sigma(\text{Bayes})$-close to optimal under supervision $v^*$, and Stream 2 conditions on a strict subset of that information, then any $g \neq 0$ adds variance without reducing bias → the optimum is $g=0$.

### Workable? Yes — but not with this objective.

The architecture (frozen base + additive LoRA residual + retrieval-conditioned features) is standard and sound. The *training recipe* needs to be reframed so that Stream 2 has information the base lacks, or a target the base cannot hit alone. See §3.

---

## 2. Is the "gate collapsed" diagnosis right? Any bugs?

### Diagnosis: correct symptom, wrong (or incomplete) cause

The reported $g = 0.017$ after [`epoch=4-step=625`](experiments/workdirs/rag_patch/lightning_logs/version_64043) is entirely consistent with the equilibrium analysis above. "Training collapsed" suggests a pathology; this is closer to **the optimizer finding the correct minimum of a poorly-posed objective**. Since $\|S_\varphi\|$ is on the order of $\|e_0\|$, a gate of $1.7\%$ means $v_\text{pred}$ differs from $e_0$ by $\sim\!1.7\%$ — well below visible-change threshold in the rendered images. **Your observation that baseline and patched images look identical is exactly what this value predicts.**

Secondary contributors:

- **Only 625 optimizer steps** at $\eta = 10^{-5}$ (5 epochs × 125 steps with batch=1, accum=8, 2 GPUs). Too short for LoRA + 2 MLP projectors to learn even if the objective were well-posed.
- **Out-of-distribution context length for the DiT**: Stream 2 calls the DiT with a **1-token** context at [model.py:541-549](src/rag_patch_training/model.py#L541-L549), while FLUX was trained with $\sim 512$ T5 tokens. The model's behavior on 1 token is unexplored territory; LoRA has to first stabilize it before $S_\varphi$ becomes a useful direction at all.

### Code bugs / concerns I spotted

1. **Inference path not verified.** [`scripts/eval/test_rag_patch_controlled.sh`](scripts/eval/test_rag_patch_controlled.sh) only compares **training loss** with random $t$ — it never generates images. The `rag_patch_results_controlled/` directory contains ground truth and references but no generated outputs. Confirm how the patch is actually plugged into [`modeling/decoder/pipelines.py`](src/Nexus-Gen/modeling/decoder/pipelines.py) at generation time. If the inference pipeline does not compute $e_0 + g\cdot S_\varphi$ at **every** denoising step, the patch is a no-op.

2. **Redundant argmin for sigma lookup.** [model.py:410-412](src/rag_patch_training/model.py#L410-L412):
   ```python
   diffs = (sched.timesteps.unsqueeze(0) - t.cpu().unsqueeze(1)).abs()
   tids = diffs.argmin(dim=1)
   ```
   `timestep_ids` is already the index used to look up `t`. Use `sched.sigmas[timestep_ids]` directly. Not a correctness bug, just wasted work and a potential float-precision hazard.

3. **`style_text_ids` is all zeros.** [model.py:530-532](src/rag_patch_training/model.py#L530-L532). Fine in principle (zero positions), but it means the style token has no positional identity — with multi-token style this would need to be rethought.

4. **Stream 2 drops the prompt entirely.** [model.py:541-549](src/rag_patch_training/model.py#L541-L549): `prompt_emb=style_context`. This is the single biggest design choice driving the collapse analysis in §1.

5. **Dataset circularity.** [`build_dataset.py`](src/rag_patch_training/build_dataset.py) retrieves references for each image with an image-weighted MMR query ($\lambda=0.9$, 70% image / 30% text). Refs therefore carry very similar style/content to the target. Combined with the prompt already describing the target, **references add almost no new information** relative to $(\text{prompt},\text{target})$ — so even a well-posed objective would struggle to extract gradient from them.

---

## 3. Proposed improvements (ranked by expected impact)

### A. Reframe the objective so Stream 2 has a job

Two options — pick one.

**A1. Explicit residual regression (preferred).** Drop the gate during training. Let Stream 2 directly regress the base residual:

$$
\mathcal{L}_\text{res} \;=\; \big\|\, S_\varphi \;-\; \big(v^* - e_0^{\,\text{detach}}\big) \,\big\|_2^2
$$

This target is **non-degenerate**: $v^* - e_0$ is whatever the base gets wrong, by definition. Re-introduce $g$ only at inference as a strength slider, initialised at $1$. No linear-term cancellation, no gate collapse, no need for ControlNet-style warm-up.

**A2. Give Stream 2 the prompt too.** Pass `prompt_emb` concatenated (or summed) with the style token. Then Stream 2 has $\supseteq$ the conditioning of Stream 1, so $S_\varphi$ can only reduce loss. Keep the additive-gate formulation; $g$ will move.

### B. Pick a task where references carry signal the prompt lacks

The current (prompt, target, refs) triples are near-circular. Retrieval-augmented generation pays off when refs provide information the text cannot:

- **Style transfer.** Prompt = content description; refs = style exemplars from a different semantic domain; target = content-in-style.
- **Subject-driven generation.** Refs = the same subject from different angles; prompt = novel scene.
- **Editing / attribute insertion.** Refs exhibit the attribute to transfer.

Changing the *data* is likely higher-leverage than any loss tweak.

### C. Richer style encoder

Replace `mean + std → 1 token` with:

- Per-layer **Gram matrices** $G^{(\ell)} = \tfrac{1}{H_\ell W_\ell}\,F^{(\ell)} F^{(\ell)\top}$, flattened and projected. Gram is the canonical style statistic (Gatys et al.) and captures cross-channel correlations that first/second moments alone discard.
- Emit **$K$ style tokens** (e.g. one per VGG layer per ref, or a small learned Perceiver pool), so FLUX's cross-attention has structure to attend to instead of a single vector.

### D. Gate schedule / per-block gate

- If keeping the additive-gate form (A2), warm the gate from $0\to 1$ over the first few thousand steps instead of leaving it fully free.
- Or use a **per-block** scalar $g_\ell$ — different depths adopt the correction at different rates. This is strictly more expressive than a single scalar, and common in ControlNet/T2I-Adapter variants.

### E. Train for real

$625$ steps is a smoke test. Budget $\ge 10$–$20$k optimizer steps after the objective is fixed. Log $\|S_\varphi\|$, $\langle S_\varphi, e_0 - v^*\rangle$, and $g$ per step — the first confirms Stream 2 is alive; the second confirms the objective has signal; the third then actually moves.

---

## 4. Empirical validation plan

Since the gate-collapse argument in §1 is asymptotic (and real training runs are finite), verify with diagnostics rather than images:

1. **Gradient alignment probe.** Instrument the current training and log
    $$\rho_t \;=\; \frac{\langle S_\varphi, e_0 - v^*\rangle}{\|S_\varphi\|\,\|e_0 - v^*\|}$$
   per step. Prediction: $\mathbb{E}[\rho_t] \approx 0$ under the current setup. A near-zero running mean is direct evidence of the collapse mechanism described in §1.

2. **Ablation A1 (residual objective).** Re-run with $\mathcal{L}_\text{res}$. Expect $\|S_\varphi\|$ to grow and MSE against $v^* - e_0$ to decrease monotonically.

3. **Ablation A2 (prompt in Stream 2).** Re-run with the prompt passed into Stream 2. Expect $g$ to move off zero and final loss to beat the frozen baseline on a held-out split.

4. **Data ablation (B).** Replace the current retrieval with an explicit style-transfer dataset and rerun A1/A2. Expect the largest quality delta here.

If (1) confirms $\mathbb{E}[\rho_t] \approx 0$, you have a mathematical demonstration (not just an empirical one) that the current recipe cannot work — which is the "proof" you were after.
