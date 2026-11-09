# Day 06｜Tensor Dimensions, Broadcasting & 3D Linear Layer

> Transformer From Zero — 365 Days  
> Date: 2026-11-09  
> Theme: Tensor Shape / Batch × Sequence × Feature / Batched MatMul / Broadcasting  
> Status: Task A–B completed; all five exercise tests passed; final training demo succeeded.
> Learning notes: [Day 06 complete notes](weekly-notes/2026-11-09-day06-3d-sequence-linear-layer.md)
> Knowledge word cloud: [Day 06 word cloud](weekly-notes/images/day06-knowledge-wordcloud.svg)

## Today's goal

Move from the 2D batch representation used in earlier linear layers to the 3D sequence representation used by Transformer models.

By the end of this session, you should be able to:

1. Distinguish tensor rank, shape, and axis.
2. Explain the roles of Batch (B), Sequence (T), and Feature (D).
3. Derive the output shape of a batched linear projection.
4. Explain how a one-dimensional bias broadcasts over batch and token axes.
5. Implement a shared 3D linear layer and manually derive its gradients.

## 1. Tensor notation

For a batch of token sequences, use:

\[
X\in\mathbb{R}^{B\times T\times D}
\]

- B: number of sequences in the batch.
- T: number of tokens in each sequence.
- D: feature dimension of each token.

Example:

\[
X:(4,8,64)
\]

means 4 sequences, 8 tokens per sequence, and 64 features per token.

Remember:

- x.ndim is the number of axes (rank).
- x.shape gives the length of each axis.

## 2. Task A — 3D linear forward

File: exercises/week06.py  
Function: linear_3d_numpy(x, weight, bias)

Input shapes:

| Variable | Shape |
|---|---|
| x | (B, T, D) |
| weight | (D, H) |
| bias | (H,) |
| output | (B, T, H) |

Use:

\[
Y=XW+b
\]

NumPy @ performs matrix multiplication over the final two dimensions and preserves the leading batch dimension. The bias is broadcast over B and T.

Example shape derivation:

\[
(4,8,64)@(64,128)\rightarrow(4,8,128)
\]

The same W and b are used by every token. The linear layer transforms each token independently; it does not itself mix information between different token positions.

Run the focused test:

~~~powershell
python -m pytest -q -m exercise tests/test_week06_exercises.py::test_linear_3d_numpy_matches_torch_and_preserves_inputs
~~~

Then test the bias broadcast:

~~~powershell
python -m pytest -q -m exercise tests/test_week06_exercises.py::test_linear_3d_numpy_broadcasts_one_bias_to_every_token
~~~

## 3. Task B — manual gradients for a shared 3D linear layer

File: exercises/week06.py  
Function: linear_3d_mse_gradients(x, weight, bias, targets)

Shapes:

| Variable | Shape |
|---|---|
| x | (B, T, D) |
| weight | (D, H) |
| bias | (H,) |
| targets | (B, T, H) |
| predictions | (B, T, H) |

Forward computation:

\[
P=XW+b,\qquad E=P-Y
\]

Use this exact loss convention:

\[
L=\frac{1}{2BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\sum_{h=1}^{H}E_{bth}^{2}
\]

Important: sum over H for each token, then average over B*T tokens. Do NOT divide by B*T*H.

Let \(M=BT\). You can flatten the first two dimensions:

\[
X_{\mathrm{flat}}\in\mathbb{R}^{M\times D},\qquad
E_{\mathrm{flat}}\in\mathbb{R}^{M\times H}
\]

The gradient derivation should reduce to the familiar Week 02 matrix formulas:

\[
\frac{\partial L}{\partial W}
=\frac{X_{\mathrm{flat}}^\top E_{\mathrm{flat}}}{M}
\]

\[
\frac{\partial L}{\partial b}
=\frac{1}{M}\sum_{m=1}^{M}E_{\mathrm{flat},m}
\]

This is the key new idea: because W and b are shared across all tokens, their gradients accumulate contributions from every token in every sequence.

Run the gradient comparison:

~~~powershell
python -m pytest -q -m exercise tests/test_week06_exercises.py::test_linear_3d_mse_gradients_match_torch_autograd
~~~

Check the loss reduction convention:

~~~powershell
python -m pytest -q -m exercise tests/test_week06_exercises.py::test_linear_3d_mse_loss_sums_features_and_averages_tokens
~~~

Check one gradient-descent update:

~~~powershell
python -m pytest -q -m exercise tests/test_week06_exercises.py::test_linear_3d_gradient_one_step_reduces_loss
~~~

## 4. Full exercise test

After completing Tasks A and B:

~~~powershell
python -m pytest -q -m exercise tests/test_week06_exercises.py
~~~

Expected final result after all tasks are implemented:

~~~text
5 passed
~~~

The implementation is complete. The five exercise tests passed in the user's local environment; the final end-to-end training demo also completed successfully. PyTorch is used only as a test reference, not in the NumPy implementation.

## 5. Questions to answer before moving to Attention

1. Why does (B,T,D) @ (D,H) produce (B,T,H) rather than (B,T,D,H)?
2. Why can a bias of shape (H,) be added to a tensor of shape (B,T,H)?
3. Why does the linear layer not allow one token to directly read another token's features?
4. Why must the gradient of a shared W aggregate contributions from all B*T tokens?
5. In this exercise, why is the loss divided by B*T rather than B*T*H?

## 6. Day 06 completion criteria

- [x] linear_3d_numpy implemented and tested.
- [x] linear_3d_mse_gradients implemented by hand.
- [x] All five exercise tests pass.
- [x] Explain parameter sharing and gradient accumulation.
- [x] Final sequence-linear training demo succeeds and outputs are inspected.

Day 06 is complete. The next stage can move toward token-to-token information exchange with Attention.


## 7. Final training demo — shared 3D sequence linear layer

After all five exercise tests pass, run the end-to-end demo:

~~~powershell
python scripts/day06_sequence_linear_demo.py
~~~

The demo creates a deterministic synthetic sequence-regression task. Every token is generated by one shared ground-truth mapping:

\[
Y_{b,t}=X_{b,t}W_{\mathrm{true}}+b_{\mathrm{true}}
\]

It then:

1. Checks the hand-derived weight and bias gradients independently with Day 03 central finite differences.
2. Trains the shared 3D linear layer using only the Week 06 NumPy gradients.
3. Evaluates both training sequences and independently sampled validation sequences.
4. Reports parameter-recovery errors and selected token predictions.

Expected local outputs (not committed):

- \`results/day06/metrics.json\`
- \`results/day06/loss_curve.png\`
- \`results/day06/sequence_prediction_comparison.png\`

The success condition requires both training and validation loss to fall substantially, and the learned shared parameters to recover the synthetic ground truth within the configured tolerance.

Important: this demo imports the implementations in \`exercises/week06.py\`. Complete Tasks A and B and pass the five exercise tests before running it.

## 8. Day 06 completion criteria

- [ ] \`linear_3d_numpy\` implemented and tested.
- [ ] \`linear_3d_mse_gradients\` implemented by hand.
- [ ] All five exercise tests pass.
- [ ] Explain parameter sharing and gradient accumulation in your own words.
- [ ] Final sequence-linear training demo succeeds and outputs are inspected.

Only after these checks should we move on to the next sequence-modeling component.
