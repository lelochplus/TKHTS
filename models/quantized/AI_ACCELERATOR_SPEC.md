# AI Accelerator Specification

## 1. Model Architecture

MLP:

17 -> 32 -> 16 -> 3

- Input: 17 features
- Hidden Layer 1: 32 neurons
- Hidden Layer 2: 16 neurons
- Output: 3 classes

Classes:

- 0 = Normal
- 1 = DoS
- 2 = Port Scan

## 2. Input Feature Order

| Index | Feature |
|---|---|
| 0 | id.orig_p |
| 1 | id.resp_p |
| 2 | proto |
| 3 | duration |
| 4 | orig_bytes |
| 5 | resp_bytes |
| 6 | conn_state |
| 7 | orig_pkts |
| 8 | resp_pkts |
| 9 | orig_ip_bytes |
| 10 | resp_ip_bytes |
| 11 | conn_count_10s |
| 12 | unique_dst_ports_10s |
| 13 | unique_dst_ips_10s |
| 14 | failed_conn_count_10s |
| 15 | failed_ratio_10s |
| 16 | s0_ratio_10s |

## 3. INT8 Quantization

Input:

X_q = round(X_scaled / S_X)

Clamp to:

-127 <= X_q <= 127

Activation values are INT8.

Weights are INT8.

Bias values are INT32.

## 4. Quantization Scales

### Input

S_X = 0.03342512085681825

### Hidden Layer 1

S_H1 = 0.04276043599046121

### Hidden Layer 2

S_H2 = 0.23355236203651727

### Weights

S_W1 = 0.01183601189404726

S_W2 = 0.01773611642420292

S_W3 = 0.01674969680607319

### Bias

S_B1 = 0.00039562012802126797

S_B2 = 0.0007584040710764968

S_B3 = 0.003911931252453903

## 5. Layer 1

For each neuron j:

ACC1[j] = b1_q[j] + SUM(X_q[i] * W1_q[j][i])

Apply ReLU:

if ACC1[j] < 0:
    H1[j] = 0

Then requantize:

H1_q = round(ACC1 * (S_X * S_W1) / S_H1)

Clamp:

0 <= H1_q <= 127

## 6. Layer 2

For each neuron j:

ACC2[j] = b2_q[j] + SUM(H1_q[i] * W2_q[j][i])

Apply ReLU.

Then:

H2_q = round(ACC2 * (S_H1 * S_W2) / S_H2)

Clamp:

0 <= H2_q <= 127

## 7. Layer 3

For each output neuron j:

ACC3[j] = b3_q[j] + SUM(H2_q[i] * W3_q[j][i])

No ReLU is applied.

ACC3 is INT32.

## 8. Classification

The final class is the index of the maximum ACC3 value.

Example:

ACC3 = [-3022, 956, -885]

Maximum = 956

Therefore:

class = 1

Class mapping:

0 = Normal

1 = DoS

2 = Port Scan

## 9. Memory Requirements

W1:

32 x 17 = 544 INT8

b1:

32 INT32

W2:

16 x 32 = 512 INT8

b2:

16 INT32

W3:

3 x 16 = 48 INT8

b3:

3 INT32

Total:

1104 INT8 weights

51 INT32 bias values

## 10. Hardware Datapath

Input INT8

-> Layer 1 MAC

-> ReLU

-> Requantization

-> H1

-> Layer 2 MAC

-> ReLU

-> Requantization

-> H2

-> Layer 3 MAC

-> ACC3

-> Argmax

-> Class

## 11. Verification

Golden vectors are available in:

results/golden/

Files:

- input_int8.txt
- expected_h1_int8.txt
- expected_h2_int8.txt
- expected_acc3_int32.txt
- expected_class.txt
- expected_label.txt

Hardware verification target:

H1 mismatch = 0

H2 mismatch = 0

ACC3 mismatch = 0

Class mismatch = 0

## 12. Python Reference Accuracy

Float accuracy:

98.7800%

INT8 accuracy:

98.7567%

Accuracy loss:

0.0233%

Golden vector accuracy:

99.10%

## 13. Important Hardware Notes

The Verilog implementation must preserve:

1. Feature order
2. Weight values
3. Bias values
4. ReLU behavior
5. Requantization behavior
6. Argmax behavior

The hardware must NOT use floating-point arithmetic.

The requantization scale ratios must be converted to fixed-point integer multiplication and shift operations.

The final fixed-point implementation must be verified against the golden vectors.
