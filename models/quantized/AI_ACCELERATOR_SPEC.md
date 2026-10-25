# AI Accelerator Specification

## 1\. Model Architecture

MLP:

17 -> 32 -> 16 -> 3

* Input: 17 features
* Hidden Layer 1: 32 neurons
* Hidden Layer 2: 16 neurons
* Output: 3 classes

Classes:

* 0 = Normal
* 1 = DoS
* 2 = Port Scan

## 2\. Input Feature Order

|Index|Feature|
|-|-|
|0|id.orig\_p|
|1|id.resp\_p|
|2|proto|
|3|duration|
|4|orig\_bytes|
|5|resp\_bytes|
|6|conn\_state|
|7|orig\_pkts|
|8|resp\_pkts|
|9|orig\_ip\_bytes|
|10|resp\_ip\_bytes|
|11|conn\_count\_10s|
|12|unique\_dst\_ports\_10s|
|13|unique\_dst\_ips\_10s|
|14|failed\_conn\_count\_10s|
|15|failed\_ratio\_10s|
|16|s0\_ratio\_10s|

## 3\. INT8 Quantization

Input:

X\_q = round(X\_scaled / S\_X)

Clamp to:

\-127 <= X\_q <= 127

Activation values are INT8.

Weights are INT8.

Bias values are INT32.

## 4\. Quantization Scales

### Input

S\_X = 0.03342512085681825

### Hidden Layer 1

S\_H1 = 0.04276043599046121

### Hidden Layer 2

S\_H2 = 0.23355236203651727

### Weights

S\_W1 = 0.01183601189404726

S\_W2 = 0.01773611642420292

S\_W3 = 0.01674969680607319

### Bias

S\_B1 = 0.00039562012802126797

S\_B2 = 0.0007584040710764968

S\_B3 = 0.003911931252453903

## 5\. Layer 1

For each neuron j:

ACC1\[j] = b1\_q\[j] + SUM(X\_q\[i] \* W1\_q\[j]\[i])

Apply ReLU:

if ACC1\[j] < 0:
H1\[j] = 0

Then requantize:

H1\_q = round(ACC1 \* (S\_X \* S\_W1) / S\_H1)

Clamp:

0 <= H1\_q <= 127

## 6\. Layer 2

For each neuron j:

ACC2\[j] = b2\_q\[j] + SUM(H1\_q\[i] \* W2\_q\[j]\[i])

Apply ReLU.

Then:

H2\_q = round(ACC2 \* (S\_H1 \* S\_W2) / S\_H2)

Clamp:

0 <= H2\_q <= 127

## 7\. Layer 3

For each output neuron j:

ACC3\[j] = b3\_q\[j] + SUM(H2\_q\[i] \* W3\_q\[j]\[i])

No ReLU is applied.

ACC3 is INT32.

## 8\. Classification

The final class is the index of the maximum ACC3 value.

Example:

ACC3 = \[-3022, 956, -885]

Maximum = 956

Therefore:

class = 1

Class mapping:

0 = Normal

1 = DoS

2 = Port Scan

## 9\. Memory Requirements

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

## 10\. Hardware Datapath

Input INT8

\-> Layer 1 MAC

\-> ReLU

\-> Requantization

\-> H1

\-> Layer 2 MAC

\-> ReLU

\-> Requantization

\-> H2

\-> Layer 3 MAC

\-> ACC3

\-> Argmax

\-> Class

## 11\. Verification

Golden vectors are available in:

results/golden/

Files:

* input\_int8.txt
* expected\_h1\_int8.txt
* expected\_h2\_int8.txt
* expected\_acc3\_int32.txt
* expected\_class.txt
* expected\_label.txt

Hardware verification target:

H1 mismatch = 0

H2 mismatch = 0

ACC3 mismatch = 0

Class mismatch = 0

## 12\. Python Reference Accuracy

Float accuracy:

98.7800%

INT8 accuracy:

98.7567%

Accuracy loss:

0.0233%

Golden vector accuracy:

99.10%

## 13\. Important Hardware Notes

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

\## Fixed-Point Verification



The fixed-point implementation was verified against 1,000 golden test vectors.



Fixed-point parameters:



\- SHIFT = 24

\- H1 multiplier M1 = 155223

\- H2 multiplier M2 = 54480



Verification results:



\- H1 errors: 0

\- H2 errors: 0

\- ACC3 errors: 0

\- Classification errors: 0



Therefore:



\*\*FIXED-POINT VERIFICATION: PASS\*\*



The fixed-point calculation matches the Python golden reference for all 1,000 test samples.

