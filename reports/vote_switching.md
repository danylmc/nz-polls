# Vote Switching and Flows in NZ Elections

*Where do voters go when they switch? What predicts switching?*

## 5a. Transition Matrices


### 1999

| prev_vote   |   Labour |   National |   NZ First |   ACT |
|:------------|---------:|-----------:|-----------:|------:|
| Labour      |       80 |          5 |          2 |     1 |
| National    |       20 |         67 |          2 |     8 |
| NZ First    |       34 |         15 |         32 |     2 |
| ACT         |        7 |         25 |          1 |    63 |


### 2002

| prev_vote   |   Labour |   National |   Green |   NZ First |   ACT |
|:------------|---------:|-----------:|--------:|-----------:|------:|
| Labour      |       77 |          3 |       5 |          6 |     2 |
| National    |       14 |         58 |       1 |          9 |     9 |
| Green       |       21 |          4 |      58 |          5 |     3 |
| NZ First    |       21 |          7 |       4 |         60 |     2 |
| ACT         |        4 |         23 |       3 |          5 |    56 |


### 2005

| prev_vote   |   Labour |   National |   Green |   NZ First |   ACT |
|:------------|---------:|-----------:|--------:|-----------:|------:|
| Labour      |       70 |         11 |       3 |          3 |     0 |
| National    |        7 |         85 |       1 |          3 |     2 |
| Green       |       22 |          9 |      50 |          3 |     2 |
| NZ First    |       18 |         25 |       1 |         37 |     0 |
| ACT         |        5 |         72 |       2 |          2 |    18 |


### 2008

| prev_vote   |   Labour |   National |   Green |   NZ First |   ACT |
|:------------|---------:|-----------:|--------:|-----------:|------:|
| Labour      |       68 |         14 |       5 |          4 |     1 |
| National    |        2 |         91 |       1 |          1 |     4 |
| Green       |       17 |         13 |      59 |          3 |     0 |
| NZ First    |        8 |         23 |       2 |         55 |     3 |
| ACT         |        2 |         48 |       0 |          0 |    46 |


### 2011

| prev_vote   |   Labour |   National |   Green |   NZ First |   ACT |
|:------------|---------:|-----------:|--------:|-----------:|------:|
| Labour      |       68 |          8 |      11 |          9 |     0 |
| National    |        6 |         85 |       4 |          2 |     1 |
| Green       |       15 |          5 |      75 |          4 |     0 |
| NZ First    |       14 |          7 |       6 |         62 |     2 |
| ACT         |        3 |         45 |       6 |          6 |    30 |


### 2014

| prev_vote   |   Labour |   National |   Green |   NZ First |   ACT |
|:------------|---------:|-----------:|--------:|-----------:|------:|
| Labour      |       67 |          9 |      12 |          8 |     0 |
| National    |        3 |         88 |       2 |          3 |     0 |
| Green       |       13 |          6 |      70 |          6 |     0 |
| NZ First    |       15 |          7 |       5 |         69 |     0 |
| ACT         |        0 |         35 |       0 |          0 |    35 |


### 2017

| prev_vote   |   Labour |   National |   Green |   NZ First |   ACT |
|:------------|---------:|-----------:|--------:|-----------:|------:|
| Labour      |       83 |          4 |       5 |          4 |     0 |
| National    |       11 |         80 |       2 |          4 |     1 |
| Green       |       45 |          4 |      39 |          1 |     0 |
| NZ First    |       32 |          6 |       3 |         55 |     0 |
| ACT         |       20 |         40 |       0 |          0 |    20 |


### 2020

| prev_vote   |   Labour |   National |   Green |   NZ First |   ACT |
|:------------|---------:|-----------:|--------:|-----------:|------:|
| Labour      |       85 |          2 |       8 |          1 |     1 |
| National    |       23 |         59 |       1 |          1 |    15 |
| Green       |       32 |          2 |      61 |          1 |     2 |
| NZ First    |       41 |          8 |       5 |         27 |    15 |
| ACT         |       12 |         18 |       6 |          0 |    65 |


### 2023

| prev_vote   |   Labour |   National |   Green |   NZ First |   ACT |
|:------------|---------:|-----------:|--------:|-----------:|------:|
| Labour      |       54 |         15 |      16 |          6 |     3 |
| National    |        1 |         86 |       1 |          3 |     9 |
| Green       |       11 |          3 |      73 |          1 |     2 |
| NZ First    |        9 |         13 |       2 |         58 |    18 |
| ACT         |        0 |         37 |       0 |          5 |    57 |


### Retention Rates

| Year | ACT | Green | Labour | NZ First | National |
|------|------|------|------|------|------|
| 1999 | 63% | — | 80% | 32% | 67% |
| 2002 | 56% | 58% | 77% | 60% | 58% |
| 2005 | 18% | 50% | 70% | 37% | 85% |
| 2008 | 46% | 59% | 68% | 55% | 91% |
| 2011 | 30% | 75% | 68% | 62% | 85% |
| 2014 | 35% | 70% | 67% | 69% | 88% |
| 2017 | 20% | 39% | 83% | 55% | 80% |
| 2020 | 65% | 61% | 85% | 27% | 59% |
| 2023 | 57% | 73% | 54% | 58% | 86% |

![Retention Rates](../graphs/switching_retention.png)

## 5b. Who Switches?

| Year | Switching Rate | n |
|------|---------------|---|
| 1999 | 35.5% | 1742 |
| 2002 | 34.8% | 4378 |
| 2005 | 33.8% | 2891 |
| 2008 | 26.8% | 2344 |
| 2011 | 24.4% | 2191 |
| 2014 | 22.0% | 2133 |
| 2017 | 26.5% | 2660 |
| 2020 | 30.9% | 2820 |
| 2023 | 33.6% | 1612 |

### Switcher vs Loyalist Demographics

| Variable | Loyalists | Switchers | p-value |
|----------|-----------|-----------|---------|
| age | 55.33 | 51.59 | 0.0000** |
| female | 0.56 | 0.55 | 0.0539 |
| education | 1.66 | 1.82 | 0.0000** |
| lr_self | 5.42 | 5.07 | 0.0000** |
| lr_extreme | 2.08 | 1.68 | 0.0000** |

![Switcher Profile](../graphs/switching_profile.png)

## 5c. Direction of Switching

| Direction | Count | Share |
|-----------|-------|-------|
| Right To Left | 1539 | 22.4% |
| Left To Right | 1505 | 21.9% |
| Within Left | 1376 | 20.0% |
| Within Right | 1260 | 18.3% |
| Other | 1188 | 17.3% |

![Direction](../graphs/switching_direction.png)
