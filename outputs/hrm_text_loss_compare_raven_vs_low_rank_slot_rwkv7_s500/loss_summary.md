# HRM-Text Loss Comparison

Objective: causal LM over the local HRM-Text token stream.

Shape: hidden=256, layers=2, batch=4, seq=128, vocab=65536

| mixer | params M | initial val | final val | best val | final train | tok/s | peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raven | 35.39 | 11.1281 | 2.9195 | 2.9195 | 2.5000 | 49627 | 0.448 |
| low_rank_slot_rwkv7 | 35.56 | 11.0938 | 2.6273 | 2.6273 | 1.9531 | 52854 | 0.991 |

| step | raven | low_rank_slot_rwkv7 |
| ---: | ---: | ---: |
| 0 | 11.1281 | 11.0938 |
| 100 | 3.8078 | 3.2961 |
| 200 | 3.2234 | 2.8469 |
| 300 | 2.9813 | 2.6836 |
| 400 | 2.9367 | 2.6406 |
| 500 | 2.9195 | 2.6273 |
