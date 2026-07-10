| scene | kf | submaps | SLAM ATE | SLAM loops | ours ATE (base) | ours ATE (lc) | ours loops | ΔATE | max Δt | loop P | loop R | base gate | lc gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7s_chess [mapanything] | 29 | 2 | 0.0389 | 0 | 0.0517 | 0.0517 | 0 | 0.0128 | 2.2347 | — | — | — | HARMLESS |
| 7s_chess [vggt_omega] | 29 | 2 | 0.0389 | 0 | 0.0179 | 0.0179 | 0 | 0.0211 | 0.1304 | — | — | — | HARMLESS |
| 7s_chess [vggt_spark] | 29 | 2 | 0.0389 | 0 | 0.0392 | 0.0392 | 0 | 0.0003 | 0.0049 | — | — | PASS | PASS |
| 7s_chess [vggtx] | 29 | 2 | 0.0389 | 0 | 0.0386 | 0.0386 | 0 | 0.0003 | 0.0065 | — | — | — | HARMLESS |
| 7s_office [mapanything] | 58 | 6 | 0.1056 | 2 | 0.1106 | 0.1066 | 2 | 0.0010 | 1.5448 | 1.00 | 0.67 | — | HARMLESS / scale:BLOWUP |
| 7s_office@25% [mapanything] | 14 | 1 | 0.0631 | 0 | 0.0571 | 0.0571 | None | 0.0060 | 1.3483 | — | — | — | HARMLESS |
| 7s_office@50% [mapanything] | 29 | 2 | 0.1123 | 0 | 0.0997 | 0.0997 | 0 | 0.0126 | 1.3739 | — | — | — | HARMLESS |
| 7s_office [vggt_omega] | 58 | 6 | 0.1056 | 2 | 0.0274 | 0.0231 | 2 | 0.0825 | 0.2392 | 1.00 | 0.67 | — | HARMLESS / scale:OK |
| 7s_office@25% [vggt_omega] | 14 | 1 | 0.0631 | 0 | 0.0217 | 0.0217 | None | 0.0414 | 0.1308 | — | — | — | HARMLESS |
| 7s_office@50% [vggt_omega] | 29 | 2 | 0.1123 | 0 | 0.0216 | 0.0216 | 0 | 0.0908 | 0.2084 | — | — | — | HARMLESS |
| 7s_office [vggt_spark] | 58 | 6 | 0.1056 | 2 | 0.1130 | 0.1071 | 2 | 0.0015 | 0.0136 | 1.00 | 0.67 | FAIL | PASS / scale:BLOWUP |
| 7s_office@25% [vggt_spark] | 14 | 1 | 0.0631 | 0 | 0.0632 | 0.0632 | None | 0.0001 | 0.0116 | — | — | PASS | FAIL |
| 7s_office@50% [vggt_spark] | 29 | 2 | 0.1123 | 0 | 0.1106 | 0.1106 | 0 | 0.0017 | 0.0146 | — | — | PASS | PASS |
| 7s_office [vggtx] | 58 | 6 | 0.1056 | 2 | 0.1113 | 0.1043 | 2 | 0.0013 | 0.0153 | 1.00 | 0.67 | — | HARMLESS / scale:BLOWUP |
| 7s_office@25% [vggtx] | 14 | 1 | 0.0631 | 0 | 0.0631 | 0.0631 | None | 0.0000 | 0.0071 | — | — | — | HARMLESS |
| 7s_office@50% [vggtx] | 29 | 2 | 0.1123 | 0 | 0.1097 | 0.1097 | 0 | 0.0026 | 0.0096 | — | — | — | HARMLESS |
| 7s_redkitchen [mapanything] | 43 | 4 | 0.0543 | 1 | 0.0501 | 0.0740 | 1 | 0.0196 | 1.2818 | 1.00 | 1.00 | — | HARMFUL / scale:BLOWUP |
| 7s_redkitchen@25% [mapanything] | 11 | 1 | 0.0285 | 0 | 0.0225 | 0.0225 | None | 0.0060 | 0.7992 | — | — | — | HARMLESS |
| 7s_redkitchen@50% [mapanything] | 22 | 2 | 0.0329 | 0 | 0.0364 | 0.0364 | 0 | 0.0035 | 1.3029 | — | — | — | HARMLESS |
| 7s_redkitchen [vggt_omega] | 43 | 4 | 0.0543 | 1 | 0.0145 | 0.0152 | 1 | 0.0392 | 0.2111 | 1.00 | 1.00 | — | HARMLESS / scale:OK |
| 7s_redkitchen@25% [vggt_omega] | 11 | 1 | 0.0285 | 0 | 0.0111 | 0.0111 | None | 0.0174 | 0.0977 | — | — | — | HARMLESS |
| 7s_redkitchen@50% [vggt_omega] | 22 | 2 | 0.0329 | 0 | 0.0140 | 0.0140 | 0 | 0.0189 | 0.1809 | — | — | — | HARMLESS |
| 7s_redkitchen [vggt_spark] | 43 | 4 | 0.0543 | 1 | 0.0544 | 0.0517 | 1 | 0.0026 | 0.0077 | 1.00 | 1.00 | PASS | PASS / scale:BLOWUP |
| 7s_redkitchen@25% [vggt_spark] | 11 | 1 | 0.0285 | 0 | 0.0285 | 0.0285 | None | 0.0000 | 0.0013 | — | — | PASS | FAIL |
| 7s_redkitchen@50% [vggt_spark] | 22 | 2 | 0.0329 | 0 | 0.0331 | 0.0331 | 0 | 0.0002 | 0.0030 | — | — | PASS | PASS |
| 7s_redkitchen [vggtx] | 43 | 4 | 0.0543 | 1 | 0.0527 | 0.0499 | 1 | 0.0044 | 0.0092 | 1.00 | 1.00 | — | HARMLESS / scale:BLOWUP |
| 7s_redkitchen@25% [vggtx] | 11 | 1 | 0.0285 | 0 | 0.0289 | 0.0289 | None | 0.0004 | 0.0028 | — | — | — | HARMLESS |
| 7s_redkitchen@50% [vggtx] | 22 | 2 | 0.0329 | 0 | 0.0326 | 0.0326 | 0 | 0.0004 | 0.0036 | — | — | — | HARMLESS |
| tum_fr3_office [mapanything] | 75 | 7 | 0.0319 | 2 | 0.1296 | 0.0623 | 2 | 0.0304 | 4.5470 | 1.00 | 0.67 | — | HARMLESS / scale:BLOWUP |
| tum_fr3_office@25% [mapanything] | 19 | 2 | 0.0261 | 0 | 0.0321 | 0.0321 | 0 | 0.0060 | 4.1389 | — | — | — | HARMLESS |
| tum_fr3_office@50% [mapanything] | 38 | 3 | 0.0320 | 0 | 0.0728 | 0.0728 | 0 | 0.0408 | 4.5852 | — | 0.00 | — | HARMLESS |
| tum_fr3_office [vggt_omega] | 75 | 7 | 0.0319 | 2 | 0.0510 | 0.0377 | 2 | 0.0058 | 0.1009 | 1.00 | 0.67 | — | HARMLESS / scale:OK |
| tum_fr3_office@25% [vggt_omega] | 19 | 2 | 0.0261 | 0 | 0.0407 | 0.0407 | 0 | 0.0145 | 0.1023 | — | — | — | HARMLESS |
| tum_fr3_office@50% [vggt_omega] | 38 | 3 | 0.0320 | 0 | 0.0411 | 0.0411 | 0 | 0.0090 | 0.1936 | — | 0.00 | — | HARMLESS |
| tum_fr3_office [vggt_spark] | 75 | 7 | 0.0319 | 2 | 0.0450 | 0.0320 | 2 | 0.0001 | 0.0326 | 1.00 | 0.67 | FAIL | PASS / scale:OK |
| tum_fr3_office@25% [vggt_spark] | 19 | 2 | 0.0261 | 0 | 0.0300 | 0.0300 | 0 | 0.0038 | 0.0320 | — | — | PASS | PASS |
| tum_fr3_office@50% [vggt_spark] | 38 | 3 | 0.0320 | 0 | 0.0321 | 0.0321 | 0 | 0.0000 | 0.0320 | — | 0.00 | PASS | PASS |
| tum_fr3_office [vggtx] | 75 | 7 | 0.0319 | 2 | 0.0427 | 0.0303 | 2 | 0.0016 | 0.0259 | 1.00 | 0.67 | — | HARMLESS / scale:OK |
| tum_fr3_office@25% [vggtx] | 19 | 2 | 0.0261 | 0 | 0.0246 | 0.0246 | 0 | 0.0015 | 0.0162 | — | — | — | HARMLESS |
| tum_fr3_office@50% [vggtx] | 38 | 3 | 0.0320 | 0 | 0.0296 | 0.0296 | 0 | 0.0024 | 0.0171 | — | 0.00 | — | HARMLESS |
