# ToolRet direct choice

Single-turn selection of a relevant tool from real source catalogs.

Primary pooled comparison uses the same common catalogs in every arm: webtools_spotify, webtools_tmdb, tooleyes, apibank.
Secondary all_catalogs rows include every selected catalog and are not a five-arm comparison.
Original run completed: True; original stop reason: none. Applicability does not change this historical status.

95% intervals resample tasks together, 2,000 resamples, seed 0. Abstentions and errors are misses in the positive-request headline. None-option metrics use parsed requests; errors are listed separately.

| Arm | Catalog | Relevant picks / positives | Paired Δ vs hybrid | Correct / wrong / abstained | Negatives | Errors |
|---|---|---:|---:|---:|---:|---:|
| hybrid@20 | pooled | 0.545 (0.486 to 0.600) | — | 158 / 277 / 0 | 145/435 (33.3%) | 0 |
| hybrid@20 | all_catalogs | 0.524 (0.480 to 0.567) | — | 257 / 478 / 0 | 245/735 (33.3%) | 0 |
| hybrid@20 | webtools_spotify | 0.650 (0.500 to 0.800) | — | 26 / 34 / 0 | 20/60 (33.3%) | 0 |
| hybrid@20 | webtools_tmdb | 0.500 (0.370 to 0.630) | — | 27 / 54 / 0 | 27/81 (33.3%) | 0 |
| hybrid@20 | tooleyes | 0.453 (0.358 to 0.558) | — | 43 / 100 / 0 | 48/143 (33.6%) | 0 |
| hybrid@20 | apibank | 0.614 (0.515 to 0.703) | — | 62 / 89 / 0 | 50/151 (33.1%) | 0 |
| hybrid@20 | metatool_which | 0.495 (0.430 to 0.560) | — | 99 / 201 / 0 | 100/300 (33.3%) | 0 |
| hybrid@20+jev | pooled | 0.714 (0.659 to 0.766) | 0.169 (0.107 to 0.231) | 207 / 122 / 106 | 145/435 (33.3%) | 0 |
| hybrid@20+jev | all_catalogs | 0.690 (0.645 to 0.729) | 0.165 (0.118 to 0.210) | 338 / 225 / 172 | 245/735 (33.3%) | 0 |
| hybrid@20+jev | webtools_spotify | 0.900 (0.800 to 0.975) | 0.250 (0.075 to 0.425) | 36 / 11 / 13 | 20/60 (33.3%) | 0 |
| hybrid@20+jev | webtools_tmdb | 0.870 (0.778 to 0.963) | 0.370 (0.222 to 0.519) | 47 / 29 / 5 | 27/81 (33.3%) | 0 |
| hybrid@20+jev | tooleyes | 0.505 (0.400 to 0.600) | 0.053 (-0.053 to 0.147) | 48 / 40 / 55 | 48/143 (33.6%) | 0 |
| hybrid@20+jev | apibank | 0.752 (0.663 to 0.832) | 0.139 (0.040 to 0.238) | 76 / 42 / 33 | 50/151 (33.1%) | 0 |
| hybrid@20+jev | metatool_which | 0.655 (0.590 to 0.720) | 0.160 (0.095 to 0.230) | 131 / 103 / 66 | 100/300 (33.3%) | 0 |
| jev-all | pooled | 0.741 (0.690 to 0.790) | 0.197 (0.138 to 0.259) | 215 / 121 / 99 | 145/435 (33.3%) | 0 |
| jev-all | all_catalogs | 0.733 (0.692 to 0.771) | 0.208 (0.161 to 0.257) | 359 / 270 / 106 | 245/735 (33.3%) | 0 |
| jev-all | webtools_spotify | 0.975 (0.925 to 1.000) | 0.325 (0.175 to 0.475) | 39 / 12 / 9 | 20/60 (33.3%) | 0 |
| jev-all | webtools_tmdb | 0.944 (0.870 to 1.000) | 0.444 (0.296 to 0.574) | 51 / 27 / 3 | 27/81 (33.3%) | 0 |
| jev-all | tooleyes | 0.526 (0.421 to 0.632) | 0.074 (-0.032 to 0.168) | 50 / 41 / 52 | 48/143 (33.6%) | 0 |
| jev-all | apibank | 0.743 (0.663 to 0.832) | 0.129 (0.030 to 0.228) | 75 / 41 / 35 | 50/151 (33.1%) | 0 |
| jev-all | metatool_which | 0.720 (0.655 to 0.780) | 0.225 (0.145 to 0.305) | 144 / 149 / 7 | 100/300 (33.3%) | 0 |
| agent@20 | pooled | 0.617 (0.559 to 0.669) | 0.072 (0.000 to 0.145) | 179 / 145 / 111 | 145/435 (33.3%) | 0 |
| agent@20 | all_catalogs | 0.610 (0.567 to 0.653) | 0.086 (0.035 to 0.137) | 299 / 258 / 178 | 245/735 (33.3%) | 0 |
| agent@20 | webtools_spotify | 0.825 (0.700 to 0.925) | 0.175 (0.000 to 0.350) | 33 / 23 / 4 | 20/60 (33.3%) | 0 |
| agent@20 | webtools_tmdb | 0.870 (0.778 to 0.944) | 0.370 (0.241 to 0.500) | 47 / 33 / 1 | 27/81 (33.3%) | 0 |
| agent@20 | tooleyes | 0.674 (0.579 to 0.768) | 0.221 (0.126 to 0.316) | 64 / 49 / 30 | 48/143 (33.6%) | 0 |
| agent@20 | apibank | 0.347 (0.257 to 0.446) | -0.267 (-0.406 to -0.129) | 35 / 40 / 76 | 50/151 (33.1%) | 0 |
| agent@20 | metatool_which | 0.600 (0.530 to 0.665) | 0.105 (0.035 to 0.170) | 120 / 113 / 67 | 100/300 (33.3%) | 0 |
| agent-all | pooled | 0.607 (0.552 to 0.662) | 0.062 (-0.014 to 0.135) | 176 / 120 / 139 | 145/435 (33.3%) | 0 |
| agent-all | webtools_spotify | 0.800 (0.675 to 0.925) | 0.150 (0.000 to 0.300) | 32 / 21 / 7 | 20/60 (33.3%) | 0 |
| agent-all | webtools_tmdb | 0.963 (0.907 to 1.000) | 0.463 (0.333 to 0.593) | 52 / 28 / 1 | 27/81 (33.3%) | 0 |
| agent-all | tooleyes | 0.611 (0.516 to 0.705) | 0.158 (0.053 to 0.263) | 58 / 49 / 36 | 48/143 (33.6%) | 0 |
| agent-all | apibank | 0.337 (0.248 to 0.426) | -0.277 (-0.416 to -0.129) | 34 / 22 / 95 | 50/151 (33.1%) | 0 |
| agent-all | metatool_which (not applicable) | n/a | n/a | n/a | n/a | 0 rejected attempts, excluded |
| agent-luna@20 | pooled | 0.776 (0.724 to 0.821) | 0.231 (0.172 to 0.293) | 225 / 210 / 0 | 145/435 (33.3%) | 0 |
| agent-luna@20 | all_catalogs | 0.729 (0.688 to 0.765) | 0.204 (0.159 to 0.247) | 357 / 377 / 1 | 245/735 (33.3%) | 0 |
| agent-luna@20 | webtools_spotify | 0.875 (0.750 to 0.975) | 0.225 (0.050 to 0.400) | 35 / 25 / 0 | 20/60 (33.3%) | 0 |
| agent-luna@20 | webtools_tmdb | 0.796 (0.685 to 0.889) | 0.296 (0.148 to 0.444) | 43 / 38 / 0 | 27/81 (33.3%) | 0 |
| agent-luna@20 | tooleyes | 0.705 (0.610 to 0.800) | 0.253 (0.158 to 0.347) | 67 / 76 / 0 | 48/143 (33.6%) | 0 |
| agent-luna@20 | apibank | 0.792 (0.713 to 0.871) | 0.178 (0.079 to 0.277) | 80 / 71 / 0 | 50/151 (33.1%) | 0 |
| agent-luna@20 | metatool_which | 0.660 (0.595 to 0.730) | 0.165 (0.105 to 0.225) | 132 / 167 / 1 | 100/300 (33.3%) | 0 |
| agent-luna-all | pooled | 0.797 (0.748 to 0.841) | 0.252 (0.193 to 0.310) | 231 / 204 / 0 | 145/435 (33.3%) | 0 |
| agent-luna-all | webtools_spotify | 0.900 (0.800 to 0.975) | 0.250 (0.100 to 0.400) | 36 / 24 / 0 | 20/60 (33.3%) | 0 |
| agent-luna-all | webtools_tmdb | 0.889 (0.796 to 0.963) | 0.389 (0.241 to 0.537) | 48 / 33 / 0 | 27/81 (33.3%) | 0 |
| agent-luna-all | tooleyes | 0.674 (0.579 to 0.768) | 0.221 (0.126 to 0.316) | 64 / 79 / 0 | 48/143 (33.6%) | 0 |
| agent-luna-all | apibank | 0.822 (0.743 to 0.891) | 0.208 (0.109 to 0.317) | 83 / 68 / 0 | 50/151 (33.1%) | 0 |
| agent-luna-all | metatool_which (not applicable) | n/a | n/a | n/a | n/a | 0 rejected attempts, excluded |
| hybrid@20+strands | pooled | 0.614 (0.559 to 0.669) | 0.069 (0.014 to 0.131) | 178 / 165 / 92 | 145/435 (33.3%) | 0 |
| hybrid@20+strands | all_catalogs | 0.624 (0.580 to 0.665) | 0.100 (0.055 to 0.145) | 306 / 259 / 170 | 245/735 (33.3%) | 0 |
| hybrid@20+strands | webtools_spotify | 0.750 (0.600 to 0.875) | 0.100 (-0.100 to 0.300) | 30 / 26 / 4 | 20/60 (33.3%) | 0 |
| hybrid@20+strands | webtools_tmdb | 0.537 (0.407 to 0.685) | 0.037 (-0.093 to 0.167) | 29 / 50 / 2 | 27/81 (33.3%) | 0 |
| hybrid@20+strands | tooleyes | 0.547 (0.442 to 0.642) | 0.095 (-0.011 to 0.200) | 52 / 34 / 57 | 48/143 (33.6%) | 0 |
| hybrid@20+strands | apibank | 0.663 (0.574 to 0.743) | 0.050 (-0.040 to 0.139) | 67 / 55 / 29 | 50/151 (33.1%) | 0 |
| hybrid@20+strands | metatool_which | 0.640 (0.570 to 0.705) | 0.145 (0.080 to 0.210) | 128 / 94 / 78 | 100/300 (33.3%) | 0 |
| strands-all, lower detail | pooled | 0.621 (0.562 to 0.676) | 0.076 (0.014 to 0.134) | 180 / 200 / 55 | 145/435 (33.3%) | 0 |
| strands-all, lower detail | all_catalogs | 0.596 (0.553 to 0.641) | 0.071 (0.022 to 0.122) | 292 / 362 / 81 | 245/735 (33.3%) | 0 |
| strands-all | webtools_spotify | 0.600 (0.450 to 0.750) | -0.050 (-0.225 to 0.125) | 24 / 36 / 0 | 20/60 (33.3%) | 0 |
| strands-all, lower detail | webtools_tmdb | 0.519 (0.389 to 0.648) | 0.019 (-0.130 to 0.167) | 28 / 52 / 1 | 27/81 (33.3%) | 0 |
| strands-all, lower detail | tooleyes | 0.537 (0.442 to 0.632) | 0.084 (-0.021 to 0.189) | 51 / 50 / 42 | 48/143 (33.6%) | 0 |
| strands-all, lower detail | apibank | 0.762 (0.683 to 0.842) | 0.149 (0.059 to 0.248) | 77 / 62 / 12 | 50/151 (33.1%) | 0 |
| strands-all, lower detail | metatool_which | 0.560 (0.495 to 0.625) | 0.065 (-0.015 to 0.145) | 112 / 162 / 26 | 100/300 (33.3%) | 0 |
| hybrid@20+clm-local | pooled | 0.155 (0.114 to 0.200) | -0.390 (-0.455 to -0.321) | 45 / 203 / 187 | 145/435 (33.3%) | 0 |
| hybrid@20+clm-local | all_catalogs | 0.194 (0.157 to 0.229) | -0.331 (-0.382 to -0.280) | 95 / 375 / 265 | 245/735 (33.3%) | 0 |
| hybrid@20+clm-local | webtools_spotify | 0.125 (0.025 to 0.225) | -0.525 (-0.700 to -0.350) | 5 / 39 / 16 | 20/60 (33.3%) | 0 |
| hybrid@20+clm-local | webtools_tmdb | 0.074 (0.019 to 0.148) | -0.426 (-0.574 to -0.278) | 4 / 31 / 46 | 27/81 (33.3%) | 0 |
| hybrid@20+clm-local | tooleyes | 0.263 (0.179 to 0.358) | -0.189 (-0.305 to -0.074) | 25 / 70 / 48 | 48/143 (33.6%) | 0 |
| hybrid@20+clm-local | apibank | 0.109 (0.059 to 0.168) | -0.505 (-0.614 to -0.396) | 11 / 63 / 77 | 50/151 (33.1%) | 0 |
| hybrid@20+clm-local | metatool_which | 0.250 (0.190 to 0.310) | -0.245 (-0.315 to -0.175) | 50 / 172 / 78 | 100/300 (33.3%) | 0 |
| clm-local-all | pooled | 0.138 (0.100 to 0.179) | -0.407 (-0.472 to -0.338) | 40 / 258 / 137 | 145/435 (33.3%) | 0 |
| clm-local-all | all_catalogs | 0.163 (0.129 to 0.196) | -0.361 (-0.410 to -0.310) | 80 / 482 / 173 | 245/735 (33.3%) | 0 |
| clm-local-all | webtools_spotify | 0.075 (0.000 to 0.175) | -0.575 (-0.750 to -0.400) | 3 / 51 / 6 | 20/60 (33.3%) | 0 |
| clm-local-all | webtools_tmdb | 0.074 (0.019 to 0.148) | -0.426 (-0.574 to -0.278) | 4 / 36 / 41 | 27/81 (33.3%) | 0 |
| clm-local-all | tooleyes | 0.232 (0.158 to 0.316) | -0.221 (-0.337 to -0.105) | 22 / 87 / 34 | 48/143 (33.6%) | 0 |
| clm-local-all | apibank | 0.109 (0.059 to 0.168) | -0.505 (-0.614 to -0.396) | 11 / 84 / 56 | 50/151 (33.1%) | 0 |
| clm-local-all | metatool_which | 0.200 (0.145 to 0.260) | -0.295 (-0.365 to -0.220) | 40 / 224 / 36 | 100/300 (33.3%) | 0 |
| hybrid@20+clef | pooled | 0.769 (0.721 to 0.817) | 0.224 (0.166 to 0.283) | 223 / 143 / 69 | 145/435 (33.3%) | 0 |
| hybrid@20+clef | all_catalogs | 0.735 (0.694 to 0.771) | 0.210 (0.165 to 0.253) | 360 / 249 / 126 | 245/735 (33.3%) | 0 |
| hybrid@20+clef | webtools_spotify | 0.900 (0.800 to 0.975) | 0.250 (0.075 to 0.425) | 36 / 21 / 3 | 20/60 (33.3%) | 0 |
| hybrid@20+clef | webtools_tmdb | 0.852 (0.759 to 0.944) | 0.352 (0.222 to 0.481) | 46 / 32 / 3 | 27/81 (33.3%) | 0 |
| hybrid@20+clef | tooleyes | 0.653 (0.558 to 0.747) | 0.200 (0.105 to 0.305) | 62 / 40 / 41 | 48/143 (33.6%) | 0 |
| hybrid@20+clef | apibank | 0.782 (0.703 to 0.861) | 0.168 (0.079 to 0.267) | 79 / 50 / 22 | 50/151 (33.1%) | 0 |
| hybrid@20+clef | metatool_which | 0.685 (0.615 to 0.750) | 0.190 (0.125 to 0.260) | 137 / 106 / 57 | 100/300 (33.3%) | 0 |
| clef-all | pooled | 0.797 (0.752 to 0.841) | 0.252 (0.197 to 0.310) | 231 / 142 / 62 | 145/435 (33.3%) | 0 |
| clef-all | all_catalogs | 0.780 (0.743 to 0.814) | 0.255 (0.208 to 0.304) | 382 / 279 / 74 | 245/735 (33.3%) | 0 |
| clef-all | webtools_spotify | 0.950 (0.875 to 1.000) | 0.300 (0.150 to 0.450) | 38 / 21 / 1 | 20/60 (33.3%) | 0 |
| clef-all | webtools_tmdb | 0.852 (0.759 to 0.944) | 0.352 (0.222 to 0.500) | 46 / 34 / 1 | 27/81 (33.3%) | 0 |
| clef-all | tooleyes | 0.632 (0.526 to 0.726) | 0.179 (0.074 to 0.284) | 60 / 41 / 42 | 48/143 (33.6%) | 0 |
| clef-all | apibank | 0.861 (0.792 to 0.931) | 0.248 (0.158 to 0.347) | 87 / 46 / 18 | 50/151 (33.1%) | 0 |
| clef-all | metatool_which | 0.755 (0.695 to 0.810) | 0.260 (0.185 to 0.335) | 151 / 137 / 12 | 100/300 (33.3%) | 0 |
| hybrid@20+clef-flash | pooled | 0.769 (0.721 to 0.817) | 0.224 (0.169 to 0.283) | 223 / 161 / 51 | 145/435 (33.3%) | 0 |
| hybrid@20+clef-flash | all_catalogs | 0.729 (0.688 to 0.769) | 0.204 (0.159 to 0.249) | 357 / 277 / 101 | 245/735 (33.3%) | 0 |
| hybrid@20+clef-flash | webtools_spotify | 0.925 (0.850 to 1.000) | 0.275 (0.100 to 0.450) | 37 / 21 / 2 | 20/60 (33.3%) | 0 |
| hybrid@20+clef-flash | webtools_tmdb | 0.815 (0.704 to 0.907) | 0.315 (0.204 to 0.444) | 44 / 36 / 1 | 27/81 (33.3%) | 0 |
| hybrid@20+clef-flash | tooleyes | 0.600 (0.495 to 0.695) | 0.147 (0.042 to 0.253) | 57 / 51 / 35 | 48/143 (33.6%) | 0 |
| hybrid@20+clef-flash | apibank | 0.842 (0.762 to 0.911) | 0.228 (0.139 to 0.327) | 85 / 53 / 13 | 50/151 (33.1%) | 0 |
| hybrid@20+clef-flash | metatool_which | 0.670 (0.605 to 0.735) | 0.175 (0.110 to 0.245) | 134 / 116 / 50 | 100/300 (33.3%) | 0 |
| clef-flash-all | pooled | 0.786 (0.738 to 0.831) | 0.241 (0.186 to 0.300) | 228 / 173 / 34 | 145/435 (33.3%) | 0 |
| clef-flash-all | all_catalogs | 0.761 (0.722 to 0.800) | 0.237 (0.192 to 0.282) | 373 / 319 / 43 | 245/735 (33.3%) | 0 |
| clef-flash-all | webtools_spotify | 0.875 (0.775 to 0.975) | 0.225 (0.050 to 0.400) | 35 / 25 / 0 | 20/60 (33.3%) | 0 |
| clef-flash-all | webtools_tmdb | 0.833 (0.722 to 0.926) | 0.333 (0.204 to 0.463) | 45 / 35 / 1 | 27/81 (33.3%) | 0 |
| clef-flash-all | tooleyes | 0.611 (0.516 to 0.705) | 0.158 (0.042 to 0.263) | 58 / 57 / 28 | 48/143 (33.6%) | 0 |
| clef-flash-all | apibank | 0.891 (0.832 to 0.950) | 0.277 (0.188 to 0.376) | 90 / 56 / 5 | 50/151 (33.1%) | 0 |
| clef-flash-all | metatool_which | 0.725 (0.660 to 0.785) | 0.230 (0.150 to 0.305) | 145 / 146 / 9 | 100/300 (33.3%) | 0 |
| hybrid@20+laya-wide, in rounds, lower detail | pooled | 0.407 (0.348 to 0.462) | -0.138 (-0.197 to -0.079) | 118 / 125 / 192 | 145/435 (33.3%) | 0 |
| hybrid@20+laya-wide, in rounds, lower detail | all_catalogs | 0.367 (0.324 to 0.408) | -0.157 (-0.204 to -0.108) | 180 / 217 / 338 | 245/735 (33.3%) | 0 |
| hybrid@20+laya-wide, in rounds, lower detail | webtools_spotify | 0.525 (0.350 to 0.675) | -0.125 (-0.275 to 0.050) | 21 / 19 / 20 | 20/60 (33.3%) | 0 |
| hybrid@20+laya-wide, in rounds, lower detail | webtools_tmdb | 0.204 (0.111 to 0.315) | -0.296 (-0.444 to -0.167) | 11 / 27 / 43 | 27/81 (33.3%) | 0 |
| hybrid@20+laya-wide, in rounds, lower detail | tooleyes | 0.295 (0.211 to 0.389) | -0.158 (-0.274 to -0.042) | 28 / 25 / 90 | 48/143 (33.6%) | 0 |
| hybrid@20+laya-wide, in rounds, lower detail | apibank | 0.574 (0.475 to 0.673) | -0.040 (-0.129 to 0.050) | 58 / 54 / 39 | 50/151 (33.1%) | 0 |
| hybrid@20+laya-wide, in rounds, lower detail | metatool_which | 0.310 (0.245 to 0.375) | -0.185 (-0.260 to -0.110) | 62 / 92 / 146 | 100/300 (33.3%) | 0 |
| laya-wide-all, in rounds, lower detail | pooled | 0.348 (0.293 to 0.403) | -0.197 (-0.255 to -0.138) | 101 / 154 / 180 | 145/435 (33.3%) | 0 |
| laya-wide-all, in rounds, lower detail | webtools_spotify | 0.475 (0.325 to 0.625) | -0.175 (-0.325 to -0.025) | 19 / 24 / 17 | 20/60 (33.3%) | 0 |
| laya-wide-all, in rounds, lower detail | webtools_tmdb | 0.167 (0.074 to 0.278) | -0.333 (-0.481 to -0.204) | 9 / 33 / 39 | 27/81 (33.3%) | 0 |
| laya-wide-all, in rounds, lower detail | tooleyes | 0.221 (0.147 to 0.305) | -0.232 (-0.347 to -0.126) | 21 / 33 / 89 | 48/143 (33.6%) | 0 |
| laya-wide-all, in rounds, lower detail | apibank | 0.515 (0.416 to 0.614) | -0.099 (-0.198 to 0.000) | 52 / 64 / 35 | 50/151 (33.1%) | 0 |
| laya-wide-all | metatool_which (not applicable) | n/a | n/a | n/a | n/a | 0 rejected attempts, excluded |
| hybrid@20+rizzo-flow | pooled | 0.697 (0.645 to 0.745) | 0.152 (0.093 to 0.210) | 202 / 167 / 66 | 145/435 (33.3%) | 0 |
| hybrid@20+rizzo-flow | all_catalogs | 0.669 (0.627 to 0.710) | 0.145 (0.100 to 0.190) | 328 / 299 / 108 | 245/735 (33.3%) | 0 |
| hybrid@20+rizzo-flow | webtools_spotify | 0.875 (0.775 to 0.975) | 0.225 (0.074 to 0.400) | 35 / 18 / 7 | 20/60 (33.3%) | 0 |
| hybrid@20+rizzo-flow | webtools_tmdb | 0.704 (0.574 to 0.815) | 0.204 (0.037 to 0.352) | 38 / 39 / 4 | 27/81 (33.3%) | 0 |
| hybrid@20+rizzo-flow | tooleyes | 0.558 (0.463 to 0.663) | 0.105 (0.011 to 0.200) | 53 / 51 / 39 | 48/143 (33.6%) | 0 |
| hybrid@20+rizzo-flow | apibank | 0.752 (0.673 to 0.832) | 0.139 (0.040 to 0.238) | 76 / 59 / 16 | 50/151 (33.1%) | 0 |
| hybrid@20+rizzo-flow | metatool_which | 0.630 (0.565 to 0.695) | 0.135 (0.075 to 0.205) | 126 / 132 / 42 | 100/300 (33.3%) | 0 |
| rizzo-flow-all, in rounds | pooled | 0.690 (0.634 to 0.741) | 0.145 (0.090 to 0.203) | 200 / 162 / 73 | 145/435 (33.3%) | 0 |
| rizzo-flow-all, in rounds | all_catalogs | 0.682 (0.641 to 0.724) | 0.157 (0.108 to 0.204) | 334 / 322 / 79 | 245/735 (33.3%) | 0 |
| rizzo-flow-all, in rounds | webtools_spotify | 0.925 (0.825 to 1.000) | 0.275 (0.125 to 0.425) | 37 / 18 / 5 | 20/60 (33.3%) | 0 |
| rizzo-flow-all, in rounds | webtools_tmdb | 0.815 (0.704 to 0.907) | 0.315 (0.167 to 0.463) | 44 / 35 / 2 | 27/81 (33.3%) | 0 |
| rizzo-flow-all, in rounds | tooleyes | 0.484 (0.389 to 0.579) | 0.032 (-0.074 to 0.137) | 46 / 50 / 47 | 48/143 (33.6%) | 0 |
| rizzo-flow-all, in rounds | apibank | 0.723 (0.634 to 0.802) | 0.109 (0.020 to 0.198) | 73 / 59 / 19 | 50/151 (33.1%) | 0 |
| rizzo-flow-all, in rounds | metatool_which | 0.670 (0.605 to 0.730) | 0.175 (0.095 to 0.255) | 134 / 160 / 6 | 100/300 (33.3%) | 0 |

## Observed provider cost

Replay responses do not contribute usage or timing. Search cost and latency are included for strategies that search. Negatives are priced separately.

| Arm | Catalog | Cold billed | Warm billed / 1,000 | Warm list / 1,000 | Warm cache share | Warm latency p50 / p95 ms | Negative billed / 1,000 |
|---|---|---:|---:|---:|---:|---:|---:|
| hybrid@20 | pooled | $0.0000 | $0.0002 | $0.0002 | n/a | 160.2 / 252.8 | $0.0000 |
| hybrid@20 | all_catalogs | $0.0000 | $0.0004 | $0.0004 | n/a | 162.5 / 253.6 | $0.0000 |
| hybrid@20 | webtools_spotify | $0.0000 | $0.0001 | $0.0001 | n/a | 154.4 / 370.6 | $0.0000 |
| hybrid@20 | webtools_tmdb | $0.0000 | $0.0002 | $0.0002 | n/a | 164.9 / 345.2 | $0.0000 |
| hybrid@20 | tooleyes | $0.0000 | $0.0003 | $0.0003 | n/a | 159.8 / 219.6 | $0.0000 |
| hybrid@20 | apibank | $0.0000 | $0.0003 | $0.0003 | n/a | 160.7 / 221.6 | $0.0000 |
| hybrid@20 | metatool_which | $0.0000 | $0.0005 | $0.0005 | n/a | 166.3 / 239.2 | $0.0000 |
| hybrid@20+jev | pooled | n/a | $0.0828 | $0.0828 | n/a | 449.2 / 630.5 | $0.0833 |
| hybrid@20+jev | all_catalogs | n/a | $0.0675 | $0.0675 | n/a | 451.8 / 629.8 | $0.0687 |
| hybrid@20+jev | webtools_spotify | n/a | $0.0603 | $0.0603 | n/a | 471.1 / 1258.0 | $0.0613 |
| hybrid@20+jev | webtools_tmdb | n/a | $0.0881 | $0.0881 | n/a | 471.4 / 701.8 | $0.0890 |
| hybrid@20+jev | tooleyes | n/a | $0.0822 | $0.0822 | n/a | 432.0 / 575.2 | $0.0827 |
| hybrid@20+jev | apibank | n/a | $0.0883 | $0.0883 | n/a | 453.3 / 597.5 | $0.0883 |
| hybrid@20+jev | metatool_which | n/a | $0.0474 | $0.0474 | n/a | 457.8 / 609.2 | $0.0467 |
| jev-all | pooled | n/a | $0.2807 | $0.2807 | n/a | 323.4 / 439.3 | $0.2752 |
| jev-all | all_catalogs | n/a | $0.2987 | $0.2987 | n/a | 325.1 / 421.8 | $0.2946 |
| jev-all | webtools_spotify | n/a | $0.1091 | $0.1091 | n/a | 306.4 / 461.3 | $0.1026 |
| jev-all | webtools_tmdb | n/a | $0.2004 | $0.2004 | n/a | 314.3 / 349.9 | $0.1929 |
| jev-all | tooleyes | n/a | $0.2958 | $0.2958 | n/a | 309.0 / 362.9 | $0.2933 |
| jev-all | apibank | n/a | $0.3621 | $0.3621 | n/a | 342.3 / 457.8 | $0.3556 |
| jev-all | metatool_which | n/a | $0.3222 | $0.3222 | n/a | 327.2 / 389.1 | $0.3202 |
| agent@20 | pooled | n/a | $0.6804 | $0.6804 | 0.0% | 936.3 / 1999.8 | $0.6934 |
| agent@20 | all_catalogs | n/a | $0.5631 | $0.5631 | 0.0% | 903.9 / 2232.5 | $0.5770 |
| agent@20 | webtools_spotify | n/a | $0.4679 | $0.4679 | 0.0% | 910.2 / 2710.5 | $0.4816 |
| agent@20 | webtools_tmdb | n/a | $0.7636 | $0.7636 | 0.0% | 854.0 / 1610.7 | $0.7769 |
| agent@20 | tooleyes | n/a | $0.6836 | $0.6836 | 0.0% | 951.7 / 2023.5 | $0.6875 |
| agent@20 | apibank | n/a | $0.7074 | $0.7074 | 0.0% | 947.3 / 1834.4 | $0.7279 |
| agent@20 | metatool_which | n/a | $0.4087 | $0.4087 | 0.0% | 863.0 / 2335.3 | $0.4096 |
| agent-all | pooled | n/a | $0.7856 | $2.4286 | 92.1% | 861.0 / 1932.3 | $1.6857 |
| agent-all | webtools_spotify | n/a | $0.3396 | $0.9130 | 86.4% | 689.4 / 846.5 | $0.8201 |
| agent-all | webtools_tmdb | n/a | $0.6362 | $1.8528 | 88.9% | 667.4 / 1069.0 | $1.5467 |
| agent-all | tooleyes | n/a | $0.8512 | $2.5855 | 91.8% | 853.6 / 2239.1 | $1.6383 |
| agent-all | apibank | n/a | $0.9435 | $3.0602 | 93.8% | 999.4 / 1632.8 | $2.0875 |
| agent-all | metatool_which (not applicable) | n/a | n/a | n/a | n/a | n/a | n/a |
| agent-luna@20 | pooled | n/a | $0.1802 | $0.1802 | 0.0% | 1465.3 / 2389.4 | $0.1813 |
| agent-luna@20 | all_catalogs | n/a | $0.1484 | $0.1484 | 0.0% | 1383.6 / 2334.7 | $0.1500 |
| agent-luna@20 | webtools_spotify | n/a | $0.1296 | $0.1296 | 0.0% | 1386.5 / 2470.1 | $0.1312 |
| agent-luna@20 | webtools_tmdb | n/a | $0.2021 | $0.2021 | 0.0% | 1354.9 / 2021.3 | $0.2053 |
| agent-luna@20 | tooleyes | n/a | $0.1820 | $0.1820 | 0.0% | 1648.0 / 2625.4 | $0.1823 |
| agent-luna@20 | apibank | n/a | $0.1845 | $0.1845 | 0.0% | 1391.1 / 2235.6 | $0.1855 |
| agent-luna@20 | metatool_which | n/a | $0.1067 | $0.1067 | 0.0% | 1277.7 / 2205.9 | $0.1055 |
| agent-luna-all | pooled | n/a | $0.0746 | $0.6153 | 99.6% | 1452.5 / 2410.1 | $0.4490 |
| agent-luna-all | webtools_spotify | n/a | $0.0352 | $0.2402 | 99.3% | 1145.6 / 2124.9 | $0.2004 |
| agent-luna-all | webtools_tmdb | n/a | $0.0580 | $0.4745 | 99.6% | 1198.8 / 2581.2 | $0.4384 |
| agent-luna-all | tooleyes | n/a | $0.0826 | $0.6543 | 99.5% | 1668.4 / 2761.9 | $0.4525 |
| agent-luna-all | apibank | n/a | $0.0882 | $0.7706 | 99.7% | 1409.5 / 2089.2 | $0.5338 |
| agent-luna-all | metatool_which (not applicable) | n/a | n/a | n/a | n/a | n/a | n/a |
| hybrid@20+strands | pooled | local | local | local | n/a | 158.7 / 208.1 | local |
| hybrid@20+strands | all_catalogs | local | local | local | n/a | 111.4 / 193.0 | local |
| hybrid@20+strands | webtools_spotify | local | local | local | n/a | 82.2 / 94.1 | local |
| hybrid@20+strands | webtools_tmdb | local | local | local | n/a | 174.8 / 220.4 | local |
| hybrid@20+strands | tooleyes | local | local | local | n/a | 155.5 / 192.7 | local |
| hybrid@20+strands | apibank | local | local | local | n/a | 166.1 / 192.4 | local |
| hybrid@20+strands | metatool_which | local | local | local | n/a | 79.0 / 92.0 | local |
| strands-all, lower detail | pooled | local | local | local | n/a | 116.5 / 219.3 | local |
| strands-all, lower detail | all_catalogs | local | local | local | n/a | 200.3 / 249.1 | local |
| strands-all | webtools_spotify | local | local | local | n/a | 216.2 / 236.2 | local |
| strands-all, lower detail | webtools_tmdb | local | local | local | n/a | 190.5 / 196.5 | local |
| strands-all, lower detail | tooleyes | local | local | local | n/a | 116.6 / 124.7 | local |
| strands-all, lower detail | apibank | local | local | local | n/a | 107.5 / 120.6 | local |
| strands-all, lower detail | metatool_which | local | local | local | n/a | 241.2 / 256.1 | local |
| hybrid@20+clm-local | pooled | local | local | local | n/a | 54.6 / 406.0 | local |
| hybrid@20+clm-local | all_catalogs | local | local | local | n/a | 58.2 / 228.7 | local |
| hybrid@20+clm-local | webtools_spotify | local | local | local | n/a | 48.1 / 312.4 | local |
| hybrid@20+clm-local | webtools_tmdb | local | local | local | n/a | 49.0 / 277.4 | local |
| hybrid@20+clm-local | tooleyes | local | local | local | n/a | 55.1 / 228.3 | local |
| hybrid@20+clm-local | apibank | local | local | local | n/a | 55.7 / 408.9 | local |
| hybrid@20+clm-local | metatool_which | local | local | local | n/a | 62.6 / 174.7 | local |
| clm-local-all | pooled | local | local | local | n/a | 2.0 / 2.4 | local |
| clm-local-all | all_catalogs | local | local | local | n/a | 2.1 / 2.4 | local |
| clm-local-all | webtools_spotify | local | local | local | n/a | 2.0 / 2.4 | local |
| clm-local-all | webtools_tmdb | local | local | local | n/a | 1.9 / 2.6 | local |
| clm-local-all | tooleyes | local | local | local | n/a | 2.0 / 2.2 | local |
| clm-local-all | apibank | local | local | local | n/a | 2.1 / 2.4 | local |
| clm-local-all | metatool_which | local | local | local | n/a | 2.2 / 2.4 | local |
| hybrid@20+clef | pooled | n/a | $0.4418 | $0.4418 | 0.0% | 614.6 / 886.5 | $0.4463 |
| hybrid@20+clef | all_catalogs | n/a | $0.3508 | $0.3508 | 0.0% | 577.7 / 877.5 | $0.3599 |
| hybrid@20+clef | webtools_spotify | n/a | $0.3193 | $0.3193 | 0.0% | 510.1 / 773.9 | $0.3257 |
| hybrid@20+clef | webtools_tmdb | n/a | $0.4798 | $0.4798 | 0.0% | 673.8 / 1133.3 | $0.4857 |
| hybrid@20+clef | tooleyes | n/a | $0.4368 | $0.4368 | 0.0% | 616.7 / 882.2 | $0.4418 |
| hybrid@20+clef | apibank | n/a | $0.4686 | $0.4686 | 0.0% | 617.6 / 803.6 | $0.4708 |
| hybrid@20+clef | metatool_which | n/a | $0.2312 | $0.2312 | 0.0% | 505.9 / 815.2 | $0.2303 |
| clef-all | pooled | n/a | $1.6271 | $1.6271 | 0.0% | 1428.4 / 1783.6 | $1.5935 |
| clef-all | all_catalogs | n/a | $1.7580 | $1.7580 | 0.0% | 1517.7 / 1817.2 | $1.7337 |
| clef-all | webtools_spotify | n/a | $0.6221 | $0.6221 | 0.0% | 726.9 / 922.0 | $0.5824 |
| clef-all | webtools_tmdb | n/a | $1.1672 | $1.1672 | 0.0% | 1073.3 / 1362.8 | $1.1211 |
| clef-all | tooleyes | n/a | $1.7275 | $1.7275 | 0.0% | 1403.7 / 1573.4 | $1.7117 |
| clef-all | apibank | n/a | $2.0871 | $2.0871 | 0.0% | 1656.1 / 1911.2 | $2.0487 |
| clef-all | metatool_which | n/a | $1.9303 | $1.9303 | 0.0% | 1580.1 / 1905.2 | $1.9181 |
| hybrid@20+clef-flash | pooled | n/a | $0.1657 | $0.1657 | 0.0% | 465.0 / 697.7 | $0.1674 |
| hybrid@20+clef-flash | all_catalogs | n/a | $0.1316 | $0.1316 | 0.0% | 412.0 / 684.1 | $0.1350 |
| hybrid@20+clef-flash | webtools_spotify | n/a | $0.1197 | $0.1197 | 0.0% | 447.3 / 550.6 | $0.1222 |
| hybrid@20+clef-flash | webtools_tmdb | n/a | $0.1799 | $0.1799 | 0.0% | 428.4 / 710.5 | $0.1822 |
| hybrid@20+clef-flash | tooleyes | n/a | $0.1638 | $0.1638 | 0.0% | 487.0 / 697.1 | $0.1657 |
| hybrid@20+clef-flash | apibank | n/a | $0.1757 | $0.1757 | 0.0% | 466.4 / 662.2 | $0.1766 |
| hybrid@20+clef-flash | metatool_which | n/a | $0.0867 | $0.0867 | 0.0% | 355.7 / 611.9 | $0.0864 |
| clef-flash-all | pooled | n/a | $0.6102 | $0.6102 | 0.0% | 685.1 / 1014.7 | $0.5976 |
| clef-flash-all | all_catalogs | n/a | $0.6593 | $0.6593 | 0.0% | 735.0 / 1038.9 | $0.6501 |
| clef-flash-all | webtools_spotify | n/a | $0.2333 | $0.2333 | 0.0% | 519.6 / 777.2 | $0.2184 |
| clef-flash-all | webtools_tmdb | n/a | $0.4377 | $0.4377 | 0.0% | 500.1 / 819.2 | $0.4204 |
| clef-flash-all | tooleyes | n/a | $0.6478 | $0.6478 | 0.0% | 712.3 / 974.1 | $0.6419 |
| clef-flash-all | apibank | n/a | $0.7827 | $0.7827 | 0.0% | 761.6 / 1021.3 | $0.7682 |
| clef-flash-all | metatool_which | n/a | $0.7239 | $0.7239 | 0.0% | 812.8 / 1063.3 | $0.7193 |
| hybrid@20+laya-wide, in rounds, lower detail | pooled | local | local | local | n/a | 275.6 / 324.9 | local |
| hybrid@20+laya-wide, in rounds, lower detail | all_catalogs | local | local | local | n/a | 276.6 / 320.0 | local |
| hybrid@20+laya-wide, in rounds, lower detail | webtools_spotify | local | local | local | n/a | 283.8 / 332.4 | local |
| hybrid@20+laya-wide, in rounds, lower detail | webtools_tmdb | local | local | local | n/a | 285.2 / 310.7 | local |
| hybrid@20+laya-wide, in rounds, lower detail | tooleyes | local | local | local | n/a | 267.7 / 339.3 | local |
| hybrid@20+laya-wide, in rounds, lower detail | apibank | local | local | local | n/a | 273.4 / 316.3 | local |
| hybrid@20+laya-wide, in rounds, lower detail | metatool_which | local | local | local | n/a | 277.9 / 316.2 | local |
| laya-wide-all, in rounds, lower detail | pooled | local | local | local | n/a | 303.1 / 401.0 | local |
| laya-wide-all, in rounds, lower detail | webtools_spotify | local | local | local | n/a | 166.2 / 171.7 | local |
| laya-wide-all, in rounds, lower detail | webtools_tmdb | local | local | local | n/a | 204.2 / 209.9 | local |
| laya-wide-all, in rounds, lower detail | tooleyes | local | local | local | n/a | 301.0 / 321.2 | local |
| laya-wide-all, in rounds, lower detail | apibank | local | local | local | n/a | 388.4 / 409.3 | local |
| laya-wide-all | metatool_which (not applicable) | n/a | n/a | n/a | n/a | n/a | n/a |
| hybrid@20+rizzo-flow | pooled | local | local | local | n/a | 800.8 / 990.6 | local |
| hybrid@20+rizzo-flow | all_catalogs | local | local | local | n/a | 683.5 / 957.4 | local |
| hybrid@20+rizzo-flow | webtools_spotify | local | local | local | n/a | 611.1 / 948.6 | local |
| hybrid@20+rizzo-flow | webtools_tmdb | local | local | local | n/a | 832.3 / 1075.3 | local |
| hybrid@20+rizzo-flow | tooleyes | local | local | local | n/a | 797.4 / 942.4 | local |
| hybrid@20+rizzo-flow | apibank | local | local | local | n/a | 818.0 / 1013.3 | local |
| hybrid@20+rizzo-flow | metatool_which | local | local | local | n/a | 490.9 / 618.8 | local |
| rizzo-flow-all, in rounds | pooled | local | local | local | n/a | 2945.1 / 3376.3 | local |
| rizzo-flow-all, in rounds | all_catalogs | local | local | local | n/a | 3014.1 / 3361.4 | local |
| rizzo-flow-all, in rounds | webtools_spotify | local | local | local | n/a | 989.1 / 1034.6 | local |
| rizzo-flow-all, in rounds | webtools_tmdb | local | local | local | n/a | 1937.5 / 2221.5 | local |
| rizzo-flow-all, in rounds | tooleyes | local | local | local | n/a | 2931.0 / 3113.9 | local |
| rizzo-flow-all, in rounds | apibank | local | local | local | n/a | 3309.0 / 3403.1 | local |
| rizzo-flow-all, in rounds | metatool_which | local | local | local | n/a | 3032.0 / 3327.6 | local |

## Non-applicable pairs and historical rejections

- agent-all / metatool_which: OpenAI Chat Completions rejected 200 function tools: HTTP 400 array_above_max_length, param tools (4/4 attempts); exact maximum not established. Evidence: bench/runs/20260929T221520Z-direct-pilot/run.jsonl (2026-09-30); 0 historical requests excluded, 0 rejected attempts.
- agent-luna-all / metatool_which: OpenAI Chat Completions rejected 200 function tools for gpt-6-luna: HTTP 400 array_above_max_length, param tools (4/4 attempts). Evidence: bench/runs/20260930T165646Z-direct-luna-pilot/run.jsonl (2026-09-30); 0 historical requests excluded, 0 rejected attempts.
- laya-wide-all / metatool_which: 198 candidates do not fit two rounds for 'laya 0.3.27 serve.py (100 options, 64 questions) and common.py build_head/render_options (48-token option cap); convaiinnovations/laya @ 55cf4c4 rl_agent_config.json (max_len 512, head_max_len 192); sent with every request: head_max_len 512, max_len 1024': even one finalist from each of the 13 groups does not fit the final question. Evidence: planner (2026-10-05); 0 historical requests excluded, 0 rejected attempts.

| Arm | Catalog | Historical phase | Rejected provider attempts | Billed | List | Billed / 1,000 | List / 1,000 |
|---|---|---|---:|---:|---:|---:|---:|

## Caveats

- The metric is selecting a relevant tool, not completing a task; tasks can list several relevant tools.
- Negatives remove the task's labeled relevant tools; an unlabeled alternative may still serve it.
- Pooled none-option metrics state their observed positive/negative mix; errors are reported separately.
- Replay outcomes are scored, but their historical usage, latency and cache reads are excluded.
- Search overhead is attributed to each strategy that searches; physical search calls are paid once.
- Cold is the first scored positive; a replay there provides no new cold measurement.
- Caching reflects one provider's routing in one single-turn run and does not isolate a latency effect.
- **CLM (Mac bf16 MPS).** We run CLM on a Mac (transformers, bf16, MPS) instead of vLLM on CUDA. Its parity with our Modal deployment was checked on what reports publish, P@1 per cell and the answer-or-abstain decision (`bench/results/2026-10-toolret-decision-p2/clm-parity.json`); its figures stay provisional until the CLM authors confirm parity. `clm-serve` keeps the vectors of the texts it has embedded (its action cache, on by default), and our arms ask the same requests and cards more than once, so its latency here is mostly a warm-cache latency.
- **Clef.** Workers AI serves the current Clef behind `@cf/cloudflare/clef`, and no version can be pinned: a later run may answer differently from the one reported here.
- **Clef-flash.** Workers AI serves the current Clef-flash behind `@cf/cloudflare/clef-flash`, and no version can be pinned: a later run may answer differently from the one reported here.
- **Laya wide.** Its authors describe the base checkpoints as a fast base to specialise, not a zero-shot decision engine; we run the English one zero-shot. Its questions are planned with Laya's own tokenizer, and an option whose name would repeat its key past 48 tokens is sent as its key alone. It runs with a 512-token option budget in a 1,024-token window, beyond the 192 and 512 the checkpoint ships with, as its model card advises for many options.
- **rizzo-flow.** The authors' fine-tune, 4B at q8_0 on llama.cpp with Metal; quantization and hardware change its probabilities, by its authors' account.
- **Local deciders.** Strands Decider 2B and CLM (Mac bf16 MPS) and Laya wide and rizzo-flow ran on one machine (Apple M5 Max, 128 GB, macOS 26.6.2): their cost reads “local”, and their latency is that machine's, not comparable like for like with a hosted API's.
- **Decision time is the sum of calls.** A direct record's decision time adds up the seconds of its calls: first-round questions that a decider is asked at the same time are added, not overlapped, so its searches asked in two rounds read slower here than on their critical path.
- **Search time differs between runs.** Median search of the searching arms: `20260930T050455Z-direct` 150 ms, `20261004T192417Z-direct-strands` 3 ms, `20261004T195347Z-direct-clm-local` 3 ms, `20261004T202207Z-direct-clef` 5 ms, `20261004T210405Z-direct-clef-flash` 5 ms, `20261005T142646Z-direct-luna` 171 ms, `20261005T144321Z-direct-laya-wide` 156 ms, `20261005T171409Z-direct-rizzo-flow` 195 ms. A run that found its query embeddings already cached searched without an embedding call: compare the latency of searching arms across runs only after this difference.
- **agent-luna@20 measured again.** Its rows come from `20261005T142646Z-direct-luna`, which replaced the records and calls of `20260930T170103Z-direct-luna` for this arm.
- **Key-only options and failed searches.** A card whose name would take its option past the model's per-option window is sent as its key alone, the name itself. A search with a card no question can show fails: it counts as an error and is left out of the relevant-pick rate. A search that failed otherwise, at the provider or once round one was asked, counts as an error and a miss.
- hybrid@20+laya-wide, webtools_tmdb: 3.5% of the card options sent as their key alone.
- laya-wide-all, webtools_tmdb: 3.3% of the card options sent as their key alone.

## Reproduce

```shell
uv run toolhunch-bench direct --dry-run
uv run toolhunch-bench direct --pilot
uv run toolhunch-bench direct
uv run toolhunch-bench direct-report bench/runs/RUN --out bench/results/2026-09-toolret-direct/
```

Verified provider usage in this run: $1.0094.
Guarded charges, including all failed/excluded attempts: $1.0094; 0 attempts have unknown usage.
Projected remaining spend: $0.0000; cumulative P1 including historical prior and all guarded charges: $1.3166.
Method: completed full run.
Separate recorded uncached budget reference for remaining workload: $0.0000. Recorded full uncached/max-output estimate retained as a conservative reference for any unfinished workload, without subtracting successful request costs. Historical retrieval approximations and token framing prevent a guaranteed upper bound; physical-call guard reservations remain independent. This report does not authorize another run.
