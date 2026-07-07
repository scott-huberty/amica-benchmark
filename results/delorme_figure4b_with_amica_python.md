# Delorme Figure 4B Data With Py-EM and Py-DAAREM

Source workdir: `/Users/scotterik/devel/projects/amica-python/amica-benchmark/benchmark_runs/mica_release_amica_python_matlab`

Reduction mirrors `plotresults.m`: datasets `[1:9, 11:14]`, `mir * 1.4427 * 250 / 1000`, and `count(allrv < 0.05) / (13 * 71) * 100`.

## Regression

| Point set | R^2 | p-value | slope | intercept |
| --- | ---: | ---: | ---: | ---: |
| Original 18 algorithms | 0.964111 | 5.493123e-13 | 20.388759 | -852.098986 |
| Original + Python algorithms | 0.961888 | 3.201771e-14 | 21.519381 | -899.968097 |

## Coordinates

| Rank by MIR | Algorithm | MIR (kbits/s) | RV < 5% components (%) | Count |
| ---: | --- | ---: | ---: | ---: |
| 1 | Amica | 43.131985 | 30.010834 | 277 |
| 2 | Py-EM | 43.110919 | 29.144095 | 269 |
| 3 | Py-DAAREM | 43.107553 | 30.877573 | 285 |
| 4 | Infomax | 43.066338 | 25.352113 | 234 |
| 5 | Ext. Infomax | 43.022236 | 25.243770 | 233 |
| 6 | Pearson | 43.007867 | 25.893824 | 239 |
| 7 | SHIBBS | 42.743488 | 18.959913 | 175 |
| 8 | JADE | 42.736411 | 18.418202 | 170 |
| 9 | FastICA | 42.713636 | 20.151679 | 186 |
| 10 | TICA | 42.684173 | 17.226436 | 159 |
| 11 | JADE opt. | 42.642132 | 14.842904 | 137 |
| 12 | SOBI | 42.513346 | 12.459372 | 115 |
| 13 | JADE-TD | 42.473200 | 13.434453 | 124 |
| 14 | SOBIRO | 42.440866 | 13.434453 | 124 |
| 15 | FOBI | 42.314889 | 10.725894 | 99 |
| 16 | EVD24 | 42.299731 | 10.400867 | 96 |
| 17 | EVD | 42.186115 | 9.642470 | 89 |
| 18 | icaMS | 42.183455 | 7.475623 | 69 |
| 19 | AMUSE | 42.135959 | 5.742145 | 53 |
| 20 | PCA | 41.861102 | 3.791983 | 35 |
