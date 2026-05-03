# Delorme Figure 4B Data With AMICA-Python

Source workdir: `/Users/scotterik/devel/projects/amica-python/amica-benchmark/benchmark_runs/mica_release_amica_python_matlab`

Reduction mirrors `plotresults.m`: datasets `[1:9, 11:14]`, `mir * 1.4427 * 250 / 1000`, and `count(allrv < 0.05) / (13 * 71) * 100`.

## Regression

| Point set | R^2 | p-value | slope | intercept |
| --- | ---: | ---: | ---: | ---: |
| Original 18 algorithms | 0.964111 | 5.493123e-13 | 20.388759 | -852.098986 |
| Original + AMICA-Python | 0.965413 | 7.377487e-14 | 20.882585 | -873.007397 |

## Coordinates

| Rank by MIR | Algorithm | MIR (kbits/s) | RV < 5% components (%) | Count |
| ---: | --- | ---: | ---: | ---: |
| 1 | Amica | 43.131985 | 30.010834 | 277 |
| 2 | AMICA-Python | 43.110706 | 29.252438 | 270 |
| 3 | Infomax | 43.066338 | 25.352113 | 234 |
| 4 | Ext. Infomax | 43.022236 | 25.243770 | 233 |
| 5 | Pearson | 43.007867 | 25.893824 | 239 |
| 6 | SHIBBS | 42.743488 | 18.959913 | 175 |
| 7 | JADE | 42.736411 | 18.418202 | 170 |
| 8 | FastICA | 42.713636 | 20.151679 | 186 |
| 9 | TICA | 42.684173 | 17.226436 | 159 |
| 10 | JADE opt. | 42.642132 | 14.842904 | 137 |
| 11 | SOBI | 42.513346 | 12.459372 | 115 |
| 12 | JADE-TD | 42.473200 | 13.434453 | 124 |
| 13 | SOBIRO | 42.440866 | 13.434453 | 124 |
| 14 | FOBI | 42.314889 | 10.725894 | 99 |
| 15 | EVD24 | 42.299731 | 10.400867 | 96 |
| 16 | EVD | 42.186115 | 9.642470 | 89 |
| 17 | icaMS | 42.183455 | 7.475623 | 69 |
| 18 | AMUSE | 42.135959 | 5.742145 | 53 |
| 19 | PCA | 41.861102 | 3.791983 | 35 |
