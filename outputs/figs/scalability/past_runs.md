# Optimizer runtime from past runs

Wall time per solve; optimizer $/month assumes one i4i.2xlarge ($0.686/h) and daily reconfiguration.

| pipeline | RSMs | replicas | nodes used | solves | wall s (median / max) | optimizer $/mo | savings $/mo (median) |
|---|---|---|---|---|---|---|---|
| fleet ILP | 50 | 150-350 | 30-86 | 43 | 89 / 1243 | 0.51 | 9427 |
| type-count ILP (exec-02) | 50 | 150-150 | 30-30 | 326 | 4 / 11 | 0.02 | 6075 |
| type-count ILP (exec-02) | 10000 | 30000-50000 | 60-100 | 6 | 27 / 35 | 0.15 | 10995 |
