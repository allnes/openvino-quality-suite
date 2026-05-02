from oviqs.domain.metrics.long_context import lost_in_middle_score_from_ppl

print(lost_in_middle_score_from_ppl({"0_10": 6.7, "30_50": 12.8, "50_70": 13.5, "90_100": 6.9}))
