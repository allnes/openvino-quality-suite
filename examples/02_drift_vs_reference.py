from oviqs.adapters.runners.dummy import DummyLogitsRunner
from oviqs.domain.metrics.distribution_drift import aggregate_drift, distribution_drift

ids = [[1, 2, 3, 4]]
ref = DummyLogitsRunner(correct_bias=5.0).forward_logits(ids)[:, :-1, :]
cur = DummyLogitsRunner(correct_bias=4.8).forward_logits(ids)[:, :-1, :]
print(aggregate_drift(distribution_drift(ref, cur)))
