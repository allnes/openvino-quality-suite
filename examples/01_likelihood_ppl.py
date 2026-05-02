from oviqs.adapters.runners.dummy import DummyLogitsRunner
from oviqs.domain.metrics.likelihood import nll_ppl_from_logits

runner = DummyLogitsRunner()
encoded = {"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1, 1, 1, 1]]}
logits = runner.forward_logits(encoded["input_ids"], encoded["attention_mask"])
print(nll_ppl_from_logits(logits, encoded["input_ids"], encoded["attention_mask"]))
