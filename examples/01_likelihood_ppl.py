from oviqs.metrics.likelihood import nll_ppl_from_logits
from oviqs.runners.dummy import DummyLogitsRunner

runner = DummyLogitsRunner()
encoded = {"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1, 1, 1, 1]]}
logits = runner.forward_logits(encoded["input_ids"], encoded["attention_mask"])
print(nll_ppl_from_logits(logits, encoded["input_ids"], encoded["attention_mask"]))
