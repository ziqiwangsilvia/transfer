error[invalid-argument-type]: Argument to bound method `_filter_nlp_samples` is incorrect
  --> evaluator/conversational_content/content_evaluator.py:85:22
   |
84 |         nlp_records, nlp_outputs, nlp_predictions = self._filter_nlp_samples(
85 |             records, outputs, predictions
   |                      ^^^^^^^ Expected `list[dict[str, Any]]`, found `Any | list[None | Unknown]`
86 |         )
87 |         log.info(f"{len(nlp_records)} NLP samples (of {len(records)} total)")
   |
info: Element `list[None | Unknown]` of this union is not assignable to `list[dict[str, Any]]`
info: Method defined here
   --> evaluator/conversational_content/content_evaluator.py:126:9
    |
124 |         }
125 |
126 |     def _filter_nlp_samples(
    |         ^^^^^^^^^^^^^^^^^^^
127 |         self,
128 |         records: List[Dict[str, Any]],
129 |         outputs: List[Dict[str, Any]],
    |         ----------------------------- Parameter declared here
130 |         predictions: List[Dict[str, Any]],
131 |     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    |
info: rule `invalid-argument-type` is enabled by default

Found 1 diagnostic