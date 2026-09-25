# Gemma answers on the golden set

`gemma3:12b` wrote 50 answers. `gpt-oss:20b` scored them. Means are over the questions that returned a score. Groundedness is 49 questions because gold-030 was cut off.

Source: `data/generation/answers.json` and `data/generation/evals.json`, 24 Sep 2026.

| Metric | Value |
|---|---:|
| Mean correctness | 0.878 |
| Mean faithfulness | 0.932 |
| Mean groundedness | 0.902 |
| Perfect on every check | 33 / 50 |
| Mean generation latency | 21.19 s |

Rounded, those means are 0.88, 0.93, and 0.90, and the latency is 21.2 s. A perfect question scores 1 on correctness, faithfulness, and groundedness, and passes both deterministic checks.

## Deterministic scores

Computed from `answers.json` and the golden file by `token_f1`, `phrase_coverage`, `citation_recall`, `distractor_citation`, and `abstain_contamination` in `src/eval_generation.py`. No judge call. A score that does not apply is left out of its mean. Token F1, phrase coverage, and citation recall skip the 3 abstain questions.

| Score | Value | Questions |
|---|---:|---:|
| Token F1 | 0.420 | 47 |
| Phrase coverage | 0.973 | 47 |
| Citation recall | 0.915 | 47 |
| Distractor citation | 0.007 | 50 |
| Abstain contamination | 0.667 | 3 |
| Confidence when the hard checks pass | 0.972 | 41 |
| Confidence when a hard check fails | 0.733 | 9 |

Token F1 is word overlap with `expected_answer`. Gold answers are short and model answers are sentences, so precision is 0.301 while word recall is 0.879. The fact is usually present. The extra words pull F1 down to 0.420.

Phrase coverage is the share of `answer_must_include` phrases, so a partial answer is not a flat fail. Citation recall is the share of gold `section_id`s the answer names. gold-007 and gold-009 score 0: the numbers match and the cited file is the archived plan. gold-022 scores 0.5. It names `Carbon_New_2040#6` and not the environmental-policy section. It also names `Carbon_Old_2050#9`, which is not the gold distractor, so distractor citation stays 0 on that row.

Distractor citation is above 0 on one answer. gold-049 declines in the text and still lists `Envi_2040-1#13`, a distractor, among three cited sections.

Abstain contamination is 2 of 3. gold-048 adds the UK Scope 2 figure. gold-050 names the ESG Committee. gold-049's text stays a decline.

A hard check fails when the phrase check fails, the section check fails, phrase coverage or citation recall is below 1, a cited section is a distractor, or an abstain answer adds a figure or a name. The 9 failures are gold-007, gold-009, gold-018, gold-022, gold-029, gold-044, gold-048, gold-049, and gold-050. Two of them, gold-018 and gold-049, report confidence 0.0. The other seven still sit at 0.9 or above.

## Judge score by question type

Mean of faithfulness, groundedness, and correctness. A missing score is left out of that question's average. gold-030 is the only missing score, on groundedness, in the numeric group.

| Question type | Questions | Mean GEval |
|---|---:|---:|
| Unanswerable | 3 | 0.500 |
| Conflict | 1 | 0.500 |
| Comparison | 2 | 0.850 |
| Yes/no | 3 | 0.867 |
| Numeric | 11 | 0.894 |
| Factual | 11 | 0.930 |
| Cross-document | 2 | 0.967 |
| List | 7 | 0.990 |
| Definition | 1 | 1.000 |
| Multi-section | 1 | 1.000 |
| Temporal | 6 | 1.000 |
| Version conflict | 2 | 1.000 |

## Two correct answers were marked 0.5

gold-005 answers "Version 1.0". The fact is in the answer. The judge docked it for not repeating the instruction line "The answer must include: 1.0." that was added to the expected output.

gold-014 answers "over 100%". The gold fact is in the answer. The judge scored 0.5 because it did not count that phrase as the separate figure "100%", and because the answer added growth context the expected output did not ask for.

## Questions that are not clean

Phrase and section columns are the deterministic checks. Source: `answers.json` joined to `evals.json` and the golden set.

| Question | What happened | Correctness | Phrases | Section |
|---|---|---:|---|---|
| gold-018 | Said the EV fleet targets for 2028 and 2032 are not in the documents. They are in `Carbon_New_2040#6`. | 0 | Fail | Fail |
| gold-050 | Answered who may amend the water policy, not who approved it. | 0 | Pass | Fail |
| gold-048 | Declined the US figure, then added a UK Scope 2 number. | 0 | Pass | Fail |
| gold-044 | Reported the quarterly water report and missed the half-year one. | 0.4 | Fail | Pass |
| gold-022 | Gave the current plan ~25% green energy by 2030. Gold is about 50%. | 0.6 | Pass | Pass |
| gold-029 | Right scope (rented, leased, owned) without the word "No". | 0.7 | Fail | Pass |
| gold-006 | Right superseded facts, without the word "No". | 0.8 | Pass | Pass |
| gold-020 | Right years, 2040 and 2050, without the RE100 commitment. | 0.8 | Pass | Pass |
| gold-041 | Includes "technically feasible" without the word "No". | 0.8 | Pass | Pass |
| gold-043 | Names both water-risk tools, joined with "or" instead of "and". | 0.8 | Pass | Pass |
| gold-007 | Right FY24 India totals, cited the archived plan instead of `Carbon_New_2040#2`. | 1.0 | Pass | Fail |
| gold-009 | Right UK Scope 1 facts, cited the archived plan instead of `Carbon_New_2040#3`. | 1.0 | Pass | Fail |
| gold-005 | Answer is "Version 1.0". Judge wanted the instruction text in the answer. | 0.5 | Pass | Pass |
| gold-030 | Answer scored 1.0. Groundedness JSON was cut off at `done_reason=length`. | 1.0 | Pass | Pass |
| gold-049 | Declined, which is the right reply. Faithfulness and groundedness are 0, and no supporting section is cited. | 1.0 | Pass | Fail |

## What the numbers say

Temporal, list, and version-conflict questions are essentially solved. The model names the section it used on 44 of 50 questions, and 47 answers contain every required phrase.

The three unanswerable questions are the weak set. One decline is right (gold-049, confidence 0.0). The other two still assert a nearby fact. Confidence is otherwise stuck at 0.9 or above on 48 answers, including those two wrong ones. The other confidence of 0.0 is gold-018, which declined a question the documents do answer.

Phrase match does not catch a wrong document version. gold-007 and gold-009 copy the right numbers from the superseded carbon plan. The section check is what marks them.

## Diagnosis: 2040 versus 2050 source data

The flawed green-energy answer is gold-022. Retrieval returned the current section. The wrong figure is in the superseded plan, which is still in the corpus.

**Question.** What green-energy shares for 2025 and 2030 does the current Carbon Reduction Plan state?

**Flawed answer.** 10% by 2025 and about 25% by 2030.

**Was it retrieval?** No. `Carbon_New_2040#6` was retrieved. It states 10% by 2025 and about 50% by 2030.

**Was it the source?** Yes. `Carbon_Old_2050.md` is still in the corpus, marked superseded (version 1.0, 12 March 2023, document ID `CARBON-POL-2023-V1.0`). The same section name states about 25% by 2030. The generator joined the current 10% to the archived 25%.
