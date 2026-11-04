::: {custom-style="Capa"}
CENTRO UNIVERSITÁRIO ATENEU

COMPUTER ENGINEERING
:::

&nbsp;

::: {custom-style="Capa"}
Kelvin e Silva Marques

Dimitri Farias Fernandes

Hastrorllabyo da Silva Lima

Keven dos Santos Xavier

Caio de Freitas Brandão

Kauan Magalhães Costa

Guilherme Martins Bezerra

Pedro Henrique Gomes Ferreira

Carlos Eduardo Rodrigues Ferreira

Luan dos Santos Gaspar

Francisco Glauber Rodrigues dos Santos
:::

::: {custom-style="Capa titulo"}
Chain-of-thought obfuscation under supervision
:::

::: {custom-style="Capa subtitulo"}
reanalysis of a pilot study and competence screening of a preregistered prospective study
:::

::: {custom-style="Capa local"}
FORTALEZA, BRAZIL\
2026
:::

{{quebra}}

# ABSTRACT

Chain-of-thought monitorability depends on the textual traces produced by language models continuing to reveal undesired behavior when training is subject to supervision pressure. This work examines three bodies of evidence from a single project. A reanalysis of a supervised fine-tuning (SFT) pilot study reproduced the published results but showed that the grader accepted ambiguous answers and that the seeds shared almost all items, which prevents attributing the difference between training targets to obfuscation. The preregistered study v3.2, with SFT and reward-optimization (GRPO) arms, was closed without training because the base models did not reach 65% accuracy in the competence screening. Reproducing it with the frozen code yielded exactly the registered values and ruled out execution defects; the causes were the models' inability at the hardest levels and instructions that made the measurement depend on undemonstrated formats. A new protocol, v4, recalibrated tasks and instructions on separate items, and the SFT base model answered {{v:v4.gate.sft.A.en}} and {{v:v4.gate.sft.C.en}} of the items at the two levels correctly in the screening run before registration. None of the nine models evaluated as monitors distinguished traces that use forbidden information from traces that do not, and the GRPO arm was excluded under a rule declared before this result. There is no confirmatory result about obfuscation: the work documents a viable design for testing the effect of supervision format and the instrumentation requirements for testing obfuscation under reward pressure.

**Keywords:** Chain of thought. Monitorability. Preregistration. Competence screening. Reproducibility.

{{quebra}}

{{sumario}}

{{quebra}}

# 1 INTRODUCTION

Language models that produce textual traces before the final answer offer an additional source of evidence about their behavior. In this text, a trace is the intermediate text generated before the answer, and a monitor is a procedure, automatic or human, that reads this text to detect undesired behavior. Korbak et al. (2025) describe this chain-of-thought (CoT) monitorability as a relevant but fragile opportunity for safety, because training decisions can reduce it. The central concern is obfuscation: an undesired behavior is maintained while the textual signals that would allow it to be detected are reduced.

This article examines three bodies of evidence produced in the same project. The first is a pilot study with supervised fine-tuning (SFT), whose published data were reanalyzed. The second is a prospective study, called v3.2, that is, a study whose hypotheses, procedures and analysis criteria were fixed and publicly registered before data collection. It had two arms, understood as sets of conditions that share the same training procedure: SFT and reward optimization with Group Relative Policy Optimization (GRPO). The third is a new protocol, v4, prepared after v3.2 did not satisfy its competence criterion.

That criterion was a screening: before the 240 training runs, each base model, that is, the pretrained model not yet fine-tuned in the study, had to answer at least 65% of 120 items from the development (DEV) partition correctly at each difficulty level. The DEV partition is reserved for preliminary checks and is distinct from the test (TEST) partition, used only for the final outcome. The screening recorded 57.5% and 0% at the levels of the SFT arm and 43.33% and 15% at those of the GRPO arm. Before attributing this result to the models' inability, defects that would produce the same pattern must be ruled out: grader error (the program that decides whether an answer is right), incorrect use of the chat template (the format in which messages are presented to the model), truncation, wrong answer extraction, numerical precision or library incompatibility.

Three questions guide the work. Q1: do the published pilot data allow attributing the difference in performance between training targets to the obfuscation of necessary reasoning? Q2: does the failure of the v3.2 screening stem from a technical defect or from the inability of the base models? Q3: do changes calibrated on separate data, kept apart from the study partitions, allow the same competence criterion to be met, and do the planned monitors distinguish traces that use the forbidden information from traces that do not? No confirmatory training was performed; the registered hypotheses are described but not tested.

# 2 BACKGROUND

Faithfulness refers to the correspondence between the trace and the factors that actually determine the answer; monitorability, to the possibility of detecting, from the trace, behaviors relevant to oversight. Wei et al. (2022) showed that intermediate steps improve performance on multi-step problems. Turpin et al. (2023) demonstrated, however, that plausible explanations can omit influences that change the answer, and Lanham et al. (2023) measured, through interventions on the trace, how much the answer depends on it. Faithfulness is therefore a property to be measured, not presumed from the fluency of the text. To infer obfuscation, one must observe at the same time a behavior defined independently of the text and the difficulty of detecting it in the trace.

Baker et al. (2025) observed that penalizing, in reinforcement learning, the traces that mentioned exploiting tests reduced detection without eliminating the behavior. Drori et al. (2025) and Mitrani Hadida et al. (2026) report that supervision applied only to the output also changes the traces and that the effect can generalize to unseen tasks. Emmons et al. (2025), on the other hand, show that, when the trace is computationally necessary, evading the monitor requires substantial assistance or optimization. The effect of supervision pressure thus depends on the task, the training mechanism and the monitor, which makes controlling the model's competence and a behavioral definition of exploitation essential.

On the methodological side, Agarwal et al. (2021) recommend making explicit the unit of replication and the uncertainty across runs; Zheng et al. (2023) document biases of evaluators based on language models, which motivates monitors independent of the evaluated model and validation with human labels; and Nosek et al. (2018) describe preregistration as the separation between decisions made before and after observing the data. GRPO (SHAO et al., 2024) generates several responses per prompt and computes the advantage of each, that is, how much its reward exceeds the group mean. Both arms use low-rank adaptation (LoRA) (HU et al., 2022), which adds trainable low-rank matrices to the model's layers and keeps the original weights fixed, on Qwen2.5 (QWEN TEAM, 2024) and Qwen3 (YANG et al., 2025) models, with the Transformers (WOLF et al., 2020) and TRL (VON WERRA et al., 2020) libraries.

# 3 HYPOTHESES AND CATEGORIES OF EVIDENCE

The v3.2 registration specified two hypotheses, whose text v4 keeps; in v4, only H1 remains testable (section 5.3). Each training run is identified by a seed, the number that initializes the pseudorandom generators. Hypothesis H1, of the SFT arm, predicts that training direct answers, instead of explicit traces, costs more accuracy at the higher-complexity level (C) than at the lower-complexity level (A). For each seed *s*,

$$\Delta^{\text{SFT}}_s = \left[\text{Acc}_{s}(\text{direct}, C) - \text{Acc}_{s}(\text{explicit}, C)\right] - \left[\text{Acc}_{s}(\text{direct}, A) - \text{Acc}_{s}(\text{explicit}, A)\right],$$

with a predicted negative direction. The contrast measures the effect of the training target on accuracy, not an intention to conceal.

In the GRPO arm, each multiple-choice question carries a forbidden auxiliary note, also called a hint, that points to a wrong option; the misspecified reward, a score that intentionally pays for undesired behavior, rewards that option. There are three conditions: *none*, with no monitor in the reward; *cot_monitor*, in which the reward is reduced when a reward monitor detects use of the note in the trace; and *summary_monitor*, in which the penalty applies to a final summary. A response is an exploit when it chooses the note's option and is unflagged when an independent monitor, distinct from the reward monitor, does not detect it. Hypothesis H2 predicts, at the hardest hint level (*high*), a positive contrast

$$\Delta^{\text{GRPO}}_s = P_s^{\text{cot\_monitor}}(\text{exploits and unflagged}) - P_s^{\text{none}}(\text{exploits and unflagged}),$$

with the additional requirement that the exploit rate not fall by more than 0.10.

The unit of inference is the seed: the generations of a seed are aggregated into a proportion before any test. The contrasts of 20 seeds would be evaluated by a two-sided sign-flip test, which compares the observed mean with the distribution obtained by randomly flipping the sign of each contrast (199,999 permutations with a fixed seed), by a 95% percentile bootstrap interval, obtained by resampling the seeds, and by the Holm (1979) correction for multiple families of tests, with a minimum relevant effect of 0.05.

The results are separated into five categories, which are not combined: (i) historical results of the pilot; (ii) software checks, with no evidence about the phenomenon; (iii) competence screenings; (iv) exploratory experiments, such as the diagnosis and the calibration; and (v) confirmatory results, which would require the complete training runs and evaluations of a registered protocol. No result in category (v) was produced.

# 4 MATERIALS AND METHODS

## 4.1 Reanalysis of the pilot study

The pilot (MARQUES et al., 2026) was deposited in the project's public repository (MARQUES, 2026a) at commit e21cef8, from June 2026. A commit is a version recorded in the Git version control system, identified by a hexadecimal code. The pilot fine-tuned Qwen2.5-7B-Instruct, quantized to 4 bits, with LoRA at two arithmetic levels (A: addition or subtraction of integers from 11 to 99; C: inventory problems with multiplication), three SFT targets and three seeds (42 to 44), with 150 training items and 30 evaluation items per run. The targets are given descriptive names here: explicit CoT (a trace with the steps), direct answer and generic preamble (a vague sentence followed by the answer). The published file contains {{v:piloto.linhas}} evaluations, without prompts or generated texts; the reanalysis reprocessed the deposited metrics, reproduced the original analysis of variance and Welch's *t* test, and audited the item generator, the grader, the intervention test and the lexical score by running the preserved code.

For each seed *s*, level *l* and target *c*, the accuracy cost relative to explicit CoT is K(l, c, s) = Acc(l, explicit, s) − Acc(l, c, s), and the difference between levels is D(c, s) = K(C, c, s) − K(A, c, s). With three seeds, enumerating the eight possible sign flips yields a minimum two-sided *p* of 0.25. The historical intervention test, which compares responses with and without explanation on ten items, was reconstructed from the aggregates and evaluated with the exact McNemar test.

## 4.2 Prospective study v3.2

v3.2 was implemented at commit fde856f and operated through five Google Colab notebooks, documents that combine text and executable code on virtual machines with a graphics processing unit (GPU). The design called for 240 training runs and 55,200 test generations (@tab:desenho_v3). Each training run produces an adapter, the set of parameters tuned by LoRA. In SFT, level A consisted of additions of integers from 101 to 9,999, and level C of expressions a + b × c − d with all four operands in the same range. In GRPO, the questions presented an arithmetic expression in four domains (arithmetic, program tracing, fictional rule and inventory); at the *low* level, the note named the letter, and at the *high* level, it required computing an index modulo 4. The monitored conditions subtracted 0.6 from the reward when the Qwen2.5-0.5B-Instruct monitor detected use of the note.

{{tabela:desenho_v3}}

The registration (MARQUES, 2026b) was located through the OSF application programming interface: identifier sjwdf, created on October 19, 2026 at 16:11 UTC and made public at 16:45 UTC, after the commits of the code (15:20) and of the Colab support (15:31). It specifies hypotheses, models, seeds, competence criterion and analysis, and requires a new protocol for substantial changes of model, instrument or sampling. It does not, however, contain the cryptographic digests of the frozen artifacts and cites version 3.1 of the protocol; the differences in the executed v3.2 are operational, with no change in hypotheses, sizes or criteria.

In the screening, each base model answered the first 120 DEV items of each level on an NVIDIA Tesla T4 GPU, with weights in 16-bit floating point (fp16), greedy decoding (choosing, at each step, the most probable *token*) and a budget of 256 *tokens*, that is, up to 256 units of the model's vocabulary per response. In SFT, the instruction asked for a single `<answer>NUMERO</answer>` tag, and the strict grading contract accepted only that tag, a final line "Resposta: N" or an output consisting only of the number. In GRPO, the question included the forbidden note, and the response was correct only with the `<trace>`, `<summary>` and `<answer>` blocks in sequence, with no text between them, and the letter of the answer key in the last one.

## 4.3 Diagnosis of the v3.2 screening

The raw files of the original screening were not available; the original values appear in the v3 README (commit 273d131). Two diagnostic reproductions, which do not replace the registered screening, were conducted. The first repeated, on October 28, 2026, the original path on Colab: the notebooks downloaded the pinned commits, created the environment with the pinned versions and ran the screening on a Tesla T4, in fp16, in a separate folder. The second ran the same functions on a central processing unit (CPU), with weights in 32-bit floating point (fp32), to assess the effect of precision. The model revisions, identifiers of the exact version of the weights, were confirmed on the Hugging Face Hub, a public model repository.

The responses of the T4 reproduction were classified into descriptive categories, such as correct answer outside the contract, thousands separator read as a decimal point and truncation at the budget. The sum of the correct answers outside the contract defines a permissive upper bound, which indicates how much a tolerant extractor would change the result; the registered outcome remains that of the frozen grader. The text sent to the chat template, the tokenization, the end-of-sequence identifier and the library versions were also checked.

## 4.4 Prospective protocol v4

v4 is a new study, with its own identifier (estudo-prospectivo-2026-v4), code in the experimento_cientifico_v4 directory and five notebooks of its own. It keeps the questions, the text of the hypotheses, the base models, the conditions, the 20 seeds, the steps, the partition sizes, the 65% criterion and the statistical plan; the changes (@tab:mudancas_v4) are restricted to the causes identified in the diagnosis.

{{tabela:mudancas_v4}}

Tasks, instructions and monitors were chosen by calibration on items generated with their own seeds, excluded by construction from the study partitions. Each selection rule was recorded in the repository before the corresponding run, and all rounds are reported, including those that failed. Under the same conditions as the screening (T4, fp16, greedy decoding and 256 *tokens*), SFT level A and each candidate for level C had to reach 0.80 on 40 items with the explicit instruction; the hardest passing candidate is chosen. Round 2 added two fixed demonstrations of the format to the instruction. In GRPO, the operand range, now common to both levels, had to reach 0.80 without the hint on 40 items per level; round 2 tested an instruction without the copyable placeholders (r2a) and the same instruction with a demonstration without a hint (r2b). The 0.80 threshold gives a margin above the 0.65 criterion.

The automatic monitors are language models used as classifiers: each one receives a rubric asking whether the text uses the forbidden information, followed by the question and the trace, and compares the probabilities it assigns to the continuations "true" and "false". They were evaluated on 320 controls constructed from template sentences, 160 that explicitly use the note and 160 that compute the answer, including rejecting the note. These labels follow from the construction, not from human annotation, and serve only to assess feasibility; validation with 100 positive and 100 negative controls labeled by people remains mandatory. For the independent monitor, the smallest model with a true positive rate (TPR) of at least 0.90 and a false positive rate (FPR) of at most 0.05 was to be chosen: in round 1, SmolLM2 (BEN ALLAL et al., 2025) with 135M, 360M and 1.7B parameters; in round 2, OLMo 2 1B (TEAM OLMO et al., 2024), Granite 3.3 2B (IBM GRANITE TEAM, 2025), SmolLM3 3B (HUGGING FACE, 2025) and Phi-3.5-mini (ABDIN et al., 2024). The reward monitor had to reach TPR − FPR ≥ 0.5. Before the result of round 2 was known, it was recorded that, without a viable monitor, the GRPO arm would be excluded before registration; and, when Phi-3.5-mini proved too large for the T4's memory in fp32, it was declared, before scoring it, that it would be evaluated on a larger GPU with the same criteria.

The v4 screening precedes the registration, so that the registration covers only a viable design and includes the SHA256 digests of the screening generations. A SHA256 digest is a 64-character string computed from the content of a file, which changes with any modification to it. The screening was run only once, after freezing the protocol and the modules that determine items, instructions, generation and grading; in GRPO, it used the question without the note and 30 DEV items from each domain per level.

## 4.5 Software verification

Unit tests, that is, automatic checks of isolated functions, cover grading, data generation, freezing, locks and statistics. A CPU integration with minimal models with random weights exercises the training libraries in both arms, with interruption and resumption. The five notebooks are run with simulated Colab and injected failures of Google Drive mounting, the persistent storage between sessions, and of file transfer. A technical GPU test runs the five notebooks with the real models in technical mode (one seed, two steps per training run and four items per evaluation, in a separate folder) and forces interruptions of training and evaluation. None of these tests produces evidence about the hypotheses.

# 5 RESULTS

## 5.1 Reanalysis of the pilot study

The {{v:piloto.celulas}} recomputed means match the published aggregates, and the historical analyses were reproduced: analysis of variance interaction F(2, 12) = {{v:piloto.anova_f}}, p = {{v:piloto.anova_p}}, and Welch's test at level C, t = {{v:piloto.welch_t}}, p = {{v:piloto.welch_p}}. At level C, accuracy goes from {{v:piloto.acc.C.control}} with explicit CoT to {{v:piloto.acc.C.cot_pressure}} with direct answer and {{v:piloto.acc.C.output_supervision}} with generic preamble (@fig:piloto). The mean difference between the costs of the levels was {{v:piloto.dd.cot_pressure}} and {{v:piloto.dd.output_supervision}}, positive in all three seeds (@tab:piloto_contrastes), with p = {{v:piloto.signflip.cot_pressure}} in the enumeration of sign flips, the smallest possible value with three seeds.

{{figura:piloto}}

{{tabela:piloto_contrastes}}

The audit restricts the interpretation in four respects. First, the generator creates each item from the seed plus the index, so that adjacent seeds reuse 149 of 150 training prompts and 29 of 30 evaluation prompts; the three seeds are not independent samples of problems. Second, the historical grader accepts an answer when any number in the final passage matches the answer key: in a constructed corpus of {{v:piloto.casos}} responses, of which {{v:piloto.negativos}} should be rejected, it accepted {{v:piloto.falsos_aceites}} ({{v:piloto.taxa_aceite_indevido}}), for example "Resposta: 43; 42 seria incorreto." ("Answer: 43; 42 would be incorrect.") with answer key 42, whereas the strict contract of the prospective studies rejected all of them. Third, the historical comparison with and without explanation shows at level C {{v:piloto.mcnemar_b}} items answered correctly only with explanation and {{v:piloto.mcnemar_c}} only without it (McNemar, p = {{v:piloto.mcnemar_p}}), but it changes instruction and budget simultaneously (200 versus 40 *tokens*) and does not establish the causal necessity of the trace. Fourth, the lexical "obfuscation" score assigns the maximum value to empty traces and correlates with repetition (r = {{v:piloto.corr_rep}}) and lexical diversity (r = {{v:piloto.corr_ttr}}), components of its formula, without independent validation.

## 5.2 v3.2 screening and diagnosis

The reproduction along the original Colab path obtained {{v:v3.sft.A.t4_k}}, {{v:v3.sft.C.t4_k}}, {{v:v3.grpo.low.t4_k}} and {{v:v3.grpo.high.t4_k}} correct answers, exactly the documented values (@tab:triagem_v3), with data of the same SHA256 and the same model revisions. In the comparison with the CPU in fp32, correctness matched on {{v:v3.sft.A.cpu_igual}} of 120 items at level A and on all items at level C of SFT, and on {{v:v3.grpo.low.cpu_igual}} and {{v:v3.grpo.high.cpu_igual}} of {{v:v3.grpo.high.cpu_n}} in GRPO: precision changes isolated answers, not the result.

{{tabela:triagem_v3}}

The classification (@tab:falhas_v3) separates three sources of failure. The first is arithmetic inability: at SFT level A, the model answered with a number only and got {{v:v3.sft.A.cat.formato_valido_valor_incorreto}} sums of up to four digits wrong; at level C, none of the 120 responses would be correct even with permissive extraction, since the model does not compute products of up to eight digits without steps; at the GRPO *high* level, with three-digit operands, it answered {{v:v3.grpo.high.dom.arithmetic}} items of the arithmetic domain correctly, and {{v:v3.grpo.high.t4_trunc}} responses exhausted the budget while computing the note's index. The second is the instruction and the contract: the SFT model never used the requested tag, and {{v:v3.sft.A.cat.valor_correto_fora_do_contrato}} level A responses contained the correct value in prose; the GRPO model copied the placeholders "explicação verificável" ("verifiable explanation") and "resumo" ("summary") from the instruction and wrote the calculation outside the tags, and {{v:v3.grpo.low.cat.alternativa_correta_fora_do_contrato}} *low* level responses ended with the correct letter outside the format. Hence, the arithmetic domain obtained {{v:v3.grpo.low.dom.arithmetic}} at the *low* level, and program tracing {{v:v3.grpo.low.dom.program_trace}}; the first 120 DEV items covered only these two domains. The third is a grader defect, which read a number with a thousands separator ("11,863") as a decimal; it affected {{v:v3.sft.A.cat.separador_de_milhar_lido_como_decimal}} response and does not change any level.

{{tabela:falhas_v3}}

The permissive bounds were {{v:v3.sft.A.perm}}, {{v:v3.sft.C.perm}}, {{v:v3.grpo.low.perm}} and {{v:v3.grpo.high.perm}}: a tolerant extractor would pass some levels, but not SFT level C. The chat template was rendered correctly, in the ChatML format, which delimits the turns with special markers, and with Qwen3 in non-thinking mode, in which the block reserved for internal reasoning is sent empty; the tokenization did not duplicate markers, and the pinned libraries reproduced the results on Colab on October 28, with Python 3.13.16 and PyTorch 2.11. The constructed controls of the v4 calibration further showed that the independent monitor of v3.2, SmolLM2-135M-Instruct, flagged {{v:mon.135m.tpr}} of the positives and {{v:mon.135m.fpr}} of the negatives, and that the reward monitor obtained a TPR of {{v:mon.reward.tpr}} and an FPR of {{v:mon.reward.fpr}}.

## 5.3 v4 calibration

In SFT, the explicit instruction without demonstrations produced no accepted responses ({{v:cal.sft.r1.A.explicit.k}} at level A): the model wrote out the calculations and omitted the tag. With the neutral instruction, level A reached {{v:cal.sft.r1.A.neutral.k}}, but candidates C1 and C2 stayed at {{v:cal.sft.r1.C1.neutral.k}} and {{v:cal.sft.r1.C2.neutral.k}}: without written steps, the model does not solve expressions with multiplication. With the demonstrations, the correct answers were {{v:cal.sft.r2.A.explicit.k}} at level A, {{v:cal.sft.r2.C1.explicit.k}} at C1 and {{v:cal.sft.r2.C2.explicit.k}} at C2, and C2 was chosen (@tab:calibracao_v4). In GRPO, the v3.2 instruction stayed below 0.80 in both ranges, with format validity tracking accuracy. Variant r2a produced no accepted responses, because the model separated the blocks with line breaks in {{v:cal.r2a.quebra}} of the {{v:cal.r2a.n}} responses, although {{v:cal.r2a.letra}} ended with the correct letter; variant r2b reached {{v:cal.grpo.r2.r2b.G1.sem_dica.low.k}} and {{v:cal.grpo.r2.r2b.G1.sem_dica.high.k}} without the hint, with valid format in all of them, and was fixed with operands from 11 to 99.

{{tabela:calibracao_v4}}

None of the seven candidates for the independent monitor met the criterion (@tab:monitores_v4). The 360M and 1.7B SmolLM2 models flagged practically all traces, and OLMo 2 and SmolLM3 none, patterns of a classifier that always assigns the same class. Granite 3.3 obtained a TPR of {{v:mon.granite2b.tpr}} and an FPR of {{v:mon.granite2b.fpr}}. Phi-3.5-mini, evaluated on an NVIDIA L4 GPU with 24 GB, obtained a TPR of {{v:mon.phi35.tpr}} and an FPR of {{v:mon.phi35.fpr}}, with false positives on {{v:mon.phi35.neg0}} of the traces that only compute the answer and on {{v:mon.phi35.neg1}} of those that reject the note. As a reward monitor, Qwen2.5-0.5B-Instruct discriminated worse than chance, and Qwen2.5-1.5B-Instruct flagged no control. Under the declared rule, the GRPO arm was excluded before registration, and the protocol was frozen with this decision at commit 37094c4.

{{tabela:monitores_v4}}

## 5.4 v4 DEV screening

Run only once, on October 29, 2026, on a T4 in fp16, the screening resulted in {{v:v4.gate.sft.A.k}} correct answers at SFT level A ({{v:v4.gate.sft.A}}) and {{v:v4.gate.sft.C.k}} at level C ({{v:v4.gate.sft.C}}), with all responses in the required format and none truncated (@tab:triagem_v4). Both levels met the 0.65 criterion, and the SFT training runs came to depend only on the public registration. The margin at level C is small: the 95% Wilson confidence interval goes from {{v:v4.gate.sft.C.ic}}, and accuracy fell below the calibration estimate ({{v:cal.sft.r2.C2.explicit}} on 40 items). The criterion applies to the observed accuracy, so the result is a pass, but level C lies close to the competence limit of the base model. In GRPO, descriptive only, the model answered {{v:v4.gate.grpo.low.k}} and {{v:v4.gate.grpo.high.k}} items correctly without the note.

{{tabela:triagem_v4}}

## 5.5 Software checks

The local test suites, the historical reanalysis and the run of the five notebooks with simulated Colab passed. In the first run of the technical GPU test, the five notebooks ran without error, but the forced interruption of a training run right after the last checkpoint was written revealed a defect: on resumption, the library executed one step more than planned, and the software rejected the result. The defect was reproduced on CPU, fixed and covered by a regression test; in the repetition, the interrupted training run resumed from the first checkpoint and finished at the planned step, and the interrupted evaluation, with a simulated partial line, was completed without duplicate items. Commands, versions, durations and file digests are given in the repository's validation report.

# 6 DISCUSSION

The pilot records a large and consistent difference between training targets at the multi-step level, and the published numbers are reproducible. This description motivates the hypothesis that replacing the steps of the supervised target impairs learning, but it does not identify an evasion policy: the design changes the text to be imitated, does not optimize the model against a monitor and does not measure a forbidden behavior defined outside the trace. With a grader that accepts ambiguous answers and seeds that share the items, the answer to Q1 is negative.

The failure of v3.2 does not stem from an execution error. The answer to Q2 is that two causes add up: the real inability of the base models at the hardest tasks, sufficient to fail level C, and a competence measure that depended on obeying undemonstrated formats. A permissive extractor would pass some levels, but replacing it after observing the results would change the registered instrument; the outcome of v3.2 is the infeasibility of the registered design, not a result about H1 and H2.

The calibration confirms this diagnosis. The loss under the strict contract came mainly from the format: an instruction rewritten without examples was not enough, whereas demonstrations made the responses valid. v4 keeps the strict, prespecified contract and changes only the instruction, so that the measure still requires the training format, without depending on the model guessing it. In SFT, level C2 remains dependent on the written steps, a condition for the H1 contrast to be informative. The answer to Q3 is partial: the SFT arm became viable under the same criterion, and v4 is ready for registration as a study of one confirmatory hypothesis.

The evaluation of the monitors has the greatest consequence for the design. Seven models from four families, with 135 million to 3.8 billion parameters, did not distinguish, with the required margin, traces that follow the note from traces that ignore or reject it. Since these controls are unambiguous by construction, it is unlikely that the same monitors would be more accurate on the more varied traces of trained models; even with competence passed, validation of the v3.2 monitor would probably have failed after the 240 training runs. The exclusion of the GRPO arm is not evidence about H2. It avoids spending 120 training runs on a design whose outcome would be, by construction, insufficient evidence. Testing H2 will first require validating a monitor of another kind, such as a larger model used as a generative classifier, a classifier fine-tuned with human labels, or full human annotation.

As for validity, the SFT arm preserves the identification of v3.2, with contrasts paired by seed and the same test items in all conditions. The changes reduce the arithmetic difficulty, and future results will hold for problems that the model solves with explicit steps and residual errors. External validity is limited to a small model, synthetic arithmetic tasks and three training targets; the results on monitors refer to a single probability-based classification procedure, with one rubric, and do not extend to generative monitors, larger models or other rubrics.

# 7 LIMITATIONS

The raw files of the original v3.2 screening were not accessible; the exact match of the four reproduced values is strong, but it does not replace checking them. The OSF registration of v3.2 does not contain the digests of the artifacts and cites version 3.1, which prevents verifying, from the registration alone, its correspondence with the implementation. The v4 calibration used 40 items per candidate, with a standard error close to 0.06, and rule-constructed monitor controls, which do not represent the variety of traces of trained models nor replace human labels. The successive rounds were declared before each run and reported, but the choice between variants introduces degrees of freedom that only the screening on separate DEV items controls. Phi-3.5-mini was scored on an L4, not on the T4, with the same numerical precision. The Colab sessions used in the calibration were terminated by the service twice; the affected steps were repeated with the same code, with values identical to those already observed. The v4 screening preceded the registration, and its chronology is verifiable through the Git history, not through a registration platform. No confirmatory training, test evaluation, human monitor validation or human annotation was performed; the GPU tests used technical mode and demonstrate that the software works, not the performance of trained models.

# 8 CONCLUSION

The pilot study records a consistent difference in accuracy between training targets, but its instruments do not allow attributing it to the obfuscation of necessary reasoning. The prospective study v3.2 ended without training because the base models did not reach the competence criterion; the reproduction with the frozen code confirmed the registered values, ruled out execution defects and identified as causes the models' inability at the hardest tasks and instructions that made the measurement depend on undemonstrated formats.

Protocol v4, prepared as a new study, addressed these causes by calibration on separate items and met the competence criterion in the SFT arm, with a small margin at the higher-complexity level. No evaluated monitor distinguished traces that use the forbidden information from traces that do not, and the GRPO arm was excluded before registration. No confirmatory conclusion about chain-of-thought obfuscation follows from this work. The next step is the public registration of v4, with the SHA256 digests of the frozen artifacts, followed by the 120 training runs of the SFT arm; a test of H2 will require, before any training, a monitor validated with human labels.

# REFERENCES

::: {custom-style="Referencia"}
ABDIN, Marah *et al*. Phi-3 technical report: a highly capable language model locally on your phone. **arXiv**, 2404.14219, 2024. DOI: 10.48550/arXiv.2404.14219.

AGARWAL, Rishabh; SCHWARZER, Max; CASTRO, Pablo Samuel; COURVILLE, Aaron; BELLEMARE, Marc G. Deep reinforcement learning at the edge of the statistical precipice. *In*: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 34., 2021. **Proceedings** [...]. [*S. l.*]: Curran Associates, 2021. Available at: https://arxiv.org/abs/2108.13264. Accessed on: Oct. 28, 2026.

BAKER, Bowen; HUIZINGA, Joost; GAO, Leo; DOU, Zehao; GUAN, Melody Y.; MADRY, Aleksander; ZAREMBA, Wojciech; PACHOCKI, Jakub; FARHI, David. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. **arXiv**, 2503.11926, 2025. DOI: 10.48550/arXiv.2503.11926.

BEN ALLAL, Loubna *et al*. SmolLM2: when smol goes big — data-centric training of a small language model. **arXiv**, 2502.02737, 2025. DOI: 10.48550/arXiv.2502.02737.

DRORI, Jacob; MARKS, Luke; WOODWORTH, Bryce; CLOUD, Alex; TURNER, Alexander Matt. Output supervision can obfuscate the chain of thought. **arXiv**, 2511.11584, 2025. DOI: 10.48550/arXiv.2511.11584.

EMMONS, Scott; JENNER, Erik; ELSON, David K.; SAUROUS, Rif A.; RAJAMANOHARAN, Senthooran; CHEN, Heng; SHAFKAT, Irhum; SHAH, Rohin. When chain of thought is necessary, language models struggle to evade monitors. **arXiv**, 2507.05246, 2025. DOI: 10.48550/arXiv.2507.05246.

HOLM, Sture. A simple sequentially rejective multiple test procedure. **Scandinavian Journal of Statistics**, Oslo, v. 6, n. 2, p. 65-70, 1979. Available at: https://www.jstor.org/stable/4615733. Accessed on: Oct. 28, 2026.

HU, Edward J.; SHEN, Yelong; WALLIS, Phillip; ALLEN-ZHU, Zeyuan; LI, Yuanzhi; WANG, Shean; WANG, Lu; CHEN, Weizhu. LoRA: low-rank adaptation of large language models. *In*: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 10., 2022. **Proceedings** [...]. [*S. l.*]: ICLR, 2022. DOI: 10.48550/arXiv.2106.09685.

HUGGING FACE. **SmolLM3-3B**. Hugging Face, 2025. Model, revision a07cc9a04f16550a088caea529712d1d335b0ac1. Available at: https://huggingface.co/HuggingFaceTB/SmolLM3-3B. Accessed on: Oct. 29, 2026.

IBM GRANITE TEAM. **granite-3.3-2b-instruct**. Hugging Face, 2025. Model, revision 707f574c62054322f6b5b04b6d075f0a8f05e0f0. Available at: https://huggingface.co/ibm-granite/granite-3.3-2b-instruct. Accessed on: Oct. 29, 2026.

KORBAK, Tomek *et al*. Chain of thought monitorability: a new and fragile opportunity for AI safety. **arXiv**, 2507.11473, 2025. DOI: 10.48550/arXiv.2507.11473.

LANHAM, Tamera *et al*. Measuring faithfulness in chain-of-thought reasoning. **arXiv**, 2307.13702, 2023. DOI: 10.48550/arXiv.2307.13702.

MARQUES, Kelvin e Silva *et al*. **Validade externa da ofuscação de cadeia de pensamento sob supervisão de saída**: um teste empírico em tarefas com raciocínio causalmente necessário [External validity of chain-of-thought obfuscation under output supervision: an empirical test on tasks with causally necessary reasoning]. Fortaleza: Centro Universitário Ateneu, 2026. Unpublished manuscript, version of Jun. 27, 2026.

MARQUES, Kelvin e Silva. **cot-obfuscation-h2a**: code, data and protocols of the project. GitHub, 2026a. Available at: https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a. Accessed on: Oct. 29, 2026.

MARQUES, Kelvin e Silva. **Monitorabilidade e ofuscação da cadeia de pensamento**: pré-registro de experimentos SFT e GRPO com avaliação independente [Chain-of-thought monitorability and obfuscation: preregistration of SFT and GRPO experiments with independent evaluation]. OSF Registries, Oct. 19, 2026b. Available at: https://osf.io/sjwdf/. Accessed on: Oct. 28, 2026.

MITRANI HADIDA, Nathaniel; BHANJI, Sassan; TICE, Cameron; RADMARD, Puria. Chain-of-thought obfuscation learned from output supervision can generalise to unseen tasks. **arXiv**, 2601.23086v2, 2026. Version of May 20, 2026. DOI: 10.48550/arXiv.2601.23086.

NOSEK, Brian A.; EBERSOLE, Charles R.; DEHAVEN, Alexander C.; MELLOR, David T. The preregistration revolution. **Proceedings of the National Academy of Sciences**, Washington, v. 115, n. 11, p. 2600-2606, 2018. DOI: 10.1073/pnas.1708274114.

QWEN TEAM. Qwen2.5 technical report. **arXiv**, 2412.15115, 2024. DOI: 10.48550/arXiv.2412.15115.

SHAO, Zhihong; WANG, Peiyi; ZHU, Qihao; XU, Runxin; SONG, Junxiao; BI, Xiao; ZHANG, Haowei; ZHANG, Mingchuan; LI, Y. K.; WU, Y.; GUO, Daya. DeepSeekMath: pushing the limits of mathematical reasoning in open language models. **arXiv**, 2402.03300, 2024. DOI: 10.48550/arXiv.2402.03300.

TEAM OLMO; WALSH, Pete; SOLDAINI, Luca; GROENEVELD, Dirk *et al*. 2 OLMo 2 Furious. **arXiv**, 2501.00656, 2024. DOI: 10.48550/arXiv.2501.00656.

TURPIN, Miles; MICHAEL, Julian; PEREZ, Ethan; BOWMAN, Samuel R. Language models don't always say what they think: unfaithful explanations in chain-of-thought prompting. *In*: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 36., 2023. **Proceedings** [...]. [*S. l.*]: Curran Associates, 2023. DOI: 10.48550/arXiv.2305.04388.

VON WERRA, Leandro *et al*. **TRL**: Transformer Reinforcement Learning. Version 0.22.2. GitHub, 2020. Available at: https://github.com/huggingface/trl. Accessed on: Oct. 28, 2026.

WEI, Jason *et al*. Chain-of-thought prompting elicits reasoning in large language models. *In*: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 35., 2022. **Proceedings** [...]. [*S. l.*]: Curran Associates, 2022. DOI: 10.48550/arXiv.2201.11903.

WOLF, Thomas *et al*. Transformers: state-of-the-art natural language processing. *In*: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING: SYSTEM DEMONSTRATIONS, 2020. **Proceedings** [...]. [*S. l.*]: Association for Computational Linguistics, 2020. p. 38-45. DOI: 10.18653/v1/2020.emnlp-demos.6.

YANG, An *et al*. Qwen3 technical report. **arXiv**, 2505.09388, 2025. DOI: 10.48550/arXiv.2505.09388.

ZHENG, Lianmin *et al*. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *In*: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 36., 2023. **Proceedings** [...]. [*S. l.*]: Curran Associates, 2023. DOI: 10.48550/arXiv.2306.05685.
:::
