# Historical records and recovery

Only dated experimental records and variant provenance remain as local archives.
They are not current instructions. Use [PLAN](../PLAN.md), [TASK_FREE_CUT](../TASK_FREE_CUT.md)
and [METHOD](../METHOD.md) for current work.

| Retained record | Scope |
|---|---|
| [RESULTS November 2–10](RESULTS_20261102_20261110.md) | Original experiments, negative results, comparisons and corrections |
| [RESULTS November 11–12](RESULTS_20261111_20261112.md) | Retired S3, contact fixes, fixture/free-object measurements and free-cut development |
| [Historical method variants](METHOD_variants_20261112.md) | Names and implementations of old critic/risk variants |

## Removed duplicate plans and handoffs

These files are removed from the working tree, **not from Git history**. Links
below pin the last source snapshot used in this cleanup (`12b951f`), so old claims
stay recoverable without presenting them as the current task. No weights, run
directories, logs, source code or configuration files are deleted by this cleanup.

- [docs/CONTACT_FIX_REVIEW.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/CONTACT_FIX_REVIEW.md)
- [docs/FREE_CUT_REPAIR.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/FREE_CUT_REPAIR.md)
- [docs/FREE_CUT_REVIEW.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/FREE_CUT_REVIEW.md)
- [docs/READING_LIST.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/READING_LIST.md)
- [docs/archive/PLAN_20261110.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/archive/PLAN_20261110.md)
- [docs/archive/PLAN_20261111.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/archive/PLAN_20261111.md)
- [docs/archive/PLAN_20261112_before_free_task.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/archive/PLAN_20261112_before_free_task.md)
- [docs/archive/PLAN_20261112_pre_review.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/archive/PLAN_20261112_pre_review.md)
- [docs/archive/PAPER_20261111.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/archive/PAPER_20261111.md)
- [docs/archive/PAPER_20261112_pre_review.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/archive/PAPER_20261112_pre_review.md)
- [docs/archive/ACTION_PERTURBATIONS.md](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/archive/ACTION_PERTURBATIONS.md)

The [official-implementation mapping](../OFFICIAL_SUPERMIMIC.md) remains at its
original path because the adapter records it in run manifests. The short
[evidence-package pointer](EVIDENCE_20261110.md) likewise preserves a path emitted
by the legacy report builder; its obsolete plan was removed.
Paper/data entry points are consolidated in the [literature index](../literature/README.md).
Individual paper/code reading records are retained.

The former long [METHOD snapshot](https://github.com/Ju6276/Cutting_DEV/blob/12b951f/docs/METHOD.md)
and [BENCHMARK](../BENCHMARK.md) preserve legacy source/formula details.
[Current RESULTS](../RESULTS.md) retains the November 13 evidence and its corrections.

To recover a removed file on a server without changing an active checkout:

~~~bash
git show 12b951f:docs/FREE_CUT_REPAIR.md
~~~
