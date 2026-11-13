"""Count-based WM diagnostics; rates are divided once after batch aggregation."""
RATE_COUNTS = {
    'term_acc': 'term_correct_count',
    'completion_false_positive_fraction': 'completion_false_positive_count',
    'completion_incomplete_fraction': 'completion_incomplete_count',
    'completion_missed_fraction': 'completion_missed_count',
    'completion_positive_fraction': 'completion_positive_count',
}
COUNT_KEYS = frozenset(RATE_COUNTS.values()) | {'metric_transition_count'}
