import pytest

from cutting_world_model.feedback_wiring import verify_feedback_graph


def graph():
    return [[['/disect/wrench', ['/bridge']], ['/disect/wrench_sensor', ['/relay']],
             ['/b_bot/wrench/filtered', ['/filter']]],
            [['/disect/wrench', ['/relay']], ['/disect/wrench_sensor', ['/filter']],
             ['/b_bot/wrench/filtered', ['/gazebo', '/learner']]], []]


def test_material_force_reaches_controller_through_sensor_filter():
    assert '/gazebo' in verify_feedback_graph(graph())['/b_bot/wrench/filtered']['subscribers']


def test_learner_subscription_does_not_replace_controller_feedback():
    state = graph()
    state[1][-1][1] = ['/learner']
    with pytest.raises(RuntimeError, match='Gazebo controller'):
        verify_feedback_graph(state)


def test_disconnected_material_topic_is_rejected():
    state = graph()
    state[1][0][1] = []
    with pytest.raises(RuntimeError, match='Disconnected cutting-force topic'):
        verify_feedback_graph(state)


def test_unrelated_publishers_do_not_form_a_feedback_chain():
    state = graph()
    state[0][1][1] = ['/unrelated']
    with pytest.raises(RuntimeError, match='relay/filter chain'):
        verify_feedback_graph(state)
