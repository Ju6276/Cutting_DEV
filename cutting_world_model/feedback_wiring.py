"""Check the DiSECt → sensor relay → filter → Gazebo controller signal path."""


def verify_feedback_graph(system_state, namespace='b_bot'):
    publishers, subscribers, _ = system_state
    publishers, subscribers = dict(publishers), dict(subscribers)
    topics = ('/disect/wrench', '/disect/wrench_sensor', '/' + namespace.strip('/') + '/wrench/filtered')
    chain = []
    for topic in topics:
        sources = set(publishers.get(topic, []))
        sinks = set(subscribers.get(topic, []))
        if not sources or not sinks:
            raise RuntimeError('Disconnected cutting-force topic %s: publishers=%s subscribers=%s'
                               % (topic, sorted(sources), sorted(sinks)))
        chain.append((sources, sinks))
    for previous, following in zip(chain, chain[1:]):
        if not previous[1].intersection(following[0]):
            raise RuntimeError('Cutting-force topic endpoints do not form a relay/filter chain')
    if not any(node.rsplit('/', 1)[-1] == 'gazebo' for node in chain[-1][1]):
        raise RuntimeError('Filtered cutting force does not reach the Gazebo controller')
    return {topic: {'publishers': sorted(sources), 'subscribers': sorted(sinks)}
            for topic, (sources, sinks) in zip(topics, chain)}
