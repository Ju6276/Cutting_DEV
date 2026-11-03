"""Wall-clock bounds for remote shutdown operations and independent finalization."""
import math
import threading


class ShutdownError(RuntimeError):
    def __init__(self, results):
        self.results = results
        super().__init__('Environment shutdown failed: ' + repr(results))


def run_bounded(callback, timeout=5.0):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Shutdown timeout must be finite and positive')
    completed = threading.Event()
    result = {}

    def invoke():
        try:
            callback()
            result['state'] = 'ok'
        except BaseException as error:
            result.update(state='error', error=repr(error))
        finally:
            completed.set()

    thread = threading.Thread(target=invoke, name='bounded-shutdown', daemon=True)
    thread.start()
    if not completed.wait(timeout):
        return {'state': 'timeout', 'error': 'Shutdown exceeded %.3f wall seconds' % timeout}
    return dict(result)


def close_environment(env, timeout=5.0):
    base = env.unwrapped
    results = {}
    if base.use_step_control:
        connection = base.robot_connection
        unpause = getattr(connection, 'unpause_for_shutdown', connection.unpause)
        results['unpause'] = run_bounded(unpause, timeout)
    results['close'] = run_bounded(env.close, timeout)
    arm = getattr(base, 'ur3e_arm', None)
    controller_result = getattr(arm, 'shutdown_result', None)
    if controller_result is not None:
        results['controller_shutdown'] = dict(controller_result)
    if any(result['state'] != 'ok' for result in results.values()):
        raise ShutdownError(results)
    return results


def finalize_training(save, record, finish, close, state):
    errors = []
    for name, callback in (('save', save), ('close', close),
                           ('record', record), ('finish', finish), ('record_final', record)):
        try:
            callback()
        except BaseException as error:
            errors.append({'stage': name, 'error': repr(error)})
            state['state'] = 'failed'
            state['cleanup_errors'] = list(errors)
    return errors
