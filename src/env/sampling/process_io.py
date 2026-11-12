from multiprocessing.connection import wait


def receive_result(connection, process, env_id):
    try:
        ready = wait((connection, process.sentinel))

        if connection not in ready and not connection.poll():
            raise RuntimeError(f"Environment {env_id} worker exited, exitcode={process.exitcode}")
        kind, result = connection.recv()
    except (EOFError, OSError) as error:
        raise RuntimeError(f"Environment {env_id} worker disconnected") from error
    if kind == "interrupt":
        raise KeyboardInterrupt(result)
    if kind == "error":
        raise RuntimeError(f"Environment {env_id} failed: \n{result}")
    return result


def send_requests(connections, command, payloads, env_ids):
    if len(payloads) != len(env_ids):
        raise ValueError("Payload count must match the active environments")
    for env_id, payload in zip(env_ids, payloads):
        try:
            connections[env_id].send((command, payload))
        except OSError as error:
            raise RuntimeError(f"Environment {env_id} worker disconnected") from error


def close_workers(connections, processes):
    for connection in connections:
        try:
            connection.send(("close", None))
        except OSError:
            pass
        connection.close()
    for process in processes:
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join()
        process.close()
