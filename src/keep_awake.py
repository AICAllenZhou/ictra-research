"""Keep Windows out of IDLE sleep while a long job runs (Allen 2026-10-19, after v03b's sweep lost 19 h to Modern Standby).

    python src/keep_awake.py -- bash src/21_run_v07.sh        # run a command, exit with its exit code
    python src/keep_awake.py --pid 12345                       # hold until an already-running process exits

SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) tells Windows a task is running, so it does not sleep for lack
of user input. It does NOT stop a deliberate lid close or power-button press (those still enter Modern Standby when the
active power scheme says so). The state is released in `finally` with SetThreadExecutionState(ES_CONTINUOUS), on normal
exit and on exceptions/Ctrl-C; Windows also drops it when this process ends, so a hard kill cannot leave the machine
unable to sleep. No effect on any result: it only changes power management while the job runs.
"""
import argparse, ctypes, subprocess, sys, time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def _set(flags):
    if sys.platform != "win32":
        return 1
    return ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, help="hold until this (Windows) process id exits")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd and a.pid is None:
        ap.error("give -- <command> or --pid")
    if _set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) == 0:
        print("keep_awake: SetThreadExecutionState failed; the job runs without the keep-awake request", file=sys.stderr, flush=True)
    else:
        print(f"keep_awake: idle sleep blocked ({'pid ' + str(a.pid) if a.pid else ' '.join(cmd)})", file=sys.stderr, flush=True)
    code = 0
    try:
        if cmd:
            code = subprocess.call(cmd)
        else:
            import psutil
            while psutil.pid_exists(a.pid):
                time.sleep(30)
    finally:
        _set(ES_CONTINUOUS)                 # release: the machine may idle-sleep again
        print("keep_awake: released", file=sys.stderr, flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
