import zmq
import veinsgym_pb2 as pb


def main():
    step_count=0
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:5555")
    print("[Python] Listening on 127.0.0.1:5555")

    while True:
        msg = sock.recv()
        print(f"[Python] Received {len(msg)} bytes")

        req = pb.Request()
        try:
            req.ParseFromString(msg)
        except Exception as e:
            print("[Python] ParseFromString failed:", e)
            sock.send(pb.Reply().SerializeToString())
            continue

        which = req.WhichOneof("payload")  # this should be correct for veins-gym proto
        print("[Python] Request oneof:", which)

        reply = pb.Reply()
        reply.id = req.id

        # Shutdown handling
        if req.HasField("shutdown"):
            print("[Python] Shutdown requested")
            sock.send(reply.SerializeToString())
            break

        # Init handling (optional, but nice to log)
        if req.HasField("init"):
            print("[Python] Init received")
            sock.send(reply.SerializeToString())
            continue

        # Step handling: send an action = discrete.value
        if req.HasField("step"):
            obs = list(req.step.observation.box.values)
            rew = list(req.step.reward.box.values) if req.step.HasField("reward") else []
            print("[Python] obs:", obs)
            if rew:
                print("[Python] reward:", rew[0])

            # Example policy: if delay high -> ACTIVE, else try SLEEP
            thr, delay, jitter, loss, numUe, energyJ = obs
            if delay > 0.02 or loss > 0.0:
                a = 0  # ACTIVE
            else:
                a = 1  # SLEEP (toy policy)
            
            a=2
            step_count += 1
    # alternate: run 10 steps unpaused, 10 steps paused
            a = 0 if (step_count // 10) % 2 == 0 else 2

            reply.action.discrete.value = a
            print("[Python] Action sent:", a)

            reply.action.discrete.value = a
            sock.send(reply.SerializeToString())
            continue

        # Fallback
        sock.send(reply.SerializeToString())
        print("[Python] Replied (fallback)")


if __name__ == "__main__":
    main()
