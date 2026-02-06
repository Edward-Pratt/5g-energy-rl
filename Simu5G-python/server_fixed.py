import argparse, time, csv
import zmq
import veinsgym_pb2 as pb

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--action", type=float, default=1.0,
                    help="Continuous action m in [0,2] (e.g. 0.7).")
    ap.add_argument("--out", default="rollout_fixed.csv")
    ap.add_argument("--clamp_lo", type=float, default=0.0)
    ap.add_argument("--clamp_hi", type=float, default=2.0)
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"[FIXED] Listening on {args.host}:{args.port} | action={args.action}")

    logf = open(args.out, "a", newline="")
    w = csv.writer(logf)
    if logf.tell() == 0:
        w.writerow(["ts","episode","step","thr","delay","jitter","loss","numUe","stepEnergyJ","rew","act_m"])

    episode = 0
    step = 0
    ep_return = 0.0

    while True:
        msg = sock.recv()
        req = pb.Request()
        req.ParseFromString(msg)

        reply = pb.Reply()
        reply.id = req.id

        if req.HasField("init"):
            step = 0
            ep_return = 0.0
            sock.send(reply.SerializeToString())
            continue

        if req.HasField("shutdown"):
            print(f"[FIXED] Shutdown (episode {episode}) return={ep_return:.3f} steps={step}")
            episode += 1
            sock.send(reply.SerializeToString())
            logf.flush()
            continue

        if req.HasField("step"):
            obs = list(req.step.observation.box.values)

            rew = 0.0
            if req.step.reward and req.step.reward.HasField("box") and len(req.step.reward.box.values) > 0:
                rew = float(req.step.reward.box.values[0])

            # continuous action (clamped)
            m = clamp(float(args.action), args.clamp_lo, args.clamp_hi)

            # IMPORTANT: send action as Box (continuous)
            # This assumes your action_space in the OMNeT++ init is Box with 1 value.
            del reply.action.box.values[:]     # clear any existing values safely
            reply.action.box.values.append(m)

            sock.send(reply.SerializeToString())

            # log
            thr, delay, jitter, loss, numUe, stepE = obs
            w.writerow([time.time(), episode, step, thr, delay, jitter, loss, numUe, stepE, rew, m])
            logf.flush()

            ep_return += rew
            step += 1
            continue

        sock.send(reply.SerializeToString())

if __name__ == "__main__":
    main()
