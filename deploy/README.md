# deploy/

The files of a served installation: four Sparks on one switch, systemd user units, one settings file. Put each in the place
named on its first lines; [`docs/setup.md`](../docs/setup.md) gives the order and
[`docs/operations.md`](../docs/operations.md) says how to run it.

- `pico.env.example` is `~/dsgb10/pico.env` on rank 0: every setting of the served line, with a comment.
- `fleet.conf.example` is the fleet file (`~/dsgb10/bin/fleet.conf`): addresses, RDMA devices, and where each Spark keeps its files.
- `pico-engine.service` runs rank 0 and, through `pico-ctl.sh`, ranks 1-3 on the workers.
- `pico-front.service` runs `dsfront`, the API and the dashboard; it stays up while the engine restarts.
- `pico-media.service` runs the media worker in a container (images and clips).
- `pico-ctl.sh` is the helper the units call: `prepare`, `ready`, `reap`, `install SRC_TREE`, `status`.
- The GPU clock tool is [`ds41f-4xgx10-switchless/tools/gpu-clock`](../ds41f-4xgx10-switchless/tools/gpu-clock); `pico-ctl.sh install` puts it in `~/dsgb10/bin` and stops if it is missing.

Replace every `<placeholder>` (a user name, an address, a host name) before the first start. `pico-ctl.sh install` copies the
binaries, the clock tool and the three units into `~/dsgb10/bin` and `~/.config/systemd/user` and starts nothing. There is no unit for
the container image encoder (`--vision container`, [`docs/vision.md`](../docs/vision.md)): the served engine encodes
images itself. `engine/server/service.sh` (no systemd) is the other way to run the same engine; use one, never both.
