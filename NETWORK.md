# Preparing the network: four DGX Sparks on one RoCE switch

This document describes the network of the four Sparks that this repository's numbers come from. Use it to prepare your
own RoCE switch the same way, or to configure the same MikroTik switch with this repository's configuration. The values
below are those of the devices the numbers of this repository were measured on, and everything described was measured on this hardware unless it names another source.

**Result:** under a 3:1 incast at line rate (three Sparks sending to one) there were 0 drops and 0 RoCE retransmissions,
against 1.45 million drops in the switch's factory state. While the engine serves, all its RoCE traffic is in the lossless
class (priority 3), half on each rail, with 0 drops, 0 retransmissions and 0 timeouts.

Files: [`network/crs812-fabric.rsc`](network/crs812-fabric.rsc) (the switch, from its factory state, in order),
[`network/crs812-export.rsc`](network/crs812-export.rsc) (the measured switch's whole configuration, for reference),
[`network/51-cx7-roce-static.yaml`](network/51-cx7-roce-static.yaml) (each Spark's addresses),
[`network/roce-pfc.service`](network/roce-pfc.service) and
[`network/install-roce-pfc.sh`](network/install-roce-pfc.sh) (each Spark's flow control, at every boot).
Each Spark is configured as section 4 describes; the switch as section 3 does.

## 1. What the engine needs from the network

- **Two rails per Spark.** Each Spark has one 200G cable into its ConnectX-7's first QSFP port. The card reaches the GB10
  processor through two PCIe Gen5 x4 links, so that one port appears as two network devices ("rails") of about
  100 Gb/s each: rail A `enp1s0f0np0` (RDMA device `mlx5_0`) and rail B `enP2p1s0f0np0` (`mlx5_2`). The engine's ring uses both rails (its fleet file names one RDMA device per rail) and sends half of its traffic on each.
- **RoCE v2**, MTU 9000 end to end (the switch's L2 MTU 9216), one subnet per rail, every port forwarded in hardware.
- **Priority flow control (PFC) is recommended.** Without it RoCE still runs but drops packets under overload and recovers
  by retransmitting: 1 to 2 drops per million packets under inference, and 1.45 million drops in 20 s of a 3:1 incast
  (section 6). With PFC on one traffic class, the same tests show 0 drops.

## 2. Your own switch: what to configure

A switch that does the following should behave the same way (MikroTik's own commands are in section 3):

1. Four 200G ports (or 400G ports split into two 200G ports), one Spark on each, linked at 200 Gb/s with RS-FEC, the error
   correction that 200G over copper needs; auto-negotiation selects it.
2. One untagged layer-2 VLAN that holds exactly the four ports and is forwarded in hardware, not by the switch's CPU,
   which caps near 10 Gb/s. Jumbo frames: L2 MTU 9216, IP MTU 9000.
3. Trust each packet's DSCP. Map **DSCP 0** into one traffic class, here **3**; the engine sends all its RoCE traffic
   unmarked, so it arrives as DSCP 0. Map **DSCP 26**, the RoCE convention for programs that mark their traffic, into the
   same class. Map congestion notifications (DSCP 48) into a strict-priority class, here 6.
4. **PFC on that class, both directions** (send and obey pause frames), with the class in the switch's no-drop buffer
   pool. Leave the switch's global 802.3x pause off (section 7).
5. Management on its own port, never in the fabric's VLAN.

The engine has no setting to mark its packets, so mapping DSCP 0 into the PFC class is what keeps its packets from being
dropped. Keep other traffic off the four ports' VLAN.

## 3. The MikroTik CRS812 measured here, from its factory state

| | |
|---|---|
| Switch | MikroTik CRS812-8DS-2DQ-2DDQ, chip Marvell 98DX7335, RouterOS 7.25 |
| Sparks | four GB10 (Dell Pro Max GB10), DGX OS 7.6, kernel 7.0.0-1019-nvidia, ConnectX-7 firmware 28.50.1002, rdma-core 2510.0.12 (NVIDIA's DOCA repository) |
| Cables | two 400G QSFP-DD to 2x 200G QSFP56 breakout cables (2 m, copper), one per QSFP-DD cage |

In this document the four Sparks are called node1 to node4: node1 is the first Spark (rank 0, the one that serves the
API); node2 to node4 are the workers.

| Spark | switch port (renamed) | breakout lane | rail A | rail B |
|---|---|---|---|---|
| node1 | node1 | qsfp56-dd-2-5 | 10.255.0.1/24 | 10.255.1.1/24 |
| node2 | node2 | qsfp56-dd-2-1 | 10.255.0.2/24 | 10.255.1.2/24 |
| node3 | node3 | qsfp56-dd-1-5 | 10.255.0.3/24 | 10.255.1.3/24 |
| node4 | node4 | qsfp56-dd-1-1 | 10.255.0.4/24 | 10.255.1.4/24 |

These are example addresses; use your own subnets, one per rail. Other addresses work if the fleet file names them (`NEXT_IP<d>` and `PREV_IP<d>`; a fleet file for one switch is `deploy/fleet.conf.example`, the ring file is
`ds41f-4xgx10-switchless/fleet/homelab.conf`). Use one subnet per rail, so that every RoCE path has one route and no two interfaces of a
host share a subnet; shared subnets cause ARP flux, which drops RDMA packets.

Log in over ssh (`ssh admin@<switch>`) and save a restore point (`/export file=before-fabric`). With no traffic on the
fabric, run the commands in [`network/crs812-fabric.rsc`](network/crs812-fabric.rsc) in order, either pasted or with
`/import crs812-fabric.rsc` after uploading the file. The parts:

- **Management:** uses `ether1` alone (interface list `MGMT`), with ftp, telnet, www and the API turned off. `ether1` is
  never in the fabric's bridge, so a fabric link restart never cuts your ssh session.
- **The four ports:** the script sets each QSFP-DD cage to 2x 200G breakout; the 200G lanes are `-1` and `-5` of each
  cage. Auto-negotiation stays on, advertising only 200G-baseCR4 (it negotiates RS-FEC). Changing the L2 MTU resets the
  switch chip, so every link drops briefly. Check: `/interface ethernet monitor node1,node2,node3,node4 once` gives `status: link-ok`
  and `rate: 200Gbps`.
- **One bridge, VLAN 200:** `/interface bridge port print` must show the `H` (hardware offload) flag on all four ports.
  Use one bridge per switch chip: a second bridge, or unused breakout lanes left in another bridge, stops the hardware
  offload.
- **QoS for RoCE with PFC:** MikroTik's own RoCE example (the Quality of Service manual, RouterOS 7.17 and later), with
  ECN left off and one addition, DSCP 0 into class 3. The settings are:
  - classes 1 and 3 share bandwidth equally, and class 6 is strict priority;
  - `pfc-tc3` sends and obeys pause frames for class 3 (thresholds `auto`);
  - `lossless-traffic-class=3` gives class 3 the no-drop buffer pool;
  - the ports trust DSCP;
  - PFC's receive side needs the queue's egress rate set, even at wire speed.

  RouterOS 7.23 and later create the DSCP maps for profiles by themselves; DSCP 0 needs its own map line. Changing
  tx-manager queues briefly pauses forwarding.

[`network/crs812-export.rsc`](network/crs812-export.rsc) is the whole
configuration of the measured switch (`/export terse`, with the serial number and keys removed). RouterOS writes it in its own
order, so build from `crs812-fabric.rsc` and compare against the export.

## 4. Each Spark

**Addresses and MTU** (persistent, with netplan): [`network/51-cx7-roce-static.yaml`](network/51-cx7-roce-static.yaml)
as `/etc/netplan/51-cx7-roce-static.yaml` (mode 600), with N set to the last number of this Spark's addresses (1 to 4 with
the defaults); then `sudo netplan generate && sudo netplan apply`. NetworkManager on a
desktop install deletes fabric addresses added by hand, so use netplan. The DGX OS defaults protect against ARP flux
(`arp_ignore=1`, `arp_announce=2`, `rp_filter=2`). Check that a full-size ping with the don't-fragment bit passes between every pair of Sparks on both
rails: `ping -M do -s 8972 10.255.0.<peer>` and `ping -M do -s 8972 10.255.1.<peer>`.

**RDMA:** with NVIDIA's rdma-core the devices are `mlx5_0` and `mlx5_2` (Ubuntu's rdma-core names them `rocep1s0f0` and
`roceP2p1s0f0`). `rdma link show`: both `ACTIVE`, `LINK_UP`. The engine's fleet file names each rail's device
(`NEXT_DEV0` and `PREV_DEV0` = `mlx5_0`, `NEXT_DEV1` and `PREV_DEV1` = `mlx5_2`) and the GID index (`GID_INDEX=3`: on the
Sparks measured here index 3 is RoCE v2 over IPv4; check with `cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/3`,
which gives `RoCE v2`).
Both PCIe functions should be at x4 (`/sys/bus/pci/devices/<bdf>/current_link_width`); after a cold boot a PCIe function can
come up at x2. Check the width, and if it is x2, do a warm reboot.

**Flow control on the card** (DSCP 0 and 26 to priority 3, PFC on priority 3), on both rails:
```
sudo dcb app add dev enp1s0f0np0 dscp-prio 0:3 26:3
sudo dcb app add dev enP2p1s0f0np0 dscp-prio 0:3 26:3
sudo dcb pfc set dev enp1s0f0np0 prio-pfc 3:on
sudo dcb pfc set dev enP2p1s0f0np0 prio-pfc 3:on
```
The first DSCP entry switches the card to DSCP trust (`mlnx_qos -i enp1s0f0np0`: `Priority trust state: dscp`,
`PFC configuration: enabled 0 0 0 1 0 0 0 0`). Enabling PFC turns the card's 802.3x pause off (`ethtool -a`: RX off, TX
off), and setting PFC can restart the link for a few seconds. Check: `dcb pfc show dev <rail>` gives `3:on`, and
`dcb app show dev <rail>` gives `dscp-prio 0:3 AF31:3` (AF31 is DSCP 26).

**A reboot or a driver reload clears these settings.** [`network/roce-pfc.service`](network/roce-pfc.service)
sets them again at every boot (`dcb app replace` is safe to run repeatedly). Install it on every Spark with
`network/install-roce-pfc.sh --start user@spark1 user@spark2 user@spark3 user@spark4` (it needs passwordless sudo),
or by hand: `sudo install -m 644 roce-pfc.service /etc/systemd/system/ && sudo systemctl daemon-reload &&
sudo systemctl enable --now roce-pfc.service`. After a driver reload, run `sudo systemctl restart
roce-pfc.service`. After rebooting each Spark, the unit ran at boot, the settings were present and the
overload test gave 0 drops and 0 resends at 195.7 and 195.9 Gb/s.

## 5. The engine

- The ring connects over RoCE v2 (the fleet file's `GID_INDEX`) and sets no traffic class, so every packet carries
  DSCP 0. With DSCP 0 in the PFC class on the switch and on the cards, its traffic is lossless with no engine setting.
- Its RDMA MTU is 4096 bytes, which the fabric's MTU 9000 carries whole.
- With all four Sparks on one switch, `DSGB10_RING_DIRECT=2` in the engine's arguments sends every collective to its
  peer in one hop (README-pico.md); the ring's own test (`engine/comm/run4.sh` with `ring_test`) checks it.
- The engine does not use NCCL. If you run NCCL yourself (for the check in section 6), name both rails:
  `NCCL_IB_HCA=mlx5_0,mlx5_2`, with `NCCL_SOCKET_IFNAME=enp1s0f0np0`, `NCCL_CROSS_NIC=1` and
  `NCCL_IB_SUBNET_AWARE_ROUTING=1`.

## 6. Checking it

| check | how | expected |
|---|---|---|
| switch links | `/interface ethernet monitor node1,node2,node3,node4 once` | link-ok, 200Gbps |
| full rate, no congestion | perftest on each rail, each Spark writing to the next: on the receiver `ib_write_bw -d mlx5_0 -x 3 -s 1048576 --report_gbits -D 10 -F`, on the sender the same plus the receiver's rail A address (and `-d mlx5_2` with the rail B address) | ~98 Gb/s per flow, 0 pause frames |
| overload | three Sparks write into one on both rails (six flows of the command above, each on its own port `-p`, 20 s, the engine stopped) | switch drops 0; the Sparks' `out_of_sequence` and `packet_seq_err` unchanged; ~196 Gb/s into the receiving Spark; the switch pauses the senders, so their `rx_prio3_pause` counters rise (about 4.25 million each in 20 s here) |
| NCCL (optional) | nccl-tests `all_reduce_perf` over the four, both rails | ~24 GB/s bus bandwidth at 16 GB (measured here) |
| the real workload | the counters below, before and after a period of serving | 0 drops, 0 retransmissions, 0 timeouts |

The counters: on the switch, `/interface ethernet print stats` (tx-drop-packet, tx-drop-queueN-packet, tx-pause,
rx-pause; RouterOS prints them transposed, one column per port, with spaces as thousands separators) and
`/interface ethernet switch qos port print pfc` (per class); on each Spark,
`/sys/class/infiniband/mlx5_{0,2}/ports/1/hw_counters/` (`out_of_sequence`, `packet_seq_err`, `local_ack_timeout_err`)
and `ethtool -S enp1s0f0np0` (`rx_prio3_pause`, `tx_prio3_pause`, `rx_discards_phy`).

**Measured here** (the overload test above, 20 s):

| state | overload: switch drops | RoCE retransmissions | into the one Spark |
|---|---|---|---|
| factory (no flow control, no QoS) | 1,454,393 | 136,495 | 159.8 Gb/s |
| 802.3x flow control on | 2,275,830 | 137,086 | 160 Gb/s |
| ECN on the default queue | 710,206 | 5,039 | 196 Gb/s |
| **PFC on class 3 (this document)** | **0** | **0** | **195.9 Gb/s** |

## 7. What does not work on this switch, and why

- **802.3x flow control** (`rx-flow-control` / `tx-flow-control`). With QoS hardware offloading on, which this switch
  generation forces, the switch never sends pause frames to senders. It obeys pause frames from a receiver but cannot send its own, which makes
  drops worse (2.28 million against 1.45 million). MikroTik's forum (thread 273446) reports the same on a CRS310.
  Changing the setting also renegotiates each link (8-20 s down).
- **ECN marking on the default queue alone** cut overload drops by about half and RoCE retransmissions by more than 95 %,
  because the ConnectX-7 marks RoCE as ECN-capable and runs DCQCN by default, so marking slows the senders. With ECN marking
  alone on the default queue, a 15-minute run still dropped 1,340 packets (measured on an engine without PFC, not this repository's).

## 8. Rolling it back

- **Switch:** `/interface ethernet switch qos port set node1,node2,node3,node4 trust-l3=ignore pfc=disabled`,
  `/interface ethernet switch qos map ip remove [find dscp=0]`, `/interface ethernet switch qos settings set
  lossless-traffic-class=auto`; then remove the profiles `roce` and `cnp` and the PFC profile `pfc-tc3`, reset `egress-rate-queue3` on the four
  ports and the tx-manager queue schedules of classes 1, 3 and 6. Or import your restore point, which undoes all of it.
- **Sparks:** `sudo systemctl disable --now roce-pfc.service`, then on each rail `sudo dcb pfc set dev <rail>
  prio-pfc 3:off` and `sudo dcb app del dev <rail> dscp-prio 0:3 26:3`; or disable the unit and reboot.
