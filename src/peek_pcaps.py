"""Quick look inside big CIC-DDoS2019 capture files, to see which attack each file holds.

Usage:  python src/peek_pcaps.py "D:\\Cnet project\\PCAP-03-11" > peek_0311.txt
Reads only the first 100,000 packets of every file (fast), and prints one line per file:
start time, share of packets involving the attacker (172.16.0.5), protocol mix and the
most common UDP/TCP source ports (the reflection service: 123 NTP, 53 DNS, 389 LDAP,
1434 MSSQL, 137 NetBIOS, 161 SNMP, 1900 SSDP, 69 TFTP, 111 PortMap).
"""
import os
import re
import sys
import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcapio import read_packets

ATTACKER = "172.16.0.5"
LIMIT = 100_000


def num(name):
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else -1


def main(folder):
    files = sorted((f for f in os.listdir(folder) if f.startswith("SAT-")), key=num)
    print("file\tstart_utc\tpackets_read\tattacker_share\ttcp/udp/icmp\ttop_ports")
    for f in files:
        n = att = 0
        proto = Counter()
        ports = Counter()
        t0 = None
        try:
            for p in read_packets(os.path.join(folder, f)):
                if t0 is None:
                    t0 = p.ts
                n += 1
                if ATTACKER in (p.src, p.dst):
                    att += 1
                proto[p.proto] += 1
                if p.proto in (6, 17):
                    ports[(("tcp" if p.proto == 6 else "udp"), min(p.sport, p.dport))] += 1
                if n >= LIMIT:
                    break
        except Exception as e:  # truncated or unreadable file
            print(f"{f}\tERROR {e}")
            continue
        start = datetime.datetime.fromtimestamp(t0, datetime.timezone.utc).strftime("%H:%M:%S") if t0 else "-"
        mix = f"{proto[6]/max(n,1):.2f}/{proto[17]/max(n,1):.2f}/{proto[1]/max(n,1):.2f}"
        top = ", ".join(f"{k[0]}:{k[1]} {v/max(n,1):.0%}" for k, v in ports.most_common(3))
        print(f"{f}\t{start}\t{n}\t{att/max(n,1):.2f}\t{mix}\t{top}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
