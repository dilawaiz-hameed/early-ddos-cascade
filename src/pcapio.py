"""Minimal, dependency-free reader for pcap and pcapng files.

Yields parsed IPv4 packets as tuples so that flows can be assembled
without scapy/dpkt (neither is installable in the build environment).
Supported link types: Ethernet (1, with 802.1Q VLAN tags), raw IP (101, 228),
Linux cooked capture SLL (113) and SLL2 (276).
"""
import struct

ETH, RAW, RAW4, SLL, SLL2 = 1, 101, 228, 113, 276


class Pkt:
    __slots__ = ("ts", "src", "dst", "proto", "ip_len", "ip_hl", "ttl", "frag",
                 "sport", "dport", "flags", "win", "l4_hl", "payload")

    def __init__(self, ts, src, dst, proto, ip_len, ip_hl, ttl, frag,
                 sport, dport, flags, win, l4_hl, payload):
        self.ts, self.src, self.dst, self.proto = ts, src, dst, proto
        self.ip_len, self.ip_hl, self.ttl, self.frag = ip_len, ip_hl, ttl, frag
        self.sport, self.dport, self.flags, self.win = sport, dport, flags, win
        self.l4_hl, self.payload = l4_hl, payload


def _ip_offset(buf, linktype):
    """Return the byte offset of the IPv4 header, or -1 if not IPv4."""
    if linktype == ETH:
        if len(buf) < 14:
            return -1
        off, et = 12, struct.unpack_from("!H", buf, 12)[0]
        while et in (0x8100, 0x88A8):          # VLAN / QinQ
            off += 4
            if len(buf) < off + 2:
                return -1
            et = struct.unpack_from("!H", buf, off)[0]
        return off + 2 if et == 0x0800 else -1
    if linktype in (RAW, RAW4):
        return 0 if buf and (buf[0] >> 4) == 4 else -1
    if linktype == SLL:
        return 16 if len(buf) >= 16 and struct.unpack_from("!H", buf, 14)[0] == 0x0800 else -1
    if linktype == SLL2:
        return 20 if len(buf) >= 20 and struct.unpack_from("!H", buf, 0)[0] == 0x0800 else -1
    return -1


def parse_ipv4(buf, linktype, ts):
    off = _ip_offset(buf, linktype)
    if off < 0 or len(buf) < off + 20:
        return None
    vihl, _, tot_len, _, frag_field, ttl, proto = struct.unpack_from("!BBHHHBB", buf, off)
    ihl = (vihl & 0x0F) * 4
    if (vihl >> 4) != 4 or ihl < 20:
        return None
    src = buf[off + 12:off + 16]
    dst = buf[off + 16:off + 20]
    src = "%d.%d.%d.%d" % tuple(src)
    dst = "%d.%d.%d.%d" % tuple(dst)
    more_frag = (frag_field >> 13) & 1
    frag_off = frag_field & 0x1FFF
    frag = 1 if (more_frag or frag_off) else 0
    l4 = off + ihl
    sport = dport = flags = win = l4_hl = 0
    if frag_off == 0:                         # only the first fragment carries the L4 header
        if proto == 6 and len(buf) >= l4 + 14:
            sport, dport = struct.unpack_from("!HH", buf, l4)
            l4_hl = (buf[l4 + 12] >> 4) * 4
            flags = buf[l4 + 13]
            win = struct.unpack_from("!H", buf, l4 + 14)[0] if len(buf) >= l4 + 16 else 0
        elif proto == 17 and len(buf) >= l4 + 4:
            sport, dport = struct.unpack_from("!HH", buf, l4)
            l4_hl = 8
        elif proto == 1 and len(buf) >= l4 + 1:
            sport = buf[l4]                    # ICMP type stored in the port slot
            l4_hl = 8
    payload = max(0, tot_len - ihl - l4_hl)
    return Pkt(ts, src, dst, proto, tot_len, ihl, ttl, frag, sport, dport, flags, win, l4_hl, payload)


def _read_pcap(f, magic):
    endian = "<" if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
    nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    hdr = f.read(20)
    linktype = struct.unpack(endian + "HHiIII", hdr)[5] & 0x0FFFFFFF
    div = 1e9 if nano else 1e6
    rec = struct.Struct(endian + "IIII")
    while True:
        h = f.read(16)
        if len(h) < 16:
            return
        sec, frac, incl, _ = rec.unpack(h)
        data = f.read(incl)
        if len(data) < incl:
            return
        yield sec + frac / div, linktype, data


def _read_pcapng(f):
    ifaces = []                               # (linktype, ts_divisor)
    endian = "<"
    while True:
        head = f.read(8)
        if len(head) < 8:
            return
        btype = struct.unpack(endian + "I", head[:4])[0]
        if btype == 0x0A0D0D0A:                # section header: re-detect byte order
            bom = f.read(4)
            endian = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            blen = struct.unpack(endian + "I", head[4:8])[0]
            f.read(blen - 12)
            ifaces = []
            continue
        blen = struct.unpack(endian + "I", head[4:8])[0]
        body = f.read(blen - 8)
        if len(body) < blen - 8:
            return
        if btype == 1:                         # interface description
            lt = struct.unpack_from(endian + "H", body, 0)[0]
            div = 1e6
            p = 8
            while p + 4 <= len(body) - 4:
                code, olen = struct.unpack_from(endian + "HH", body, p)
                if code == 0:
                    break
                if code == 9 and olen >= 1:    # if_tsresol
                    r = body[p + 4]
                    div = float(2 ** (r & 0x7F)) if r & 0x80 else float(10 ** r)
                p += 4 + ((olen + 3) & ~3)
            ifaces.append((lt, div))
        elif btype == 6:                       # enhanced packet
            iid, hi, lo, cap = struct.unpack_from(endian + "IIII", body, 0)
            lt, div = ifaces[iid] if iid < len(ifaces) else (ETH, 1e6)
            yield ((hi << 32) | lo) / div, lt, body[20:20 + cap]
        elif btype == 3:                       # simple packet (no timestamp)
            lt, div = ifaces[0] if ifaces else (ETH, 1e6)
            yield 0.0, lt, body[4:]


def read_packets(path):
    """Yield Pkt objects for every IPv4 packet in a pcap/pcapng file."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic == b"\x0a\x0d\x0d\x0a":
            f.seek(0)
            gen = _read_pcapng(f)
        elif magic in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"):
            gen = _read_pcap(f, magic)
        else:
            raise ValueError("not a pcap/pcapng file: %s" % path)
        for ts, lt, data in gen:
            p = parse_ipv4(data, lt, ts)
            if p is not None:
                yield p
