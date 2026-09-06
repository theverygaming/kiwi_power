# nix-shell -p python314 -p python314Packages.websockets -p python314Packages.numpy
import argparse
import datetime
import re
import struct
import enum
import csv
import asyncio
import websockets
import numpy as np

class KiwiWSPacketType(enum.Enum):
    MSG = 1
    WATERFALL = 2

def parse_kiwi_ws_packet(msg):
    id4 = msg[:4]
    data = msg[4:]
    match id4:
        case b"MSG ":
            return (KiwiWSPacketType.MSG, {m.group("key"): m.group("value") for m in re.finditer(r"(?P<key>[^= ]+)(?:=(?P<value>[^ ]+))?", data.decode("utf-8"))})
        case b"W/F ":
            # https://github.com/jks-prv/KiwiSDR/blob/c40ecb471dced33689e335689f8ffd35a54f47fa/rx/rx_waterfall.h#L163-L181
            wfheader = data[:12]
            wfdata = data[12:]
            x_bin_server, flags_x_zoom_server, seq = struct.unpack("<III", wfheader)
            flags = (flags_x_zoom_server >> 16)
            has_compression = (flags & 1) == 1
            if has_compression:
                raise Exception("cannot handle compression")
            return (KiwiWSPacketType.WATERFALL, {
                "bin_start": x_bin_server,
                "zoom": flags_x_zoom_server & 0xFFFF,
                "seq": seq,
                "data_raw": np.frombuffer(wfdata, dtype=np.uint8),
            })
        case _:
            raise Exception(f"unknown ws packet id4 '{id4}'")

_kiwi_cfg = {}

async def get_kiwi_wf_lines(websocket, bin_start, zoom, nlines, nlines_discard, timeout):
    if nlines <= 0:
        return []
    lines = []
    try:
        await websocket.send(f"SET zoom={zoom} start={bin_start}")
        async with asyncio.timeout(timeout):
            state = 0
            while True:
                if len(lines) == nlines:
                    break
                t, data = parse_kiwi_ws_packet(await websocket.recv())
                match t:
                    case KiwiWSPacketType.MSG:
                        if state == 0 and "zoom" in data and "start" in data:
                            if int(data["zoom"]) == zoom and int(data["start"]) == bin_start:
                                state = 1
                    case KiwiWSPacketType.WATERFALL:
                        if state == 1 and data["zoom"] == zoom and data["bin_start"] == bin_start:
                            if nlines_discard > 0:
                                nlines_discard -= 1
                                continue
                            # https://github.com/jks-prv/KiwiSDR/blob/c40ecb471dced33689e335689f8ffd35a54f47fa/rx/rx_waterfall.cpp#L1328-L1331
                            dbm_min = -200
                            dbm_max = 0
                            lines.append(dbm_min + (data["data_raw"].astype(np.float32) * (dbm_max - dbm_min) / 255))
    except TimeoutError as e:
        raise Exception("timeout while waiting for kiwi to ackowledge commands") from e
    return lines

async def main(
    cfg_url,
    cfg_nlines,
    cfg_nlines_discard,
    cfg_zoom,
    cfg_nbins_discard_start,
    cfg_nbins_discard_end,
    cfg_fstart,
    cfg_fend,
    cfg_outfile,
):
    match = re.match(r"^http(?P<secure>s)?:\/\/(?P<host>[^\/:]+)(?::(?P<port>[0-9]+))?(?:\/.*)?$", cfg_url)
    if match is None:
        raise Exception("could not match URL")
    secure = match.group("secure") or ""
    host = match.group("host")
    port = int(match.group("port") or (80 if not secure else 443))
    ws_url = f"ws{secure}://{host}:{port}"

    while True:
        try:
            async with websockets.connect(
                f"{ws_url}/ws/kiwi/{int(datetime.datetime.now().timestamp())}/W/F",
            ) as websocket:
                for msg in [
                    "SET auth t=kiwi p=#",
                    "SET ident_user=kiwi_power",
                    "SET zoom=0 start=0",
                    "SET maxdb=0 mindb=-100", # does not have an effect on the waterfall as far as I can see, probably for automatic calibration
                    "SET interp=13", # starts at 10, https://github.com/jks-prv/KiwiSDR/blob/c40ecb471dced33689e335689f8ffd35a54f47fa/rx/rx_waterfall.h#L208
                    "SET window_func=2",
                    "SET wf_speed=4",
                    "SET wf_comp=0", # we don't support any compression for now
                    "SET aper=0 algo=3 param=0.00", # aperture 0 -> manual, algo 3 -> off, param as far as I can see only applies to auto
                ]:
                    await websocket.send(msg)
                try:
                    async with asyncio.timeout(10):
                        while True:
                            t, data = parse_kiwi_ws_packet(await websocket.recv())
                            if t != KiwiWSPacketType.MSG:
                                continue

                            # grab relevant config values
                            if "wf_setup" in data:
                                _kiwi_cfg["wf_fft_size"] = int(data["wf_fft_size"])
                                _kiwi_cfg["zoom_max"] = int(data["zoom_max"])
                            if "bandwidth" in data:
                                _kiwi_cfg["bandwidth"] = int(data["bandwidth"])

                            # do we have all the config values?
                            if "wf_fft_size" in _kiwi_cfg and "bandwidth" in _kiwi_cfg:
                                break
                except TimeoutError as e:
                    raise Exception("timeout while waiting for kiwi to send WF config parameters") from e
                nbins = _kiwi_cfg["wf_fft_size"] << _kiwi_cfg["zoom_max"]
                span = _kiwi_cfg["bandwidth"] / (1 << cfg_zoom) # https://github.com/jks-prv/KiwiSDR/blob/c40ecb471dced33689e335689f8ffd35a54f47fa/rx/rx_waterfall_cmd.cpp#L145
                span_bins = int(nbins / (1 << cfg_zoom))
                hzperbin = _kiwi_cfg["bandwidth"] / nbins # https://github.com/jks-prv/KiwiSDR/blob/c40ecb471dced33689e335689f8ffd35a54f47fa/rx/rx_waterfall.cpp#L320
                def bin_to_freq(fftbin):
                    return fftbin * hzperbin
                def freq_to_bin(f):
                    return round((f / _kiwi_cfg["bandwidth"]) * nbins)
                with open(cfg_outfile, "a", encoding="utf-8", newline="") as f:
                    csvwriter = csv.writer(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                    date_str = None
                    time_str = None
                    binstartnow = -1
                    print("ready")
                    while True:
                        await websocket.send("SET keepalive")
                        if bin_to_freq(binstartnow) >= cfg_fend or binstartnow < 0 or binstartnow >= nbins:
                            binstartnow = freq_to_bin(cfg_fstart)
                            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                            time_str = datetime.datetime.now().strftime("%H:%M:%S")

                        line_avg = np.mean(await get_kiwi_wf_lines(websocket, int(binstartnow), cfg_zoom, cfg_nlines, cfg_nlines_discard, 5), axis=0)[cfg_nbins_discard_start:-cfg_nbins_discard_end or None]

                        csvwriter.writerow([
                            date_str, # date
                            time_str, # time
                            int(bin_to_freq(int(binstartnow)) + (span * (cfg_nbins_discard_start / _kiwi_cfg["wf_fft_size"]))), # Hz low # FIXME: research what the right calculation for this value would be lmao
                            int(bin_to_freq(int(binstartnow)) + (span * (cfg_nbins_discard_start / _kiwi_cfg["wf_fft_size"])) + (span * (len(line_avg) / _kiwi_cfg["wf_fft_size"]))), # Hz high # FIXME: research what the right calculation for this value would be lmao
                            span / _kiwi_cfg["wf_fft_size"], # Hz step
                            1, # samples
                        ] + list(line_avg))
                        binstartnow += span_bins * (len(line_avg) / _kiwi_cfg["wf_fft_size"])
        except (OSError, websockets.exceptions.InvalidHandshake, asyncio.TimeoutError) as e:
            print(e)
        await asyncio.sleep(30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="kiwipower",
    )
    parser.add_argument("--url", help="KiwiSDR URL", type=str, required=True)
    parser.add_argument("--fstart", help="start frequency in Hz", type=float, default=0.0)
    parser.add_argument("--fend", help="end frequency in Hz", type=float, default=30e6)
    parser.add_argument("--zoom", help="zoom level", type=int, required=True)
    parser.add_argument("--out", help="output CSV path", type=str, required=True)
    parser.add_argument("--nlinesavg", help="number of lines to average", type=int, default=1)
    parser.add_argument("--nlinesdiscard", help="number of lines to skip at the start", type=int, default=10)
    parser.add_argument("--nbinsdiscardstart", help="number of bins to skip at the start", type=int, default=1)
    parser.add_argument("--nbinsdiscardend", help="number of bins to skip at the end", type=int, default=0)
    parsed_args = parser.parse_args()

    asyncio.run(main(
        cfg_url=parsed_args.url,
        cfg_nlines=parsed_args.nlinesavg,
        cfg_nlines_discard=parsed_args.nlinesdiscard,
        cfg_zoom=parsed_args.zoom,
        cfg_nbins_discard_start=parsed_args.nbinsdiscardstart,
        cfg_nbins_discard_end=parsed_args.nbinsdiscardend,
        cfg_fstart=parsed_args.fstart,
        cfg_fend=parsed_args.fend,
        cfg_outfile=parsed_args.out,
    ))
