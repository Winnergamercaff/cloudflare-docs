#!/usr/bin/env python3
"""Render the edit described in edit_timeline.json with FFmpeg.

Usage (from the project root):
    python3 output/render.py                 # final.mp4 (burned subs) + final_nosub.mp4 + subtitles.srt
    python3 output/render.py --preview       # low-res preview_360p.mp4 only
    python3 output/render.py --suffix _v2    # write final_v2.mp4 etc. instead of overwriting
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")


def run(cmd, capture=False):
    print("+", " ".join(cmd[:6]), "...", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=capture, text=True)
    if r.returncode != 0:
        if capture:
            sys.stderr.write(r.stderr[-4000:])
        sys.exit(f"command failed ({r.returncode})")
    return r


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(subs, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(subs, 1):
            f.write(f"{i}\n{srt_time(s['start'])} --> {srt_time(s['end'])}\n{s['text']}\n\n")


def build_graph(tl, scale=None):
    o = tl["output"]
    segs = tl["segments"]
    parts, vlabels, alabels = [], [], []
    inputs = []
    for s in segs:
        if s["source"] not in inputs:
            inputs.append(s["source"])
    for k, s in enumerate(segs):
        idx = inputs.index(s["source"])
        parts.append(
            f"[{idx}:v]trim=start={s['video_in']}:end={s['video_out']},setpts=PTS-STARTPTS,"
            f"fps={o['fps']},scale={o['width']}:{o['height']},setsar=1[v{k}]")
        adur = s["audio_out"] - s["audio_in"]
        parts.append(
            f"[{idx}:a]atrim=start={s['audio_in']}:end={s['audio_out']},asetpts=PTS-STARTPTS,"
            f"aresample={o['sample_rate']},aformat=channel_layouts=stereo,"
            f"volume={s['audio_gain_db']}dB,"
            f"afade=t=in:d=0.015,afade=t=out:st={adur - 0.015:.3f}:d=0.015[a{k}]")
        vlabels.append(f"[v{k}]")
        alabels.append(f"[a{k}]")
    total = sum(s["video_out"] - s["video_in"] for s in segs)
    vchain = (f"{''.join(vlabels)}concat=n={len(segs)}:v=1:a=0,"
              f"fade=t=in:st=0:d={o['fade_in_sec']},"
              f"fade=t=out:st={total - o['fade_out_sec']:.3f}:d={o['fade_out_sec']}")
    if scale:
        vchain += f",scale={scale}"
    parts.append(vchain + ",format=yuv420p[vout]")
    parts.append(
        f"{''.join(alabels)}concat=n={len(segs)}:v=0:a=1,highpass=f=60,"
        # gentle dialog compression + peak limiter so the final loudnorm can stay linear
        f"acompressor=threshold=-22dB:ratio=2.5:attack=10:release=150:makeup=1,"
        f"alimiter=limit=-5dB:attack=3:release=60:level=disabled,"
        f"afade=t=in:st=0:d={o['fade_in_sec']},"
        f"afade=t=out:st={total - o['fade_out_sec']:.3f}:d={o['fade_out_sec']}[aout]")
    return inputs, ";".join(parts), total


def render_master(tl, dst, scale=None, fast=False):
    """Render picture + edited (not yet normalized) audio."""
    inputs, graph, _ = build_graph(tl, scale)
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for i in inputs:
        cmd += ["-i", i]
    cmd += ["-filter_complex", graph, "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "ultrafast" if fast else "slow",
            "-crf", "28" if fast else str(tl["output"]["crf"]),
            "-c:a", "pcm_s16le", dst]
    run(cmd)


def loudnorm(tl, src, dst, burn_ass=None, srt=None):
    o = tl["output"]
    ln = f"loudnorm=I={o['loudness_target_lufs']}:TP={o['true_peak_dbtp']}:LRA=11"
    r = run(["ffmpeg", "-hide_banner", "-i", src, "-af", ln + ":print_format=json",
             "-f", "null", "-"], capture=True)
    m = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr).group(0))
    ln2 = (f"{ln}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
           f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
           f"offset={m['target_offset']}:linear=true,aresample={o['sample_rate']}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src]
    if srt:
        cmd += ["-i", srt]
    if burn_ass:
        fontsdir = os.path.join(OUT, "fonts")
        cmd += ["-vf", f"subtitles={burn_ass}:fontsdir={fontsdir}", "-c:v", "libx264",
                "-preset", "slow", "-crf", str(o["crf"]), "-profile:v", "high", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-af", ln2, "-c:a", o["audio_codec"], "-b:a", o["audio_bitrate"], "-ar", str(o["sample_rate"])]
    if srt:
        cmd += ["-map", "0:v", "-map", "0:a", "-map", "1:s", "-c:s", "mov_text",
                "-metadata:s:s:0", "language=tha"]
    cmd += ["-movflags", "+faststart", dst]
    run(cmd)
    return m


def write_ass(tl, path):
    st = tl["subtitle_style"]
    o = tl["output"]

    def ass_t(t):
        cs = int(round(t * 100))
        h, cs = divmod(cs, 360000)
        m, cs = divmod(cs, 6000)
        s, cs = divmod(cs, 100)
        return f"{h}:{m:02}:{s:02}.{cs:02}"

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {o['width']}", f"PlayResY: {o['height']}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{st['font_name']},{st['font_size']},{st['primary_color']},&H000000FF,"
        f"{st['outline_color']},&H80000000,{-1 if st.get('bold') else 0},0,0,0,100,100,0,0,1,"
        f"{st['outline']},{st['shadow']},"
        f"2,120,120,{st['margin_v']},1", "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for s in tl["subtitles"]:
        lines.append(f"Dialogue: 0,{ass_t(s['start'])},{ass_t(s['end'])},Default,,0,0,0,,{s['text']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    tl = json.load(open(os.path.join(OUT, "edit_timeline.json"), encoding="utf-8"))
    work = os.path.join(OUT, "_work")
    os.makedirs(work, exist_ok=True)
    ass = os.path.join(work, "subs.ass")
    write_ass(tl, ass)
    srt = os.path.join(OUT, f"subtitles{a.suffix}.srt")
    write_srt(tl["subtitles"], srt)

    if a.preview:
        master = os.path.join(work, "preview_master.mkv")
        render_master(tl, master, scale="640:360", fast=True)
        dst = os.path.join(OUT, f"preview_360p{a.suffix}.mp4")
        # scale subtitles along with the picture: libass rescales PlayRes to the frame size
        loudnorm(tl, master, dst, burn_ass=ass)
        print("wrote", dst)
        return

    master = os.path.join(work, "master.mkv")
    render_master(tl, master)
    m = loudnorm(tl, master, os.path.join(OUT, f"final{a.suffix}.mp4"), burn_ass=ass)
    loudnorm(tl, master, os.path.join(OUT, f"final_nosub{a.suffix}.mp4"), srt=srt)
    print("loudnorm pass-1 measurement:", json.dumps(m))
    print("done")


if __name__ == "__main__":
    main()
