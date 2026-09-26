"""Greedy Whisper large-v3 decode (sherpa-onnx ONNX export) in numpy, joining token BYTES
before UTF-8 decoding so Thai characters split across tokens are not lost.
Ported from k2-fsa/sherpa-onnx scripts/whisper/test.py."""
import base64
import glob
import numpy as np
import onnxruntime as ort
import kaldi_native_fbank as knf

MD = glob.glob("/tmp/claude-0/models/sherpa-onnx-whisper-large-v3")[0]


class W:
    def __init__(self):
        so = ort.SessionOptions(); so.inter_op_num_threads = 1; so.intra_op_num_threads = 4
        self.enc = ort.InferenceSession(f"{MD}/large-v3-encoder.int8.onnx", so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(f"{MD}/large-v3-decoder.int8.onnx", so, providers=["CPUExecutionProvider"])
        m = self.enc.get_modelmeta().custom_metadata_map
        g = lambda k: int(m[k])
        self.L, self.ctx, self.st, self.nmels = g("n_text_layer"), g("n_text_ctx"), g("n_text_state"), g("n_mels")
        self.sot, self.eot, self.translate, self.nots, self.nosp, self.blank = (
            g("sot"), g("eot"), g("translate"), g("no_timestamps"), g("no_speech"), g("blank_id"))
        self.sot_seq = list(map(int, m["sot_sequence"].split(","))) + [self.nots]
        self.lang2id = dict(zip(m["all_language_codes"].split(","), map(int, m["all_language_tokens"].split(","))))
        self.table = {}
        for line in open(f"{MD}/large-v3-tokens.txt"):
            t, i = line.split(); self.table[int(i)] = base64.b64decode(t)

    def mel(self, wave):
        o = knf.WhisperFeatureOptions(); o.dim = self.nmels
        fb = knf.OnlineWhisperFbank(o); fb.accept_waveform(16000, wave); fb.input_finished()
        f = np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])
        ls = np.log10(np.clip(f, 1e-10, None)); ls = np.maximum(ls, ls.max() - 8.0); mel = (ls + 4.0) / 4.0
        mel = np.pad(mel, ((0, 1500), (0, 0)))
        if mel.shape[0] > 3000:
            mel = np.pad(mel[:2950], ((0, 50), (0, 0)))
        return mel.T[None].astype(np.float32)

    def run_dec(self, toks, kc, vc, ck, cv, off):
        ins = self.dec.get_inputs(); outs = [o.name for o in self.dec.get_outputs()[:3]]
        return self.dec.run(outs, {ins[0].name: toks, ins[1].name: kc, ins[2].name: vc,
                                   ins[3].name: ck, ins[4].name: cv, ins[5].name: off})

    def transcribe(self, wave, lang="th"):
        ck, cv = self.enc.run(None, {self.enc.get_inputs()[0].name: self.mel(wave)})[:2]
        seq = list(self.sot_seq); seq[1] = self.lang2id[lang]
        kc = np.zeros((self.L, 1, self.ctx, self.st), np.float32); vc = kc.copy()
        off = np.zeros(1, np.int64)
        logits, kc, vc = self.run_dec(np.array([seq], np.int64), kc, vc, ck, cv, off)
        off += len(seq)
        out = []
        for step in range(self.ctx):
            lg = logits[0, -1].copy()
            if step == 0:
                lg[self.eot] = lg[self.blank] = -np.inf
            for t in (self.nots, self.sot, self.nosp, self.translate):
                lg[t] = -np.inf
            nxt = int(lg.argmax())
            if nxt == self.eot or len(out) > 200:
                break
            out.append(nxt)
            logits, kc, vc = self.run_dec(np.array([[nxt]], np.int64), kc, vc, ck, cv, off)
            off += 1
        return b"".join(self.table.get(i, b"") for i in out).decode("utf-8", errors="replace").strip()
