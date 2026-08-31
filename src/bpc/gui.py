"""Tkinter batch window with a manual review mode.

Two screens.  The *batch* screen runs a folder and logs OK / SKIPPED / ERROR.
The *review* screen opens one image and lets the decision be overridden by hand:
sliders for roll, pitch and focal length, and a clickable overlay for striking
out the lines the detector should not have trusted.

Anything that is not toolkit plumbing lives in :mod:`bpc.review`, which has no
Tkinter dependency and is covered by the test suite; this file is the shell.

Tkinter ships with the python.org installer on Windows, which is the target
platform.  On Linux it may need a separate package (``python3-tk``); the CLI
works without it.
"""
from __future__ import annotations

import math
import os
import queue
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .config import Settings
from .imageio import READABLE
from .pipeline import ERROR, OK, SKIPPED, process
from .review import AUTO, MANUAL, ReviewSession

STATUS_COLOUR = {OK: "#1a7f37", SKIPPED: "#9a6700", ERROR: "#b62324"}


def _to_photo(bgr, box):
    """BGR array -> PhotoImage, letterboxed into ``box`` = (w, h)."""
    bw, bh = max(box[0], 1), max(box[1], 1)
    h, w = bgr.shape[:2]
    s = min(bw / w, bh / h, 4.0)
    if s <= 0:
        s = 1.0
    out = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb)), s


# ==========================================================================
# review window
# ==========================================================================
class ReviewWindow(tk.Toplevel):
    def __init__(self, master, path, settings, dest_path, on_saved=None):
        super().__init__(master)
        self.title(f"Review - {os.path.basename(path)}")
        self.geometry("1280x820")
        self.settings = settings
        self.dest_path = dest_path
        self.on_saved = on_saved
        self._busy = False
        self._before_scale = 1.0

        try:
            self.session = ReviewSession(path, settings)
        except Exception as exc:
            messagebox.showerror("Review", f"cannot open image:\n{exc}", parent=master)
            self.destroy()
            return

        self._build()
        self.after(60, self._sync_from_session)

    # -- layout ----------------------------------------------------------
    def _build(self):
        top = ttk.Frame(self, padding=6)
        top.pack(fill="both", expand=True)

        panes = ttk.Frame(top)
        panes.pack(fill="both", expand=True)
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)
        panes.rowconfigure(1, weight=1)

        ttk.Label(panes, text="before  (click a line to strike it out / bring it back)"
                  ).grid(row=0, column=0, sticky="w")
        ttk.Label(panes, text="after").grid(row=1 - 1, column=1, sticky="w")

        self.c_before = tk.Canvas(panes, bg="#1e1e1e", highlightthickness=0)
        self.c_before.grid(row=1, column=0, sticky="nsew", padx=(0, 3))
        self.c_after = tk.Canvas(panes, bg="#1e1e1e", highlightthickness=0)
        self.c_after.grid(row=1, column=1, sticky="nsew", padx=(3, 0))
        self.c_before.bind("<Button-1>", self._on_click_before)
        for c in (self.c_before, self.c_after):
            c.bind("<Configure>", lambda e: self._schedule_redraw())

        self.status = tk.Text(top, height=4, wrap="word", relief="flat",
                              background="#f4f4f4")
        self.status.pack(fill="x", pady=(6, 4))
        self.status.configure(state="disabled")

        ctl = ttk.LabelFrame(top, text="manual correction", padding=6)
        ctl.pack(fill="x")
        self.v_roll = tk.DoubleVar(value=0.0)
        self.v_pitch = tk.DoubleVar(value=0.0)
        self.v_focal = tk.DoubleVar(value=28.0)
        self._slider(ctl, 0, "roll (level)", self.v_roll, -20, 20, "deg")
        self._slider(ctl, 1, "pitch (verticals)", self.v_pitch, -30, 30, "deg")
        self._slider(ctl, 2, "focal length", self.v_focal, 8, 200, "mm eq")

        btns = ttk.Frame(top, padding=(0, 8))
        btns.pack(fill="x")
        ttk.Button(btns, text="strike out slanted lines (>18 deg)",
                   command=self._strike_slanted).pack(side="left")
        ttk.Button(btns, text="reset to automatic",
                   command=self._reset).pack(side="left", padx=6)
        ttk.Checkbutton(btns, text="show detected lines", command=self._schedule_redraw,
                        variable=self._mk_show()).pack(side="left", padx=12)
        ttk.Button(btns, text="keep original", command=self._keep).pack(side="right")
        ttk.Button(btns, text="save correction", command=self._save).pack(side="right", padx=6)

    def _mk_show(self):
        self.v_show_lines = tk.BooleanVar(value=True)
        return self.v_show_lines

    def _slider(self, parent, row, label, var, lo, hi, unit):
        ttk.Label(parent, text=label, width=18).grid(row=row, column=0, sticky="w")
        sc = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal",
                       command=lambda _v: self._on_slider())
        sc.grid(row=row, column=1, sticky="ew", padx=6)
        parent.columnconfigure(1, weight=1)
        lbl = ttk.Label(parent, width=12)
        lbl.grid(row=row, column=2, sticky="e")
        setattr(self, f"_lbl_{row}", (lbl, var, unit))

    def _update_slider_labels(self):
        for row in range(3):
            lbl, var, unit = getattr(self, f"_lbl_{row}")
            lbl.configure(text=f"{var.get():+.2f} {unit}" if unit == "deg"
                          else f"{var.get():.0f} {unit}")

    # -- state -----------------------------------------------------------
    def _sync_from_session(self):
        roll, pitch, f, _ = self.session.current_angles()
        from .model import focal_35mm_from_px
        self.v_roll.set(round(math.degrees(roll), 2))
        self.v_pitch.set(round(math.degrees(pitch), 2))
        if f:
            self.v_focal.set(round(focal_35mm_from_px(f, self.session.w, self.session.h), 0))
        self._redraw()

    def _on_slider(self):
        self.session.set_manual(roll_deg=self.v_roll.get(), pitch_deg=self.v_pitch.get(),
                                focal_35mm=self.v_focal.get())
        self._schedule_redraw()

    def _strike_slanted(self):
        n = self.session.disable_lines_by_angle(18.0)
        if self.session.mode == AUTO:
            self._sync_from_session()
        else:
            self._redraw()
        self._set_status_extra(f"struck out {n} slanted candidate(s)")

    def _reset(self):
        self.session.reset_to_auto()
        self._sync_from_session()

    def _on_click_before(self, event):
        idx = self.session.pick_line(event.x - self._before_off[0],
                                     event.y - self._before_off[1],
                                     display_scale=self._before_scale)
        if idx is None:
            return
        self.session.toggle_line(idx)
        if self.session.mode == AUTO:
            self._sync_from_session()
        else:
            self._redraw()

    # -- drawing ---------------------------------------------------------
    def _schedule_redraw(self):
        if self._busy:
            return
        self._busy = True
        self.after(60, self._redraw)

    def _redraw(self):
        self._busy = False
        try:
            box_b = (self.c_before.winfo_width(), self.c_before.winfo_height())
            box_a = (self.c_after.winfo_width(), self.c_after.winfo_height())
            if min(box_b) < 20 or min(box_a) < 20:
                self.after(120, self._redraw)
                return
            before = self.session.render_before(max_edge=max(box_b), 
                                                show_lines=self.v_show_lines.get())
            after = self.session.render_after(max_edge=max(box_a))
            ph_b, s_b = _to_photo(before, box_b)
            ph_a, _ = _to_photo(after, box_a)
            # scale from the *original* image to what is on screen
            self._before_scale = s_b * (before.shape[1] / self.session.w)
            self._before_off = ((box_b[0] - ph_b.width()) // 2,
                                (box_b[1] - ph_b.height()) // 2)
            self.c_before.delete("all")
            self.c_before.create_image(self._before_off[0], self._before_off[1],
                                       anchor="nw", image=ph_b)
            self._ph_b = ph_b
            self.c_after.delete("all")
            self.c_after.create_image((box_a[0] - ph_a.width()) // 2,
                                      (box_a[1] - ph_a.height()) // 2,
                                      anchor="nw", image=ph_a)
            self._ph_a = ph_a
            self._set_status(self.session.status_text())
            self._update_slider_labels()
        except Exception:
            self._set_status("preview failed:\n" + traceback.format_exc(limit=2))

    def _set_status(self, text):
        self.status.configure(state="normal")
        self.status.delete("1.0", "end")
        self.status.insert("1.0", text)
        self.status.configure(state="disabled")

    def _set_status_extra(self, text):
        self.status.configure(state="normal")
        self.status.insert("end", "\n" + text)
        self.status.configure(state="disabled")

    # -- output ----------------------------------------------------------
    def _save(self):
        try:
            self.session.save(self.dest_path)
        except Exception as exc:
            messagebox.showerror("Save", str(exc), parent=self)
            return
        if self.on_saved:
            self.on_saved(self.session.path, self.dest_path)
        self.destroy()

    def _keep(self):
        try:
            from .imageio import copy_through
            if os.path.abspath(self.dest_path) != os.path.abspath(self.session.path):
                copy_through(self.session.path, self.dest_path)
        except Exception as exc:
            messagebox.showerror("Save", str(exc), parent=self)
            return
        self.destroy()


# ==========================================================================
# batch window
# ==========================================================================
class App(tk.Tk):
    def __init__(self, initial=None):
        super().__init__()
        self.title("Batch Perspective Correction")
        self.geometry("1080x720")
        self.queue = queue.Queue()
        self.worker = None
        self.stop_flag = threading.Event()
        self.results = {}
        self._build()
        if initial:
            for item in initial:
                if os.path.isdir(item):
                    self.v_input.set(item)
                    break
        self.after(120, self._pump)

    def _build(self):
        pad = dict(padx=6, pady=4)
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        self.v_input = tk.StringVar()
        self.v_output = tk.StringVar()
        ttk.Label(top, text="input folder").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.v_input, width=70).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(top, text="browse...", command=self._pick_in).grid(row=0, column=2, **pad)
        ttk.Label(top, text="output folder").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.v_output, width=70).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(top, text="browse...", command=self._pick_out).grid(row=1, column=2, **pad)
        top.columnconfigure(1, weight=1)

        opt = ttk.LabelFrame(self, text="settings", padding=8)
        opt.pack(fill="x", padx=8)
        self.v_strength = tk.DoubleVar(value=1.0)
        self.v_conf = tk.DoubleVar(value=Settings.min_confidence)
        self.v_maxpitch = tk.DoubleVar(value=Settings.max_pitch_deg)
        self.v_crop = tk.StringVar(value=Settings.crop)
        self.v_recursive = tk.BooleanVar(value=False)
        self.v_overwrite = tk.BooleanVar(value=False)
        self.v_review = tk.BooleanVar(value=True)
        self._spin(opt, 0, 0, "strength", self.v_strength, 0.0, 1.0, 0.05)
        self._spin(opt, 0, 3, "min confidence", self.v_conf, 0.0, 1.0, 0.05)
        self._spin(opt, 1, 0, "max pitch (deg)", self.v_maxpitch, 0.0, 45.0, 1.0)
        ttk.Label(opt, text="crop").grid(row=1, column=3, sticky="e", padx=4)
        ttk.Combobox(opt, textvariable=self.v_crop, values=["aspect", "inside", "none"],
                     width=8, state="readonly").grid(row=1, column=4, sticky="w")
        ttk.Checkbutton(opt, text="subfolders", variable=self.v_recursive).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(opt, text="overwrite originals", variable=self.v_overwrite).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(opt, text="offer manual review for unclear images",
                        variable=self.v_review).grid(row=2, column=2, columnspan=3, sticky="w")

        bar = ttk.Frame(self, padding=8)
        bar.pack(fill="x")
        self.btn_run = ttk.Button(bar, text="start", command=self._start)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="stop", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="review selected...", command=self._review_selected).pack(side="left", padx=6)
        self.progress = ttk.Progressbar(bar, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)
        self.lbl_count = ttk.Label(bar, text="")
        self.lbl_count.pack(side="right")

        cols = ("status", "file", "roll", "pitch", "conf", "note")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse")
        for c, w in zip(cols, (80, 320, 70, 70, 60, 380)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.tree.bind("<Double-1>", lambda e: self._review_selected())
        for status, colour in STATUS_COLOUR.items():
            self.tree.tag_configure(status, foreground=colour)

    def _spin(self, parent, r, c, label, var, lo, hi, step):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="e", padx=4)
        ttk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=step,
                    width=7).grid(row=r, column=c + 1, sticky="w")

    # -- settings --------------------------------------------------------
    def _settings(self):
        s = Settings()
        s.pitch_strength = s.roll_strength = float(self.v_strength.get())
        s.min_confidence = float(self.v_conf.get())
        s.max_pitch_deg = float(self.v_maxpitch.get())
        s.crop = self.v_crop.get()
        return s

    def _pick_in(self):
        d = filedialog.askdirectory(title="folder with photos")
        if d:
            self.v_input.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="output folder")
        if d:
            self.v_output.set(d)

    # -- run -------------------------------------------------------------
    def _files(self):
        root = self.v_input.get()
        if not root or not os.path.isdir(root):
            return []
        out = []
        if self.v_recursive.get():
            for base, _, names in os.walk(root):
                out += [os.path.join(base, n) for n in sorted(names)
                        if os.path.splitext(n)[1].lower() in READABLE]
        else:
            out = [os.path.join(root, n) for n in sorted(os.listdir(root))
                   if os.path.splitext(n)[1].lower() in READABLE
                   and os.path.isfile(os.path.join(root, n))]
        return out

    def _dest(self, src):
        if self.v_overwrite.get():
            return src
        stem, ext = os.path.splitext(os.path.basename(src))
        out_dir = self.v_output.get() or os.path.dirname(src)
        return os.path.join(out_dir, f"{stem}_corr{ext}")

    def _start(self):
        files = self._files()
        if not files:
            messagebox.showinfo("Batch", "no readable images in that folder")
            return
        if self.v_overwrite.get() and not messagebox.askyesno(
                "Overwrite", f"Replace {len(files)} original file(s)?"):
            return
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        self.progress.configure(maximum=len(files), value=0)
        self.stop_flag.clear()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        settings = self._settings()
        self.worker = threading.Thread(target=self._run, args=(files, settings), daemon=True)
        self.worker.start()

    def _run(self, files, settings):
        for i, src in enumerate(files, 1):
            if self.stop_flag.is_set():
                self.queue.put(("done", "stopped"))
                return
            try:
                r = process(src, self._dest(src), settings)
            except Exception as exc:                  # never let one file kill the run
                self.queue.put(("error", (src, str(exc))))
                continue
            self.queue.put(("row", (i, r)))
        self.queue.put(("done", "finished"))

    def _stop(self):
        self.stop_flag.set()

    def _pump(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "row":
                    i, r = payload
                    self._add_row(r)
                    self.progress.configure(value=i)
                elif kind == "error":
                    src, msg = payload
                    self.tree.insert("", "end", values=(ERROR, os.path.basename(src),
                                                        "", "", "", msg), tags=(ERROR,))
                elif kind == "done":
                    self.btn_run.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    n_skip = sum(1 for r in self.results.values() if r.status == SKIPPED)
                    self.lbl_count.configure(text=f"{payload}: {len(self.results)} file(s)")
                    if n_skip and self.v_review.get():
                        messagebox.showinfo(
                            "Manual review",
                            f"{n_skip} image(s) were left unchanged because the detection "
                            f"was not clear enough.\n\nDouble-click any SKIPPED row to "
                            f"correct it by hand.")
        except queue.Empty:
            pass
        self.after(120, self._pump)

    def _add_row(self, r):
        iid = self.tree.insert("", "end", tags=(r.status,), values=(
            r.status, os.path.basename(r.src),
            f"{r.roll_deg:+.2f}" if r.status == OK else "",
            f"{r.pitch_deg:+.2f}" if r.status == OK else "",
            f"{r.confidence:.2f}",
            r.reason if r.status != OK else
            f"f={r.focal_35mm:.0f}mm ({r.focal_source}), keeps {r.coverage * 100:.0f}%"))
        self.results[iid] = r
        self.tree.see(iid)

    def _review_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Review", "select a row first")
            return
        r = self.results.get(sel[0])
        if r is None:
            return
        ReviewWindow(self, r.src, self._settings(), self._dest(r.src),
                     on_saved=lambda s, d: self._mark_manual(sel[0]))

    def _mark_manual(self, iid):
        vals = list(self.tree.item(iid, "values"))
        vals[0] = OK
        vals[5] = "corrected manually"
        self.tree.item(iid, values=vals, tags=(OK,))


def run(initial=None) -> int:
    App(initial).mainloop()
    return 0
