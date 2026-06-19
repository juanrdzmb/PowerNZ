from __future__ import annotations

import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from io_video import read_video_metadata


ROOT = Path(__file__).resolve().parent
EXERCISES = {
    "Peso muerto": "deadlift",
    "Sentadilla": "squat",
    "Press de banca": "bench",
}
PROFILES = {
    "Equilibrado": "balanced",
    "Máxima precisión": "precision",
    "Máxima velocidad": "fast",
}
PROGRESS_RE = re.compile(r"^PROGRESS\s+(\w+)\s+(\d+)\s+(\d+)")


class PowerNZLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PowerNZ")
        self.geometry("820x720")
        self.minsize(760, 650)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.exercise_label = tk.StringVar(value="Peso muerto")
        self.profile_label = tk.StringVar(value="Equilibrado")
        self.output_format = tk.StringVar(value="portrait-720")
        self.calibration_mode = tk.StringVar(value="auto")
        self.plate_diameter = tk.StringVar(value="")
        self.video_info = tk.StringVar(value="Elige un vídeo para ver sus datos.")
        self.status = tk.StringVar(value="Listo para analizar.")
        self.progress_value = tk.DoubleVar(value=0.0)
        self._controls: list[tk.Widget] = []
        self._advanced_visible = False

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(10, weight=1)

        ttk.Label(root, text="PowerNZ", font=("Segoe UI", 23, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(
            root,
            text="Selecciona el levantamiento y el vídeo. La calibración automática y el perfil equilibrado se ocupan del resto.",
            wraplength=720,
        ).grid(row=1, column=0, sticky="w", pady=(2, 14))

        video_frame = ttk.LabelFrame(root, text="Vídeo", padding=12)
        video_frame.grid(row=2, column=0, sticky="ew")
        video_frame.columnconfigure(0, weight=1)
        entry = ttk.Entry(video_frame, textvariable=self.input_path)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        choose = ttk.Button(video_frame, text="Elegir vídeo", command=self._choose_input)
        choose.grid(row=0, column=1)
        self._controls.extend([entry, choose])
        ttk.Label(video_frame, textvariable=self.video_info, foreground="#666666").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        exercise_frame = ttk.LabelFrame(root, text="Ejercicio", padding=12)
        exercise_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        for index, label in enumerate(EXERCISES):
            button = ttk.Radiobutton(exercise_frame, text=label, value=label, variable=self.exercise_label)
            button.grid(row=0, column=index, sticky="w", padx=(0, 24))
            self._controls.append(button)

        settings_frame = ttk.Frame(root)
        settings_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        settings_frame.columnconfigure(1, weight=1)
        ttk.Label(settings_frame, text="Perfil").grid(row=0, column=0, sticky="w", padx=(0, 8))
        profile = ttk.Combobox(
            settings_frame,
            textvariable=self.profile_label,
            values=tuple(PROFILES),
            state="readonly",
            width=24,
        )
        profile.grid(row=0, column=1, sticky="w")
        advanced = ttk.Button(settings_frame, text="Opciones avanzadas", command=self._toggle_advanced)
        advanced.grid(row=0, column=2, sticky="e")
        self._controls.extend([profile, advanced])

        self.advanced_frame = ttk.LabelFrame(root, text="Opciones avanzadas", padding=12)
        self.advanced_frame.columnconfigure(1, weight=1)
        ttk.Label(self.advanced_frame, text="Calibración").grid(row=0, column=0, sticky="w", padx=(0, 8))
        calibration = ttk.Combobox(
            self.advanced_frame,
            textvariable=self.calibration_mode,
            values=("auto", "manual"),
            state="readonly",
            width=18,
        )
        calibration.grid(row=0, column=1, sticky="w")
        ttk.Label(self.advanced_frame, text="Diámetro del disco (px)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        diameter = ttk.Entry(self.advanced_frame, textvariable=self.plate_diameter, width=18)
        diameter.grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(self.advanced_frame, text="Formato de salida").grid(row=2, column=0, sticky="w", pady=(8, 0))
        output_format = ttk.Combobox(
            self.advanced_frame,
            textvariable=self.output_format,
            values=("portrait-720", "source"),
            state="readonly",
            width=18,
        )
        output_format.grid(row=2, column=1, sticky="w", pady=(8, 0))
        self._controls.extend([calibration, diameter, output_format])

        output_frame = ttk.LabelFrame(root, text="Resultado", padding=12)
        output_frame.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        output_frame.columnconfigure(0, weight=1)
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path)
        output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        save = ttk.Button(output_frame, text="Guardar como", command=self._choose_output)
        save.grid(row=0, column=1)
        self._controls.extend([output_entry, save])

        action_frame = ttk.Frame(root)
        action_frame.grid(row=7, column=0, sticky="ew", pady=(14, 6))
        self.analyze_button = ttk.Button(action_frame, text="Analizar vídeo", command=self._analyze_video)
        self.analyze_button.pack(side="left")
        self.models_button = ttk.Button(action_frame, text="Preparar modelos", command=self._download_models)
        self.models_button.pack(side="left", padx=8)
        open_folder = ttk.Button(action_frame, text="Abrir resultados", command=self._open_output_folder)
        open_folder.pack(side="left")
        self._controls.extend([self.analyze_button, self.models_button, open_folder])

        progress = ttk.Progressbar(root, maximum=100.0, variable=self.progress_value)
        progress.grid(row=8, column=0, sticky="ew", pady=(4, 2))
        ttk.Label(root, textvariable=self.status).grid(row=9, column=0, sticky="w")

        self.log = tk.Text(root, height=12, wrap="word", state="disabled")
        self.log.grid(row=10, column=0, sticky="nsew", pady=(8, 0))

    def _toggle_advanced(self) -> None:
        if self._advanced_visible:
            self.advanced_frame.grid_remove()
        else:
            self.advanced_frame.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        self._advanced_visible = not self._advanced_visible

    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Elegir vídeo",
            filetypes=(("Vídeos", "*.mp4 *.mov *.avi *.mkv"), ("Todos", "*.*")),
        )
        if not path:
            return
        self.input_path.set(path)
        source = Path(path)
        if not self.output_path.get():
            self.output_path.set(str(ROOT / "outputs" / f"{source.stem}_analizado.mp4"))
        try:
            metadata = read_video_metadata(source)
            duration = metadata.frame_count / max(metadata.fps, 1.0)
            self.video_info.set(f"{metadata.width}×{metadata.height} · {metadata.fps:.1f} FPS · {duration:.1f} s")
        except Exception as exc:  # noqa: BLE001
            self.video_info.set(f"No pude leer los datos del vídeo: {exc}")

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Guardar vídeo analizado",
            defaultextension=".mp4",
            filetypes=(("MP4", "*.mp4"), ("Todos", "*.*")),
        )
        if path:
            self.output_path.set(path)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        for control in self._controls:
            try:
                control.configure(state="disabled" if running else "normal")
            except tk.TclError:
                continue
        if not running:
            # Comboboxes must return to readonly rather than editable.
            for control in self._controls:
                if isinstance(control, ttk.Combobox):
                    control.configure(state="readonly")

    def _update_progress(self, line: str) -> None:
        match = PROGRESS_RE.match(line.strip())
        if not match:
            return
        phase, current, total = match.groups()
        fraction = int(current) / max(1, int(total))
        phase_range = {"analyzing": (0, 55), "rendering": (55, 95), "exporting": (95, 100)}
        start, end = phase_range.get(phase, (0, 100))
        self.progress_value.set(start + (end - start) * fraction)
        label = {"analyzing": "Analizando", "rendering": "Preparando overlay", "exporting": "Exportando"}.get(phase, "Procesando")
        self.status.set(f"{label}: {int(fraction * 100)} %")

    def _run_commands(self, commands: list[list[str]], done_message: str) -> None:
        def worker() -> None:
            self.after(0, self._set_running, True)
            self.after(0, self.progress_value.set, 0.0)
            self.after(0, self.status.set, "Preparando análisis…")
            for command in commands:
                self.after(0, self._append_log, "\n" + " ".join(command) + "\n")
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    self.after(0, self._append_log, line)
                    self.after(0, self._update_progress, line)
                if process.wait() != 0:
                    self.after(0, self._set_running, False)
                    self.after(0, self.status.set, "No pude completar el análisis.")
                    self.after(0, messagebox.showerror, "PowerNZ", "No pude completar la tarea. Revisa el registro.")
                    return
            self.after(0, self.progress_value.set, 100.0)
            self.after(0, self.status.set, done_message)
            self.after(0, self._set_running, False)
            self.after(0, messagebox.showinfo, "PowerNZ", done_message)

        threading.Thread(target=worker, daemon=True).start()

    def _models_are_ready(self) -> bool:
        result = subprocess.run(
            [sys.executable, "model_downloader.py", "--check"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def _download_models(self) -> None:
        self._run_commands([[sys.executable, "model_downloader.py"]], "Modelos listos.")

    def _analyze_video(self) -> None:
        if not self.input_path.get():
            messagebox.showwarning("PowerNZ", "Primero elige un vídeo.")
            return
        if not self.output_path.get():
            messagebox.showwarning("PowerNZ", "Elige dónde guardar el vídeo analizado.")
            return
        if self.calibration_mode.get() == "manual":
            try:
                if float(self.plate_diameter.get()) <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("PowerNZ", "Escribe un diámetro de disco válido para la calibración manual.")
                return

        output = Path(self.output_path.get())
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "main.py",
            "--input", self.input_path.get(),
            "--output", str(output),
            "--exercise", EXERCISES[self.exercise_label.get()],
            "--profile", PROFILES[self.profile_label.get()],
            "--pose-backend", "auto",
            "--calibration-mode", self.calibration_mode.get(),
            "--output-format", self.output_format.get(),
        ]
        if self.calibration_mode.get() == "manual":
            command.extend(["--plate-diameter-px", self.plate_diameter.get()])
        commands = [command]
        if not self._models_are_ready():
            should_download = messagebox.askyesno(
                "PowerNZ",
                "Faltan modelos. ¿Quieres prepararlos antes de analizar?",
            )
            if not should_download:
                return
            commands.insert(0, [sys.executable, "model_downloader.py"])
        self._run_commands(commands, f"Análisis terminado: {output}")

    def _open_output_folder(self) -> None:
        output = Path(self.output_path.get()) if self.output_path.get() else ROOT / "outputs"
        folder = output if output.is_dir() else output.parent
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(folder)])


if __name__ == "__main__":
    app = PowerNZLauncher()
    app.mainloop()
