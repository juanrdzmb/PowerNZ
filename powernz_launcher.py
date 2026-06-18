from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent


class PowerNZLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PowerNZ")
        self.geometry("760x560")
        self.minsize(720, 520)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.exercise = tk.StringVar(value="deadlift")
        self.plate_diameter = tk.StringVar(value="120")
        self.output_format = tk.StringVar(value="portrait-720")
        self.status = tk.StringVar(value="Listo para analizar.")

        self._build_ui()

    def _build_ui(self) -> None:
        padding = {"padx": 14, "pady": 8}
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=16, pady=16)

        title = ttk.Label(frame, text="PowerNZ", font=("Segoe UI", 20, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w", **padding)

        ttk.Label(frame, text="Video").grid(row=1, column=0, sticky="w", **padding)
        ttk.Entry(frame, textvariable=self.input_path).grid(row=1, column=1, sticky="ew", **padding)
        ttk.Button(frame, text="Elegir", command=self._choose_input).grid(row=1, column=2, sticky="ew", **padding)

        ttk.Label(frame, text="Salida").grid(row=2, column=0, sticky="w", **padding)
        ttk.Entry(frame, textvariable=self.output_path).grid(row=2, column=1, sticky="ew", **padding)
        ttk.Button(frame, text="Guardar como", command=self._choose_output).grid(row=2, column=2, sticky="ew", **padding)

        ttk.Label(frame, text="Ejercicio").grid(row=3, column=0, sticky="w", **padding)
        exercise_box = ttk.Combobox(
            frame,
            textvariable=self.exercise,
            values=("deadlift", "squat", "bench"),
            state="readonly",
        )
        exercise_box.grid(row=3, column=1, sticky="ew", **padding)

        ttk.Label(frame, text="Diametro del plato en px").grid(row=4, column=0, sticky="w", **padding)
        ttk.Entry(frame, textvariable=self.plate_diameter).grid(row=4, column=1, sticky="ew", **padding)

        ttk.Label(frame, text="Formato").grid(row=5, column=0, sticky="w", **padding)
        format_box = ttk.Combobox(
            frame,
            textvariable=self.output_format,
            values=("portrait-720", "source"),
            state="readonly",
        )
        format_box.grid(row=5, column=1, sticky="ew", **padding)

        button_row = ttk.Frame(frame)
        button_row.grid(row=6, column=0, columnspan=3, sticky="ew", padx=14, pady=(10, 6))
        ttk.Button(button_row, text="Descargar modelos", command=self._download_models).pack(side="left")
        ttk.Button(button_row, text="Analizar video", command=self._analyze_video).pack(side="left", padx=10)
        ttk.Button(button_row, text="Abrir carpeta de salida", command=self._open_output_folder).pack(side="left")

        ttk.Label(frame, textvariable=self.status).grid(row=7, column=0, columnspan=3, sticky="w", **padding)

        self.log = tk.Text(frame, height=12, wrap="word")
        self.log.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=14, pady=8)
        self.log.configure(state="disabled")

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(8, weight=1)

    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Elegir video",
            filetypes=(("Videos", "*.mp4 *.mov *.avi *.mkv"), ("Todos", "*.*")),
        )
        if not path:
            return
        self.input_path.set(path)
        if not self.output_path.get():
            source = Path(path)
            self.output_path.set(str(ROOT / "outputs" / f"{source.stem}_analizado.mp4"))

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Guardar video analizado",
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

    def _run_commands(self, commands: list[list[str]], done_message: str) -> None:
        def worker() -> None:
            self.after(0, self.status.set, "Trabajando...")
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
                code = process.wait()
                if code != 0:
                    self.after(0, self.status.set, "No pude completar la tarea.")
                    self.after(0, messagebox.showerror, "PowerNZ", "No pude completar la tarea. Revisa el registro.")
                    return
            self.after(0, self.status.set, done_message)
            self.after(0, messagebox.showinfo, "PowerNZ", done_message)

        threading.Thread(target=worker, daemon=True).start()

    def _run_command(self, command: list[str], done_message: str) -> None:
        self._run_commands([command], done_message)

    def _models_are_ready(self) -> bool:
        result = subprocess.run(
            [sys.executable, "model_downloader.py", "--check"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def _download_models(self) -> None:
        self._run_command([sys.executable, "model_downloader.py"], "Modelos listos.")

    def _analyze_video(self) -> None:
        if not self.input_path.get():
            messagebox.showwarning("PowerNZ", "Primero elige un video.")
            return
        if not self.output_path.get():
            messagebox.showwarning("PowerNZ", "Elige donde guardar el video analizado.")
            return
        try:
            float(self.plate_diameter.get())
        except ValueError:
            messagebox.showwarning("PowerNZ", "El diametro del plato debe ser un numero.")
            return

        output = Path(self.output_path.get())
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "main.py",
            "--input",
            self.input_path.get(),
            "--output",
            str(output),
            "--exercise",
            self.exercise.get(),
            "--pose-backend",
            "yolo",
            "--plate-diameter-px",
            self.plate_diameter.get(),
            "--output-format",
            self.output_format.get(),
        ]
        commands = [command]
        if not self._models_are_ready():
            should_download = messagebox.askyesno(
                "PowerNZ",
                "Faltan modelos entrenados. ¿Quieres descargarlos antes de analizar?",
            )
            if not should_download:
                return
            commands.insert(0, [sys.executable, "model_downloader.py"])
        self._run_commands(commands, f"Analisis terminado: {output}")

    def _open_output_folder(self) -> None:
        output = Path(self.output_path.get()) if self.output_path.get() else ROOT / "outputs"
        folder = output if output.is_dir() else output.parent
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(folder)])


if __name__ == "__main__":
    app = PowerNZLauncher()
    app.mainloop()
