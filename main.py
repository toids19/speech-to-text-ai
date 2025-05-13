import os
import sys
import torch
import whisper
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from threading import Thread
import time

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Ensure cross-platform compatibility
if sys.platform == "darwin":  # macOS
    os.environ["PYTHONWARNINGS"] = "ignore"
elif sys.platform == "win32":  # Windows
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["PYTHONWARNINGS"] = "ignore"
else:  # Linux
    os.environ["PYTHONWARNINGS"] = "ignore"

def transcribe_audio(file_path: str):
    """ Transcribes an MP3 file into text using Whisper and updates UI."""
    global text_output, progress_label, progress_bar, completion_time_label

    if not os.path.exists(file_path):
        messagebox.showerror("Error", "Audio file not found.")
        return

    progress_label.config(text="Loading model...")
    progress_bar.config(text="0%")
    completion_time_label.config(text="")
    root.update_idletasks()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    model_path = "./models/medium.pt"
    model = whisper.load_model(model_path, device=device)

    progress_label.config(text="Transcribing audio...")
    root.update_idletasks()

    start_time = time.time()
    result = model.transcribe(file_path, language='ro', fp16=torch.cuda.is_available(), no_speech_threshold=0.6,
                              temperature=0.0, beam_size=1, verbose=False)
    end_time = time.time()
    elapsed_time = end_time - start_time

    progress_label.config(text="Transcription In Progress...")
    text_output.delete(1.0, tk.END)
    words_processed = 0
    total_words = sum(len(segment["text"].split()) for segment in result["segments"])

    for i, segment in enumerate(result["segments"], start=1):
        formatted_text = f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}\n\n"
        text_output.insert(tk.END, formatted_text)
        words_processed += len(segment['text'].split())
        progress_percentage = (words_processed / total_words) * 100
        progress_bar.config(text=f"{progress_percentage:.2f}%")
        root.update_idletasks()
        time.sleep(0.07)

    progress_label.config(text="Transcription Complete!")
    progress_bar.config(text="100%")
    completion_time_label.config(text=f"Completed in: {elapsed_time:.2f} seconds")

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
    if file_path:
        Thread(target=transcribe_audio, args=(file_path,)).start()

def save_transcription():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_output.get(1.0, tk.END))
        messagebox.showinfo("Success", f"Transcription saved to {file_path}")

# GUI Setup
root = tk.Tk()
# Set app icon
icon_path = resource_path("icon.png")
root.iconphoto(False, tk.PhotoImage(file=icon_path))
root.title("MP3 to Text Beta Version - Moldovanu Tudor")
root.geometry("700x550")
root.resizable(False, False)

tk.Label(root, text="Select an MP3 file to transcribe", font=("Helvetica", 14, "bold")).pack(pady=10)
btn_select = tk.Button(root, text="Browse", command=select_file, font=("Helvetica", 12, "bold"))
btn_select.pack(pady=5)

progress_label = tk.Label(root, text="", font=("Helvetica", 12))
progress_label.pack(pady=5)

progress_bar = tk.Label(root, text="", font=("Helvetica", 12))
progress_bar.pack(pady=5)

text_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20, font=("Courier", 12))
text_output.pack(pady=10)

btn_save = tk.Button(root, text="Save Transcription", command=save_transcription, font=("Helvetica", 12, "bold"))
btn_save.pack(pady=5)

completion_time_label = tk.Label(root, text="", font=("Helvetica", 12))
completion_time_label.pack(pady=5)

root.mainloop()
