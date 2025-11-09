from tqdm import tqdm
from rich.console import Console

console = Console(record=False)

def progress_bar(iterable, description="Processing", total=None):
    return tqdm(iterable, desc=description, total=total)

def status(msg, style="bold cyan"):
    console.print(f"[{style}]{msg}[/]")
