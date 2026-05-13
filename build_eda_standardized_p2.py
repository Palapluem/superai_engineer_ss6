
print("Merging standardized cells with original notebook...")
final_cells = old_cells + cells

final_nb = {
    "cells": final_cells,
    "metadata": old_nb.get("metadata", {}),
    "nbformat": old_nb.get("nbformat", 4),
    "nbformat_minor": old_nb.get("nbformat_minor", 5)
}

with open(out_nb_path, 'w', encoding='utf-8') as f:
    json.dump(final_nb, f, indent=2, ensure_ascii=False)

print(f"[SUCCESS] Standarized Notebook generated at:\n{out_nb_path}")
