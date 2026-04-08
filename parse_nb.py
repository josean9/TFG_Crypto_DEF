import json

with open(r'c:\Users\josit\CUARTO CURSO\TFG\TFG_Crypto_DEF\notebooks\lstm_transformer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(r'c:\Users\josit\CUARTO CURSO\TFG\TFG_Crypto_DEF\parse_output.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb.get('cells', [])):
        out.write(f"\n--- Cell {i} ({cell['cell_type']}) ---\n")
        source = ''.join(cell.get('source', []))
        out.write(source[:800] + ('\n[...]' if len(source) > 800 else ''))
        out.write('\n')
