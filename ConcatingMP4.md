Here's a solid approach: instead of re-encoding each file in a loop (slow, lossy), the best method is **ffmpeg's concat demuxer** — it stitches files losslessly and near-instantly, as long as they have the same codec/resolution/fps.

```bash
#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-.}"          # folder to scan (default: current dir)
OUT="${2:-merged.mp4}" # output file

cd "$DIR"

# Build the concat list (sorted by name)
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT

for f in *.mp4; do
    [[ -e "$f" ]] || { echo "No .mp4 files found."; exit 1; }
    printf "file '%s'\n" "$PWD/${f//\'/\'\\\'\'}" >> "$LIST"
done

ffmpeg -f concat -safe 0 -i "$LIST" -c copy "$OUT"
echo "Done: $OUT"
```

**Usage:**
```bash
chmod +x merge.sh
./merge.sh /path/to/folder output.mp4
```

**Notes / caveats:**

- `-c copy` reuses the existing streams without re-encoding — fast and lossless. It only works reliably if all files share the same codec, resolution, framerate, and audio settings (e.g. chunks from the same camera/recording session).
- If your files are **mixed codecs/resolutions**, `-c copy` will produce a broken or glitchy output. Then re-encode instead:
  ```bash
  ffmpeg -f concat -safe 0 -i "$LIST" -c:v libx264 -crf 20 -c:a aac output.mp4
  ```
- The loop is only used to build the file list (sorted alphabetically) — the actual concatenation is one single ffmpeg call, which is far more efficient than encoding files one by one.

Want me to add support for other extensions (e.g. `.mkv`, `.mov`) or a specific sort order (like natural/numeric sorting for `file1, file2, ... file10`)?
