import csv
import re
from pathlib import Path

# ===================== 설정 =====================
TXT_DIR = Path.home() / "Desktop" / "txt"   # txt 폴더
BASELINE_SECONDS = 10.0

# ===================== 유틸 =====================
def fnum(x):
    if x is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else None

def parse_time(s):
    if s is None:
        return None
    s = str(s).replace("s", "").strip()
    return fnum(s)

def safe_write_csv(out_path: Path, rows):
    """
    기존 파일이 있으면 덮어쓰기 시도,
    열려 있으면 (1), (2) 붙여 저장
    """
    out_path = Path(out_path)

    if out_path.exists():
        try:
            out_path.unlink()
        except PermissionError:
            pass

    def _write(p: Path):
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)

    try:
        _write(out_path)
        return out_path
    except PermissionError:
        stem, suf, parent = out_path.stem, out_path.suffix, out_path.parent
        for i in range(1, 9999):
            alt = parent / f"{stem} ({i}){suf}"
            if alt.exists():
                continue
            _write(alt)
            return alt

# ===================== 정규식 =====================
RE_BASELINE = re.compile(r"Collecting\s+baseline.*?RawAvg\s*=\s*([0-9.]+)", re.I)
RE_RAWAVG_ANY = re.compile(r"RawAvg\s*=\s*([0-9.]+)", re.I)
RE_FINAL_BASE = re.compile(r"Final\s+Base\s+Raw\s+Average\s*=\s*([0-9.]+)", re.I)

RE_DETECT = re.compile(
    r"Time\s*=\s*([^|]+)\|\s*BaseRaw\s*=\s*([^|]+)\|\s*CurrentRaw\s*=\s*([^|]+)\|\s*Change\s*=\s*([^=]+)=>",
    re.I
)

# ===================== TXT 파싱 =====================
def parse_txt_to_oldstyle_rows(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    baseline_vals = []
    detect_rows = []
    final_base = None
    in_baseline = True

    for line in lines:
        line = line.strip()
        if not line:
            continue

        mfb = RE_FINAL_BASE.search(line)
        if mfb:
            final_base = fnum(mfb.group(1))

        if "Baseline collection complete" in line:
            in_baseline = False
            continue

        if in_baseline:
            mb = RE_BASELINE.search(line)
            if mb:
                v = fnum(mb.group(1))
                if v is not None:
                    baseline_vals.append(v)
                continue

            ma = RE_RAWAVG_ANY.search(line)
            if ma:
                v = fnum(ma.group(1))
                if v is not None:
                    baseline_vals.append(v)
            continue

        md = RE_DETECT.search(line)
        if md:
            t = parse_time(md.group(1))
            base = fnum(md.group(2))
            cur = fnum(md.group(3))
            chg = fnum(md.group(4))

            if t is None or cur is None or chg is None:
                continue

            detect_rows.append({
                "Time": t,
                "BaseRaw": base,
                "CurrentRaw": cur,
                "Change": chg,
            })

    # base 대표값 결정
    if final_base is None and baseline_vals:
        final_base = sum(baseline_vals) / len(baseline_vals)

    if final_base is None:
        return []

    out_rows = []

    # 1) baseline 구간을 예전 형식으로 기록
    for v in baseline_vals:
        out_rows.append([
            "Collecting baseline...",
            "RawAvg",
            round(v, 3)
        ])

    # 2) Final Base 행
    out_rows.append([
        "Final", "Base", "Raw", "Average", round(final_base, 3)
    ])

    # 3) detect 행을 예전 로그형 형식으로 기록
    for r in detect_rows:
        t = round(r["Time"], 3)
        b = r["BaseRaw"] if r["BaseRaw"] is not None else final_base
        c = r["CurrentRaw"]
        ch = r["Change"]

        state = "Liquid" if abs(ch) >= 5.0 else "Floor"

        out_rows.append([
            "Time", f"{t}s",
            "|",
            "BaseRaw", round(b, 3),
            "|",
            "CurrentRaw", round(c, 3),
            "|",
            "Change", round(ch, 3),
            "=>",
            state,
            "(empty)"
        ])

    return out_rows

# ===================== 실행 =====================
def main():
    if not TXT_DIR.exists():
        print(f"[ERROR] 폴더 없음: {TXT_DIR}")
        return

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[INFO] 변환할 txt 없음: {TXT_DIR}")
        return

    ok = 0
    for txt in txt_files:
        rows = parse_txt_to_oldstyle_rows(txt)
        if not rows:
            print(f"[SKIP] 데이터 없음 또는 base 없음: {txt.name}")
            continue

        out_csv = txt.with_suffix(".csv")
        saved = safe_write_csv(out_csv, rows)
        print(f"[OK] {txt.name} -> {saved.name} (rows={len(rows)})")
        ok += 1

    print(f"\n완료: {ok}개 csv 생성")

if __name__ == "__main__":
    main()