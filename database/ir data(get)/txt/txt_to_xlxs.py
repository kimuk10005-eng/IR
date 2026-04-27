import re
from pathlib import Path

from openpyxl import Workbook

# ===================== 설정 =====================
TXT_DIR = Path.home() / "Desktop" / "txt"   # ← TXT 폴더 경로 바꾸려면 여기만 수정
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

def safe_write_xlsx(out_path: Path, rows) -> bool:
    """
    - 같은 이름의 .xlsx가 이미 있으면 생성하지 않음(덮어쓰기 X) -> False
    - 파일이 열려있어 PermissionError가 나도 생성하지 않음 -> False
    - 정상 저장하면 True
    """
    out_path = Path(out_path)

    # ✅ 중복 파일이면 생성하지 않음
    if out_path.exists():
        return False

    wb = Workbook()
    ws = wb.active
    ws.title = "data"

    # 헤더
    ws.append(["Time", "BaseRaw", "CurrentRaw", "Change"])

    # 데이터
    for r in rows:
        ws.append([r["Time"], r["BaseRaw"], r["CurrentRaw"], r["Change"]])

    try:
        wb.save(out_path)
        return True
    except PermissionError:
        # ✅ (1) 같은 이름 붙여서 저장하지 않고 그냥 실패 처리
        return False

# ===================== 정규식 =====================
RE_BASELINE = re.compile(r"Collecting\s+baseline.*?RawAvg\s*=\s*([0-9.]+)", re.I)
RE_RAWAVG_ANY = re.compile(r"RawAvg\s*=\s*([0-9.]+)", re.I)
RE_FINAL_BASE = re.compile(r"Final\s+Base\s+Raw\s+Average\s*=\s*([0-9.]+)", re.I)

RE_DETECT = re.compile(
    r"Time\s*=\s*([^|]+)\|\s*BaseRaw\s*=\s*([^|]+)\|\s*CurrentRaw\s*=\s*([^|]+)\|\s*Change\s*=\s*([^=]+)=>",
    re.I
)

def parse_txt(path: Path):
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

            if t is None:
                continue

            detect_rows.append({
                "Time": BASELINE_SECONDS + t,  # baseline 10초 뒤로 붙이기
                "BaseRaw": base,
                "CurrentRaw": cur,
                "Change": chg,
            })

    rows = []

    # baseline을 0~10초로 균등 배치
    if baseline_vals:
        n = len(baseline_vals)
        base_val = final_base if final_base is not None else (sum(baseline_vals) / n)

        if n == 1:
            times = [0.0]
        else:
            dt = BASELINE_SECONDS / (n - 1)
            times = [round(i * dt, 3) for i in range(n)]

        for t, rawavg in zip(times, baseline_vals):
            rows.append({
                "Time": t,
                "BaseRaw": base_val,
                "CurrentRaw": rawavg,
                "Change": abs(base_val - rawavg),
            })

    rows.extend([r for r in detect_rows if r["Time"] is not None])
    rows = [r for r in rows if r["Time"] is not None]
    rows.sort(key=lambda x: x["Time"])
    return rows

def main():
    if not TXT_DIR.exists():
        print(f"[ERROR] 폴더 없음: {TXT_DIR}")
        return

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[INFO] 변환할 txt 없음: {TXT_DIR}")
        return

    ok = 0
    skip_exist = 0
    skip_nodata = 0
    skip_perm = 0

    for txt in txt_files:
        rows = parse_txt(txt)
        if not rows:
            print(f"[SKIP] 데이터 없음: {txt.name}")
            skip_nodata += 1
            continue

        out_xlsx = txt.with_suffix(".xlsx")

        # ✅ 이미 있으면 생성하지 않음
        if out_xlsx.exists():
            print(f"[SKIP] 이미 존재: {out_xlsx.name}")
            skip_exist += 1
            continue

        saved = safe_write_xlsx(out_xlsx, rows)
        if saved:
            print(f"[OK] {txt.name} -> {out_xlsx.name} (rows={len(rows)})")
            ok += 1
        else:
            # 존재하지 않았는데 저장 실패면 보통 PermissionError(열려있음) 케이스
            print(f"[SKIP] 저장 실패(열려있거나 권한 문제): {out_xlsx.name}")
            skip_perm += 1

    print(f"\n완료: {ok}개 xlsx 생성")
    print(f"SKIP(이미 존재): {skip_exist}")
    print(f"SKIP(데이터 없음): {skip_nodata}")
    print(f"SKIP(권한/열림): {skip_perm}")

if __name__ == "__main__":
    main()
