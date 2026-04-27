import csv
from pathlib import Path

# ✅ 원하는 컬럼 고정 순서 (네가 올린 CSV처럼)
PREFERRED_ORDER = [
    "Average",
    "BaseRaw",
    "Change",
    "CurrentRaw",
    "RawAvg",
    "Time",
]

def parse_line(line: str):
    """
    - 탭/스페이스로만 토큰 분리
    - 토큰 안에 '='가 있으면 key=value 로 취급
    - 그 외는 extras로 유지
    """
    line = line.strip()
    if not line:
        return {}, []

    tokens = line.replace("\t", " ").split()
    kv = {}
    extras = []

    for token in tokens:
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                kv[key] = value
        else:
            extras.append(token)

    return kv, extras


def convert_txt_to_csv(input_path: Path, output_path: Path):
    rows = []
    all_keys = set()
    max_extras = 0

    # txt 읽기 (한글 포함 대비: utf-8 우선, 깨지는 글자는 무시)
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            kv, extras = parse_line(line)
            if not kv and not extras:
                continue

            all_keys.update(kv.keys())
            max_extras = max(max_extras, len(extras))
            rows.append((kv, extras))

    # ✅ key=value 컬럼: 고정 순서 먼저, 나머지는 뒤에 정렬해서 붙이기
    ordered_keys = [k for k in PREFERRED_ORDER if k in all_keys]
    remaining_keys = sorted(k for k in all_keys if k not in PREFERRED_ORDER)
    key_cols = ordered_keys + remaining_keys

    # extras는 extra_1, extra_2 ... 로 뒤에 붙임
    extra_cols = [f"extra_{i+1}" for i in range(max_extras)]
    header = key_cols + extra_cols

    with output_path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=header)
        writer.writeheader()

        for kv, extras in rows:
            row = {k: kv.get(k, "") for k in key_cols}
            for i, val in enumerate(extras):
                row[f"extra_{i+1}"] = val
            writer.writerow(row)


def main():
    folder = Path(".")
    txt_files = sorted(folder.glob("*.txt"))

    if not txt_files:
        print("txt 파일이 없습니다.")
        return

    for input_path in txt_files:
        output_path = input_path.with_suffix(".csv")
        convert_txt_to_csv(input_path, output_path)
        print(f"변환 완료: {output_path.name}")


if __name__ == "__main__":
    main()
