"""Parse Equibase result chart XML files into flat dicts."""

import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from data.schema import make_race_id, safe_float, safe_int, to_yards, xml_text


def _parse_point_of_calls(entry: ET.Element) -> dict:
    """Extract POINT_OF_CALL elements into flat dict keyed by call number."""
    result = {}
    for poc in entry.findall("POINT_OF_CALL"):
        which = poc.get("WHICH", "")
        key = "final" if which == "FINAL" else which
        result[f"point_of_call_{key}_position"] = safe_int(xml_text(poc, "POSITION"))
        result[f"point_of_call_{key}_lengths"] = xml_text(poc, "LENGTHS")
    return result


def parse_result_chart(xml_path: Path) -> list[dict]:
    """Parse a single result chart XML file into a list of horse-in-a-race dicts."""
    tree = ET.parse(xml_path)
    chart = tree.getroot()

    race_date_str = chart.get("RACE_DATE", "")
    race_date = date.fromisoformat(race_date_str) if race_date_str else None
    track_el = chart.find("TRACK")
    track = xml_text(track_el, "CODE") if track_el is not None else None

    rows = []
    for race in chart.findall("RACE"):
        race_number = safe_int(race.get("NUMBER"))
        if race_number is None:
            continue

        race_date_str = race_date.isoformat() if race_date else ""
        race_id = make_race_id(race_date_str, track or "", race_number)

        entries = race.findall("ENTRY")
        num_runners = len(entries)

        dist_val = safe_int(xml_text(race, "DISTANCE"))
        dist_unit = xml_text(race, "DIST_UNIT")

        race_fields = {
            "race_id": race_id,
            "race_date": race_date,
            "track": track,
            "race_number": race_number,
            "breed": xml_text(race, "BREED"),
            "race_type": xml_text(race, "TYPE"),
            "course_id": xml_text(race, "COURSE_ID"),
            "course_desc": xml_text(race, "COURSE_DESC"),
            "purse": safe_float(xml_text(race, "PURSE")),
            "distance_val": dist_val,
            "distance_unit": dist_unit,
            "distance_yards": to_yards(dist_val, dist_unit),
            "run_up_distance": safe_int(xml_text(race, "RUNUPDIST")),
            "surface": xml_text(race, "SURFACE"),
            "track_condition": xml_text(race, "TRK_COND"),
            "weather": xml_text(race, "WEATHER"),
            "class_rating": safe_int(xml_text(race, "CLASS_RATING")),
            "num_runners": num_runners,
            "win_time": xml_text(race, "WIN_TIME"),
            "fraction_1": safe_float(xml_text(race, "FRACTION_1")),
            "fraction_2": safe_float(xml_text(race, "FRACTION_2")),
            "fraction_3": safe_float(xml_text(race, "FRACTION_3")),
            "fraction_4": safe_float(xml_text(race, "FRACTION_4")),
            "fraction_5": safe_float(xml_text(race, "FRACTION_5")),
            "pace_call_1": xml_text(race, "PACE_CALL1"),
            "pace_call_2": xml_text(race, "PACE_CALL2"),
            "pace_final": xml_text(race, "PACE_FINAL"),
            "footnotes": xml_text(race, "FOOTNOTES"),
        }

        for entry in entries:
            row = {**race_fields}
            row["horse_name"] = xml_text(entry, "NAME")
            row["program_number"] = xml_text(entry, "PROGRAM_NUM")
            row["post_position"] = safe_int(xml_text(entry, "POST_POS"))
            row["weight"] = safe_int(xml_text(entry, "WEIGHT"))
            row["age"] = safe_int(xml_text(entry, "AGE"))
            row["sex"] = xml_text(entry, "SEX/CODE")
            row["medication"] = xml_text(entry, "MEDS")
            row["equipment"] = xml_text(entry, "EQUIP")
            row["claim_price"] = safe_float(xml_text(entry, "CLAIM_PRICE"))
            row["dollar_odds"] = safe_float(xml_text(entry, "DOLLAR_ODDS"))
            row["official_finish"] = safe_int(xml_text(entry, "OFFICIAL_FIN"))
            row["finish_time"] = xml_text(entry, "FINISH_TIME")
            row["speed_rating"] = safe_int(xml_text(entry, "SPEED_RATING"))
            row["comment"] = xml_text(entry, "COMMENT")
            row["dh_dq_flags"] = xml_text(entry, "DH_DQ_FLAGS")

            jockey = entry.find("JOCKEY")
            if jockey is not None:
                row["jockey_first_name"] = xml_text(jockey, "FIRST_NAME")
                row["jockey_last_name"] = xml_text(jockey, "LAST_NAME")
            else:
                row["jockey_first_name"] = None
                row["jockey_last_name"] = None

            trainer = entry.find("TRAINER")
            if trainer is not None:
                row["trainer_first_name"] = xml_text(trainer, "FIRST_NAME")
                row["trainer_last_name"] = xml_text(trainer, "LAST_NAME")
            else:
                row["trainer_first_name"] = None
                row["trainer_last_name"] = None

            row["win_payoff"] = safe_float(xml_text(entry, "WIN_PAYOFF"))
            row["place_payoff"] = safe_float(xml_text(entry, "PLACE_PAYOFF"))
            row["show_payoff"] = safe_float(xml_text(entry, "SHOW_PAYOFF"))

            row.update(_parse_point_of_calls(entry))

            rows.append(row)

    return rows
