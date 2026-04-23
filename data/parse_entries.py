"""Parse Equibase PPS (Past Performance Statement) ZIP/XML files."""

import xml.etree.ElementTree as ET
import zipfile
from datetime import date
from pathlib import Path

from data.schema import (
    make_race_id,
    parse_odds,
    safe_float,
    safe_int,
    to_yards,
    xml_text,
)


def parse_pps_zip(zip_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Parse a PPs ZIP file into (entries, past_performances, workouts).

    Each ZIP contains a single XML file with an EntryRaceCard for one track on one date.
    """
    with zipfile.ZipFile(zip_path) as zf:
        xml_name = zf.namelist()[0]
        with zf.open(xml_name) as f:
            tree = ET.parse(f)

    root = tree.getroot()

    race_date = _parse_date(xml_text(root, "RaceDate"))
    track = xml_text(root, "Track/TrackID")

    entries = []
    past_performances = []
    workouts = []

    for race_el in root.findall("Race"):
        race_number = safe_int(xml_text(race_el, "RaceNumber"))
        if race_number is None:
            continue

        race_date_str = race_date.isoformat() if race_date else ""
        race_id = make_race_id(race_date_str, track or "", race_number)

        dist_val = safe_int(xml_text(race_el, "Distance/DistanceId"))
        dist_unit = xml_text(race_el, "Distance/DistanceUnit/Value")

        race_fields = {
            "race_id": race_id,
            "race_date": race_date,
            "track": track,
            "race_number": race_number,
            "breed": xml_text(race_el, "BreedType/Value"),
            "race_type": xml_text(race_el, "RaceType/RaceType"),
            "race_type_desc": xml_text(race_el, "RaceType/Description"),
            "surface": xml_text(race_el, "Course/CourseType/Value"),
            "distance_val": dist_val,
            "distance_unit": dist_unit,
            "distance_yards": to_yards(dist_val, dist_unit),
            "purse": safe_float(xml_text(race_el, "PurseUSA")),
            "grade": xml_text(race_el, "Grade"),
            "num_runners": safe_int(xml_text(race_el, "NumberOfRunners")),
            "condition_text": xml_text(race_el, "ConditionText"),
            "max_claim_price": safe_float(xml_text(race_el, "MaximumClaimPrice")),
            "age_restriction": xml_text(race_el, "AgeRestriction/Value"),
            "sex_restriction": xml_text(race_el, "SexRestriction/Value"),
        }

        for starter in race_el.findall("Starters"):
            horse_el = starter.find("Horse")
            if horse_el is None:
                continue

            horse_name = xml_text(horse_el, "HorseName")
            registration_number = xml_text(horse_el, "RegistrationNumber")

            entry = {**race_fields}
            entry["horse_name"] = horse_name
            entry["registration_number"] = registration_number
            entry["year_of_birth"] = safe_int(xml_text(horse_el, "YearOfBirth"))
            entry["sex"] = xml_text(horse_el, "Sex/Value")
            sire_el = horse_el.find("Sire")
            entry["sire_name"] = (
                xml_text(sire_el, "HorseName") if sire_el is not None else None
            )
            dam_el = horse_el.find("Dam")
            dam_sire_el = dam_el.find("Sire") if dam_el is not None else None
            entry["dam_sire_name"] = (
                xml_text(dam_sire_el, "HorseName") if dam_sire_el is not None else None
            )
            entry["post_position"] = safe_int(xml_text(starter, "PostPosition"))
            entry["program_number"] = xml_text(starter, "ProgramNumber")

            odds_raw = xml_text(starter, "Odds")
            entry["morning_line_odds"] = odds_raw
            entry["morning_line_odds_float"] = parse_odds(odds_raw)

            entry["weight_carried"] = safe_int(xml_text(starter, "WeightCarried"))

            jockey_el = starter.find("Jockey")
            if jockey_el is not None:
                entry["jockey_first_name"] = xml_text(jockey_el, "FirstName")
                entry["jockey_last_name"] = xml_text(jockey_el, "LastName")
            else:
                entry["jockey_first_name"] = None
                entry["jockey_last_name"] = None

            trainer_el = starter.find("Trainer")
            if trainer_el is not None:
                entry["trainer_first_name"] = xml_text(trainer_el, "FirstName")
                entry["trainer_last_name"] = xml_text(trainer_el, "LastName")
            else:
                entry["trainer_first_name"] = None
                entry["trainer_last_name"] = None

            entry["equipment"] = xml_text(starter, "Equipment/Value")
            entry["medication"] = xml_text(starter, "Medication/Value")
            entry["apprentice_weight_allowance"] = safe_int(
                xml_text(starter, "ApprenticeWeightAllowance")
            )
            # stored as displayed figure * 10 (e.g., 570 displays as 57); matches SpeedFigure scale
            raw_class = safe_int(xml_text(starter, "TodaysHorseClassRating"))
            entry["class_rating"] = raw_class / 10 if raw_class is not None else None

            # career stats from RaceSummary elements
            _add_career_stats(entry, starter)

            entries.append(entry)

            # Past performances — sorted by date descending, indexed 1=most recent
            pps = starter.findall("PastPerformance")
            pp_dates = []
            for pp in pps:
                pp_date = _parse_date(xml_text(pp, "RaceDate"))
                pp_dates.append((pp_date, pp))
            pp_dates.sort(key=lambda x: x[0] or date.min, reverse=True)

            for idx, (_, pp) in enumerate(pp_dates, start=1):
                past_performances.append(
                    _parse_past_performance(
                        pp, race_id, horse_name or "", registration_number or "", idx
                    )
                )

            # Workouts
            for wo in starter.findall("Workout"):
                workouts.append(
                    _parse_workout(
                        wo, race_id, horse_name or "", registration_number or ""
                    )
                )

    return entries, past_performances, workouts


def _parse_past_performance(
    pp: ET.Element,
    race_id: str,
    horse_name: str,
    registration_number: str,
    pp_index: int,
) -> dict:
    """Extract fields from a PastPerformance element."""
    start = pp.find("Start")

    pp_dist_val = safe_int(xml_text(pp, "Distance/DistanceId"))
    pp_dist_unit = xml_text(pp, "Distance/DistanceUnit/Value")

    row = {
        "race_id": race_id,
        "horse_name": horse_name,
        "registration_number": registration_number,
        "pp_index": pp_index,
        "pp_race_date": _parse_date(xml_text(pp, "RaceDate")),
        "pp_track": xml_text(pp, "Track/TrackID"),
        "pp_race_number": safe_int(xml_text(pp, "RaceNumber")),
        "pp_race_type": xml_text(pp, "RaceType/RaceType"),
        "pp_distance_val": pp_dist_val,
        "pp_distance_unit": pp_dist_unit,
        "pp_distance_yards": to_yards(pp_dist_val, pp_dist_unit),
        "pp_surface": xml_text(pp, "Course/CourseType/Value"),
        "pp_track_condition": xml_text(pp, "TrackCondition/Value"),
        "pp_num_starters": safe_int(xml_text(pp, "NumberOfStarters")) or None,
        "pp_purse": safe_float(xml_text(pp, "PurseUSA")),
    }

    if start is not None:
        row["pp_post_position"] = safe_int(xml_text(start, "PostPosition"))
        finish = safe_int(xml_text(start, "OfficialFinish"))
        dnf = (finish or 0) >= 90  # values >= 90 are DNF codes (pulled up, eased, did not finish)  # fmt: skip
        row["pp_official_finish"] = finish if finish is not None and not dnf else None
        # values are stored as displayed figure * 10 (e.g., 870 = 87); 9999 is a null sentinel
        raw_speed = safe_int(xml_text(start, "SpeedFigure"))
        row["pp_speed_figure"] = raw_speed // 10 if raw_speed is not None and raw_speed != 9999 else None
        row["pp_odds"] = parse_odds(xml_text(start, "Odds"))
        row["pp_weight_carried"] = safe_int(xml_text(start, "WeightCarried"))
        row["pp_class_rating"] = safe_int(xml_text(start, "ClassRating"))
        row["pp_jockey_last_name"] = xml_text(start, "Jockey/LastName")
        row["pp_trainer_last_name"] = xml_text(start, "Trainer/LastName")
        row["pp_long_comment"] = xml_text(start, "LongComment")
        row["pp_short_comment"] = xml_text(start, "ShortComment")
        row["pp_pace_figure_1"] = safe_int(xml_text(start, "PaceFigure1"))
        row["pp_pace_figure_2"] = safe_int(xml_text(start, "PaceFigure2"))
        row["pp_pace_figure_3"] = safe_int(xml_text(start, "PaceFigure3"))

        # point-of-call positions and lengths behind
        for poc in start.findall("PointOfCall"):
            which = xml_text(poc, "PointOfCall")
            if which == "S":
                row["pp_poc_start_pos"] = _poc_int(xml_text(poc, "Position"))
            elif which == "F":
                row["pp_poc_final_pos"] = _poc_int(xml_text(poc, "Position"))
                row["pp_poc_final_behind"] = _poc_int(xml_text(poc, "LengthsBehind"))
            elif which in ("1", "2", "3", "4", "5"):
                row[f"pp_poc_{which}_pos"] = _poc_int(xml_text(poc, "Position"))
                row[f"pp_poc_{which}_behind"] = _poc_int(xml_text(poc, "LengthsBehind"))
    else:
        for key in [
            "pp_post_position",
            "pp_official_finish",
            "pp_speed_figure",
            "pp_odds",
            "pp_weight_carried",
            "pp_class_rating",
            "pp_jockey_last_name",
            "pp_trainer_last_name",
            "pp_long_comment",
            "pp_short_comment",
            "pp_pace_figure_1",
            "pp_pace_figure_2",
            "pp_pace_figure_3",
        ]:
            row[key] = None

    return row


def _parse_workout(
    wo: ET.Element, race_id: str, horse_name: str, registration_number: str
) -> dict:
    """Extract fields from a Workout element."""
    wo_dist_val = safe_int(xml_text(wo, "Distance/DistanceId"))
    wo_dist_unit = xml_text(wo, "Distance/DistanceUnit/Value")
    return {
        "race_id": race_id,
        "horse_name": horse_name,
        "registration_number": registration_number,
        "workout_date": _parse_date(xml_text(wo, "Date")),
        "workout_track": xml_text(wo, "Track/TrackID"),
        "workout_distance_val": wo_dist_val,
        "workout_distance_unit": wo_dist_unit,
        "workout_distance_yards": to_yards(wo_dist_val, wo_dist_unit),
        "workout_time": xml_text(wo, "Timing"),
        "workout_type": xml_text(wo, "TypeOfWorkout/Value"),
        "workout_course": xml_text(wo, "CourseType/CourseType"),
        "workout_track_condition": xml_text(wo, "TrackCondition/Value"),
        "workout_ranking": xml_text(wo, "Ranking"),
        "workout_num_in_group": xml_text(wo, "NumberInRankingGroup"),
        "workout_comment": xml_text(wo, "Comment"),
    }


def _add_career_stats(entry: dict, starter: ET.Element):
    """Aggregate RaceSummary elements into career totals and surface-specific stats."""
    today_surface = entry.get("surface")  # D, T, etc.

    career_starts = 0
    career_wins = 0
    career_seconds = 0
    career_thirds = 0
    career_earnings = 0.0
    surface_starts = 0
    surface_wins = 0
    surface_seconds = 0
    surface_thirds = 0

    for rs in starter.findall("RaceSummary"):
        starts = safe_int(xml_text(rs, "NumberOfStarts")) or 0
        wins = safe_int(xml_text(rs, "NumberOfWins")) or 0
        seconds = safe_int(xml_text(rs, "NumberOfSeconds")) or 0
        thirds = safe_int(xml_text(rs, "NumberOfThirds")) or 0
        earnings = safe_float(xml_text(rs, "EarningsUSA")) or 0.0
        surface = xml_text(rs, "Surface/Value")

        career_starts += starts
        career_wins += wins
        career_seconds += seconds
        career_thirds += thirds
        career_earnings += earnings

        if surface == today_surface:
            surface_starts += starts
            surface_wins += wins
            surface_seconds += seconds
            surface_thirds += thirds

    # null starts means no RaceSummary data at all; null the whole group
    has_data = career_starts > 0
    entry["career_starts"] = career_starts if has_data else None
    entry["career_wins"] = career_wins if has_data else None
    entry["career_seconds"] = career_seconds if has_data else None
    entry["career_thirds"] = career_thirds if has_data else None
    entry["career_earnings"] = career_earnings if has_data else None

    has_surface = surface_starts > 0
    entry["surface_starts"] = surface_starts if has_surface else None
    entry["surface_wins"] = surface_wins if has_surface else None
    entry["surface_seconds"] = surface_seconds if has_surface else None
    entry["surface_thirds"] = surface_thirds if has_surface else None


def _poc_int(text: str | None) -> int | None:
    """For point-of-call positions, treat zero as null (no call at this point)."""
    val = safe_int(text)
    return val if val else None


def _parse_date(date_str: str | None) -> date | None:
    """Parse Equibase date '2023-04-29+00:00' → datetime.date."""
    if not date_str:
        return None
    return date.fromisoformat(date_str.split("+")[0].split("T")[0])
