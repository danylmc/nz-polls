#!/usr/bin/env python3
"""
NZ Political Events Timeline

Contains structured data about political events that may correlate with polling shifts:
- Leadership changes
- Elections
- Crises (earthquakes, shootings, pandemics)
- Scandals

Used by analysis.py for event study analysis.
"""

from datetime import datetime
from typing import List, Dict, Optional


# Leadership changes and elections
LEADERSHIP_CHANGES = [
    {"date": "1997-12-08", "event": "Bolger → Shipley", "party": "National", "type": "leader_change"},
    {"date": "1999-11-27", "event": "Election: Shipley → Clark", "party": "Labour", "type": "election"},
    {"date": "2008-11-08", "event": "Election: Clark → Key", "party": "National", "type": "election"},
    {"date": "2016-12-12", "event": "Key → English", "party": "National", "type": "leader_change"},
    {"date": "2017-08-01", "event": "Little → Ardern", "party": "Labour", "type": "leader_change"},
    {"date": "2017-09-23", "event": "Election: Labour-NZF-Green coalition", "party": "Labour", "type": "election"},
    {"date": "2020-10-17", "event": "Ardern wins majority", "party": "Labour", "type": "election"},
    {"date": "2023-01-19", "event": "Ardern → Hipkins", "party": "Labour", "type": "leader_change"},
    {"date": "2023-10-14", "event": "Election: Hipkins → Luxon", "party": "National", "type": "election"},
]

# Major crises
CRISES = [
    {"date": "2011-02-22", "event": "Christchurch earthquake", "type": "crisis"},
    {"date": "2019-03-15", "event": "Christchurch mosque shootings", "type": "crisis"},
    {"date": "2020-03-25", "event": "COVID-19 lockdown begins", "type": "crisis"},
    {"date": "2021-08-17", "event": "Delta outbreak, second lockdown", "type": "crisis"},
]

# Political scandals
SCANDALS = [
    {"date": "2002-07-01", "event": "Corngate (GM crops)", "type": "scandal"},
    {"date": "2008-09-01", "event": "Winston Peters donations", "type": "scandal"},
    {"date": "2014-08-01", "event": "Dirty Politics", "type": "scandal"},
    {"date": "2020-07-01", "event": "National Party leaks", "type": "scandal"},
]

# Election dates by year
ELECTION_DATES = {
    1993: "1993-11-06",
    1996: "1996-10-12",
    1999: "1999-11-27",
    2002: "2002-07-27",
    2005: "2005-09-17",
    2008: "2008-11-08",
    2011: "2011-11-26",
    2014: "2014-09-20",
    2017: "2017-09-23",
    2020: "2020-10-17",
    2023: "2023-10-14",
    2026: "2026-11-21",  # Expected
}

# Incumbent party by election cycle (party in power at start of cycle)
INCUMBENTS = {
    1993: "National",  # Bolger
    1996: "National",  # Bolger/Shipley
    1999: "National",  # Shipley
    2002: "Labour",    # Clark
    2005: "Labour",    # Clark
    2008: "Labour",    # Clark
    2011: "National",  # Key
    2014: "National",  # Key
    2017: "National",  # English
    2020: "Labour",    # Ardern
    2023: "Labour",    # Hipkins
    2026: "National",  # Luxon
}


def get_all_events() -> List[Dict]:
    """Return all events sorted by date"""
    all_events = LEADERSHIP_CHANGES + CRISES + SCANDALS
    return sorted(all_events, key=lambda x: x["date"])


def get_events_in_range(start_date: str, end_date: str) -> List[Dict]:
    """Return events within a date range (ISO format: YYYY-MM-DD)"""
    events = get_all_events()
    return [e for e in events if start_date <= e["date"] <= end_date]


def get_events_by_type(event_type: str) -> List[Dict]:
    """Return events of a specific type"""
    all_events = get_all_events()
    return [e for e in all_events if e["type"] == event_type]


def get_election_date(year: int) -> Optional[str]:
    """Get election date for a given year"""
    return ELECTION_DATES.get(year)


def get_incumbent(year: int) -> Optional[str]:
    """Get incumbent party for an election year"""
    return INCUMBENTS.get(year)


def days_to_election(poll_date: str, election_year: int) -> Optional[int]:
    """Calculate days from poll date to election date"""
    election_date = ELECTION_DATES.get(election_year)
    if not election_date:
        return None

    try:
        poll_dt = datetime.strptime(poll_date, "%Y-%m-%d")
        election_dt = datetime.strptime(election_date, "%Y-%m-%d")
        return (election_dt - poll_dt).days
    except ValueError:
        return None


def get_election_cycle(poll_date: str) -> Optional[int]:
    """Determine which election cycle a poll belongs to"""
    try:
        poll_dt = datetime.strptime(poll_date, "%Y-%m-%d")
    except ValueError:
        return None

    # Find the next election after this poll
    for year in sorted(ELECTION_DATES.keys()):
        election_date = ELECTION_DATES[year]
        election_dt = datetime.strptime(election_date, "%Y-%m-%d")
        if election_dt >= poll_dt:
            return year

    return None


def get_events_near_date(target_date: str, days_window: int = 30) -> List[Dict]:
    """Find events within N days of a target date"""
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        return []

    results = []
    for event in get_all_events():
        event_dt = datetime.strptime(event["date"], "%Y-%m-%d")
        delta = abs((event_dt - target_dt).days)
        if delta <= days_window:
            results.append({**event, "days_from_target": delta})

    return results


if __name__ == "__main__":
    # Print summary of events
    print("NZ Political Events Timeline")
    print("=" * 50)

    print("\nLeadership Changes & Elections:")
    for event in LEADERSHIP_CHANGES:
        print(f"  {event['date']}: {event['event']} ({event['party']})")

    print("\nCrises:")
    for event in CRISES:
        print(f"  {event['date']}: {event['event']}")

    print("\nScandals:")
    for event in SCANDALS:
        print(f"  {event['date']}: {event['event']}")

    print("\nElection Dates:")
    for year, date in sorted(ELECTION_DATES.items()):
        incumbent = INCUMBENTS.get(year, "Unknown")
        print(f"  {year}: {date} (Incumbent: {incumbent})")
