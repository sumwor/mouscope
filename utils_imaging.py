from datetime import datetime


# helper function to convert ISO timestamp to milliseconds since midnight
def iso_to_timeofday(ts):
    # truncate microseconds to 6 digits for Python
    if '.' in ts:
        main, frac = ts.split('.')
        if '-' in frac or '+' in frac:  # handle timezone offset
            frac, tz = frac[:-6], frac[-6:]
        else:
            tz = ''
        frac = frac[:6]  # keep 6 digits
        ts = main + '.' + frac + tz
    dt = datetime.fromisoformat(ts)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return (dt - midnight).total_seconds()