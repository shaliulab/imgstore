from time import time
import tzlocal
import re
import datetime
import time
import pytz
import os.path
import dateutil

from .path import get_fullpath

def parse_old_time(allmd, basedir):

    fullpath = get_fullpath(basedir)

    dt = tz = None
    # we don't know the local timezone, so assume it is local
    if 'timezone' in allmd:
        # noinspection PyBroadException
        try:
            tz = pytz.timezone(allmd['timezone'])
        except Exception:
            pass
    if tz is None:
        tz = tzlocal.get_localzone()

    # first the filename
    m = re.match(r"""(.*)(20[\d]{6}_\d{6}).*""", os.path.basename(basedir))
    if m:
        name, datestr = m.groups()
        # ive always been careful to make the files named with the local time
        time_tuple = time.strptime(datestr, '%Y%m%d_%H%M%S')
        _dt = datetime.datetime(*(time_tuple[0:6]))
        dt = tz.localize(_dt).astimezone(pytz.utc)

    # then the modification time of the file
    if dt is None:
        # file modifications are local time
        ts = os.path.getmtime(fullpath)
        dt = datetime.datetime.fromtimestamp(ts, tz=tzlocal.get_localzone()).astimezone(pytz.utc)
    
    return dt, tz

def parse_new_time(smd, dt):
    # ensure that created_utc always has the pytz.utc timezone object because fuck you python
    # and fuck you dateutil for having different UTC tz objects
    # https://github.com/dateutil/dateutil/issues/131
    try:
        dt = dt.astimezone(pytz.utc)  # aware object can be in any timezone
    except ValueError:  # naive
        dt = dt.replace(tzinfo=pytz.utc)  # d must be in UTC
    
    tz = pytz.timezone(smd['timezone_local'])
    
    return dt, tz
