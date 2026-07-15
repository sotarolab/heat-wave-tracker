"""
src/heat/stations.py
=====================
165 major CONUS ASOS stations with lat/lon for map markers and GFS
lookups, keyed by 4-letter ICAO code.

Two entries (Huntington WV and Flagstaff AZ) originally had wrong ICAO
codes, found while resolving stations against IEM's station metadata
API for the historical archive (see src/heat/historical.py). Both were
silently returning 0 live observations before the fix, since the wrong
codes did not match any real IEM record under any network. Corrected
in place here, with a comment at each entry. A third station, KPBI
(West Palm Beach Intl), has a correct ICAO code that IEM simply does
not index under that string, that one is handled as a fetch-only alias
in src/heat/asos.py instead of a rename here, since KPBI is the code
that should be shown to users.
"""

MAJOR_CONUS_STATIONS = [
    # ── Northeast ─────────────────────────────────────────────────────────────
    {"id": "KBOS", "name": "Boston Logan",         "lat": 42.363, "lon": -71.006, "state": "MA", "tz": "America/New_York"},
    {"id": "KPWM", "name": "Portland ME",           "lat": 43.646, "lon": -70.309, "state": "ME", "tz": "America/New_York"},
    {"id": "KBTV", "name": "Burlington VT",         "lat": 44.472, "lon": -73.150, "state": "VT", "tz": "America/New_York"},
    {"id": "KBGR", "name": "Bangor ME",             "lat": 44.807, "lon": -68.828, "state": "ME", "tz": "America/New_York"},
    {"id": "KPVD", "name": "Providence RI",         "lat": 41.724, "lon": -71.434, "state": "RI", "tz": "America/New_York"},
    {"id": "KBDL", "name": "Hartford CT",           "lat": 41.939, "lon": -72.683, "state": "CT", "tz": "America/New_York"},
    {"id": "KHVN", "name": "New Haven CT",          "lat": 41.264, "lon": -72.887, "state": "CT", "tz": "America/New_York"},
    {"id": "KJFK", "name": "JFK NY",                "lat": 40.639, "lon": -73.779, "state": "NY", "tz": "America/New_York"},
    {"id": "KLGA", "name": "LaGuardia NY",          "lat": 40.777, "lon": -73.873, "state": "NY", "tz": "America/New_York"},
    {"id": "KEWR", "name": "Newark NJ",             "lat": 40.693, "lon": -74.169, "state": "NJ", "tz": "America/New_York"},
    {"id": "KTEB", "name": "Teterboro NJ",          "lat": 40.850, "lon": -74.061, "state": "NJ", "tz": "America/New_York"},
    {"id": "KALB", "name": "Albany NY",             "lat": 42.748, "lon": -73.802, "state": "NY", "tz": "America/New_York"},
    {"id": "KBUF", "name": "Buffalo NY",            "lat": 42.940, "lon": -78.732, "state": "NY", "tz": "America/New_York"},
    {"id": "KSYR", "name": "Syracuse NY",           "lat": 43.111, "lon": -76.106, "state": "NY", "tz": "America/New_York"},
    {"id": "KROC", "name": "Rochester NY",          "lat": 43.119, "lon": -77.672, "state": "NY", "tz": "America/New_York"},
    {"id": "KPHL", "name": "Philadelphia PA",       "lat": 39.872, "lon": -75.241, "state": "PA", "tz": "America/New_York"},
    {"id": "KPIT", "name": "Pittsburgh PA",         "lat": 40.491, "lon": -80.233, "state": "PA", "tz": "America/New_York"},
    {"id": "KABE", "name": "Allentown PA",          "lat": 40.652, "lon": -75.440, "state": "PA", "tz": "America/New_York"},
    {"id": "KAVP", "name": "Scranton PA",           "lat": 41.338, "lon": -75.724, "state": "PA", "tz": "America/New_York"},
    # ── Mid-Atlantic / DC ─────────────────────────────────────────────────────
    {"id": "KBWI", "name": "BWI MD",               "lat": 39.175, "lon": -76.668, "state": "MD", "tz": "America/New_York"},
    {"id": "KDCA", "name": "Reagan Natl DC",        "lat": 38.852, "lon": -77.038, "state": "DC", "tz": "America/New_York"},
    {"id": "KIAD", "name": "Dulles VA",             "lat": 38.944, "lon": -77.456, "state": "VA", "tz": "America/New_York"},
    {"id": "KADW", "name": "Andrews AFB MD",        "lat": 38.811, "lon": -76.867, "state": "MD", "tz": "America/New_York"},
    {"id": "KORF", "name": "Norfolk VA",            "lat": 36.898, "lon": -76.012, "state": "VA", "tz": "America/New_York"},
    {"id": "KRIC", "name": "Richmond VA",           "lat": 37.505, "lon": -77.319, "state": "VA", "tz": "America/New_York"},
    {"id": "KCHO", "name": "Charlottesville VA",    "lat": 38.139, "lon": -78.452, "state": "VA", "tz": "America/New_York"},
    {"id": "KCKB", "name": "Clarksburg WV",         "lat": 39.346, "lon": -80.228, "state": "WV", "tz": "America/New_York"},
    # ICAO is KHTS (Tri-State/Milton J. Ferguson Field), not KHTW - "KHTW"
    # isn't a real IEM ASOS station under any network, so this station was
    # silently returning 0 observations in the live app until caught while
    # building the historical climate archive.
    {"id": "KHTS", "name": "Huntington WV",         "lat": 38.367, "lon": -82.558, "state": "WV", "tz": "America/New_York"},
    # ── Southeast ─────────────────────────────────────────────────────────────
    {"id": "KRDU", "name": "Raleigh-Durham NC",    "lat": 35.877, "lon": -78.787, "state": "NC", "tz": "America/New_York"},
    {"id": "KGSO", "name": "Greensboro NC",         "lat": 36.097, "lon": -79.937, "state": "NC", "tz": "America/New_York"},
    {"id": "KCLT", "name": "Charlotte NC",          "lat": 35.214, "lon": -80.943, "state": "NC", "tz": "America/New_York"},
    {"id": "KAVL", "name": "Asheville NC",          "lat": 35.436, "lon": -82.542, "state": "NC", "tz": "America/New_York"},
    {"id": "KCAE", "name": "Columbia SC",           "lat": 33.938, "lon": -81.120, "state": "SC", "tz": "America/New_York"},
    {"id": "KCHS", "name": "Charleston SC",         "lat": 32.899, "lon": -80.041, "state": "SC", "tz": "America/New_York"},
    {"id": "KSAV", "name": "Savannah GA",           "lat": 32.128, "lon": -81.202, "state": "GA", "tz": "America/New_York"},
    {"id": "KJAX", "name": "Jacksonville FL",       "lat": 30.494, "lon": -81.688, "state": "FL", "tz": "America/New_York"},
    {"id": "KGNV", "name": "Gainesville FL",        "lat": 29.690, "lon": -82.272, "state": "FL", "tz": "America/New_York"},
    {"id": "KTLH", "name": "Tallahassee FL",        "lat": 30.396, "lon": -84.353, "state": "FL", "tz": "America/New_York"},
    {"id": "KMCO", "name": "Orlando FL",            "lat": 28.429, "lon": -81.309, "state": "FL", "tz": "America/New_York"},
    {"id": "KTPA", "name": "Tampa FL",              "lat": 27.975, "lon": -82.533, "state": "FL", "tz": "America/New_York"},
    {"id": "KRSW", "name": "Fort Myers FL",         "lat": 26.536, "lon": -81.755, "state": "FL", "tz": "America/New_York"},
    {"id": "KMIA", "name": "Miami FL",              "lat": 25.796, "lon": -80.287, "state": "FL", "tz": "America/New_York"},
    {"id": "KFLL", "name": "Fort Lauderdale FL",   "lat": 26.073, "lon": -80.149, "state": "FL", "tz": "America/New_York"},
    {"id": "KPBI", "name": "West Palm Beach FL",   "lat": 26.683, "lon": -80.095, "state": "FL", "tz": "America/New_York"},
    {"id": "KDAB", "name": "Daytona Beach FL",     "lat": 29.180, "lon": -81.058, "state": "FL", "tz": "America/New_York"},
    {"id": "KPNS", "name": "Pensacola FL",          "lat": 30.473, "lon": -87.186, "state": "FL", "tz": "America/Chicago"},
    {"id": "KMOB", "name": "Mobile AL",             "lat": 30.691, "lon": -88.243, "state": "AL", "tz": "America/Chicago"},
    {"id": "KHSV", "name": "Huntsville AL",         "lat": 34.637, "lon": -86.775, "state": "AL", "tz": "America/Chicago"},
    {"id": "KBHM", "name": "Birmingham AL",         "lat": 33.563, "lon": -86.754, "state": "AL", "tz": "America/Chicago"},
    {"id": "KDHN", "name": "Dothan AL",             "lat": 31.321, "lon": -85.449, "state": "AL", "tz": "America/Chicago"},
    {"id": "KATL", "name": "Atlanta GA",            "lat": 33.637, "lon": -84.428, "state": "GA", "tz": "America/New_York"},
    {"id": "KAGS", "name": "Augusta GA",            "lat": 33.370, "lon": -81.965, "state": "GA", "tz": "America/New_York"},
    {"id": "KCHA", "name": "Chattanooga TN",        "lat": 35.035, "lon": -85.203, "state": "TN", "tz": "America/New_York"},
    {"id": "KBNA", "name": "Nashville TN",          "lat": 36.124, "lon": -86.678, "state": "TN", "tz": "America/Chicago"},
    {"id": "KTYS", "name": "Knoxville TN",          "lat": 35.811, "lon": -83.994, "state": "TN", "tz": "America/New_York"},
    {"id": "KMEM", "name": "Memphis TN",            "lat": 35.042, "lon": -89.977, "state": "TN", "tz": "America/Chicago"},
    # ── Deep South / Gulf ─────────────────────────────────────────────────────
    {"id": "KJAN", "name": "Jackson MS",            "lat": 32.311, "lon": -90.076, "state": "MS", "tz": "America/Chicago"},
    {"id": "KGPT", "name": "Gulfport MS",           "lat": 30.407, "lon": -89.070, "state": "MS", "tz": "America/Chicago"},
    {"id": "KMSY", "name": "New Orleans LA",        "lat": 29.993, "lon": -90.258, "state": "LA", "tz": "America/Chicago"},
    {"id": "KBTR", "name": "Baton Rouge LA",        "lat": 30.533, "lon": -91.150, "state": "LA", "tz": "America/Chicago"},
    {"id": "KLCH", "name": "Lake Charles LA",       "lat": 30.126, "lon": -93.223, "state": "LA", "tz": "America/Chicago"},
    {"id": "KSHV", "name": "Shreveport LA",         "lat": 32.447, "lon": -93.826, "state": "LA", "tz": "America/Chicago"},
    # ── Texas ─────────────────────────────────────────────────────────────────
    {"id": "KIAH", "name": "Houston Bush TX",       "lat": 29.984, "lon": -95.342, "state": "TX", "tz": "America/Chicago"},
    {"id": "KHOU", "name": "Houston Hobby TX",      "lat": 29.645, "lon": -95.279, "state": "TX", "tz": "America/Chicago"},
    {"id": "KDFW", "name": "Dallas-FW TX",          "lat": 32.897, "lon": -97.038, "state": "TX", "tz": "America/Chicago"},
    {"id": "KDAL", "name": "Dallas Love TX",        "lat": 32.847, "lon": -96.852, "state": "TX", "tz": "America/Chicago"},
    {"id": "KSAT", "name": "San Antonio TX",        "lat": 29.533, "lon": -98.470, "state": "TX", "tz": "America/Chicago"},
    {"id": "KAUS", "name": "Austin TX",             "lat": 30.197, "lon": -97.669, "state": "TX", "tz": "America/Chicago"},
    {"id": "KACT", "name": "Waco TX",               "lat": 31.611, "lon": -97.230, "state": "TX", "tz": "America/Chicago"},
    {"id": "KCRP", "name": "Corpus Christi TX",     "lat": 27.770, "lon": -97.502, "state": "TX", "tz": "America/Chicago"},
    {"id": "KAMA", "name": "Amarillo TX",           "lat": 35.219, "lon": -101.706, "state": "TX", "tz": "America/Chicago"},
    {"id": "KELP", "name": "El Paso TX",            "lat": 31.807, "lon": -106.378, "state": "TX", "tz": "America/Denver"},
    {"id": "KMAF", "name": "Midland TX",            "lat": 31.943, "lon": -102.213, "state": "TX", "tz": "America/Chicago"},
    {"id": "KSJT", "name": "San Angelo TX",         "lat": 31.358, "lon": -100.496, "state": "TX", "tz": "America/Chicago"},
    {"id": "KLBB", "name": "Lubbock TX",            "lat": 33.664, "lon": -101.823, "state": "TX", "tz": "America/Chicago"},
    # ── Oklahoma ──────────────────────────────────────────────────────────────
    {"id": "KOKC", "name": "Oklahoma City OK",      "lat": 35.393, "lon": -97.601, "state": "OK", "tz": "America/Chicago"},
    {"id": "KTUL", "name": "Tulsa OK",              "lat": 36.199, "lon": -95.888, "state": "OK", "tz": "America/Chicago"},
    {"id": "KLAW", "name": "Lawton OK",             "lat": 34.568, "lon": -98.417, "state": "OK", "tz": "America/Chicago"},
    # ── Kansas ────────────────────────────────────────────────────────────────
    {"id": "KICT", "name": "Wichita KS",            "lat": 37.650, "lon": -97.433, "state": "KS", "tz": "America/Chicago"},
    {"id": "KTOP", "name": "Topeka KS",             "lat": 39.068, "lon": -95.623, "state": "KS", "tz": "America/Chicago"},
    {"id": "KGLD", "name": "Goodland KS",           "lat": 39.370, "lon": -101.699, "state": "KS", "tz": "America/Denver"},
    # ── Missouri / Iowa / Nebraska ────────────────────────────────────────────
    {"id": "KSTL", "name": "St. Louis MO",          "lat": 38.748, "lon": -90.370, "state": "MO", "tz": "America/Chicago"},
    {"id": "KMCI", "name": "Kansas City MO",        "lat": 39.297, "lon": -94.714, "state": "MO", "tz": "America/Chicago"},
    {"id": "KSGF", "name": "Springfield MO",        "lat": 37.245, "lon": -93.389, "state": "MO", "tz": "America/Chicago"},
    {"id": "KCOU", "name": "Columbia MO",           "lat": 38.818, "lon": -92.220, "state": "MO", "tz": "America/Chicago"},
    {"id": "KDSM", "name": "Des Moines IA",         "lat": 41.534, "lon": -93.663, "state": "IA", "tz": "America/Chicago"},
    {"id": "KCID", "name": "Cedar Rapids IA",       "lat": 41.884, "lon": -91.711, "state": "IA", "tz": "America/Chicago"},
    {"id": "KSUX", "name": "Sioux City IA",         "lat": 42.402, "lon": -96.385, "state": "IA", "tz": "America/Chicago"},
    {"id": "KOMA", "name": "Omaha NE",              "lat": 41.303, "lon": -95.894, "state": "NE", "tz": "America/Chicago"},
    {"id": "KLNK", "name": "Lincoln NE",            "lat": 40.851, "lon": -96.759, "state": "NE", "tz": "America/Chicago"},
    {"id": "KGRI", "name": "Grand Island NE",       "lat": 40.967, "lon": -98.310, "state": "NE", "tz": "America/Chicago"},
    # ── Great Plains (North) ──────────────────────────────────────────────────
    {"id": "KFAR", "name": "Fargo ND",              "lat": 46.921, "lon": -96.816, "state": "ND", "tz": "America/Chicago"},
    {"id": "KBIS", "name": "Bismarck ND",           "lat": 46.773, "lon": -100.747, "state": "ND", "tz": "America/Chicago"},
    {"id": "KFSD", "name": "Sioux Falls SD",        "lat": 43.582, "lon": -96.742, "state": "SD", "tz": "America/Chicago"},
    {"id": "KRAP", "name": "Rapid City SD",         "lat": 44.045, "lon": -103.057, "state": "SD", "tz": "America/Denver"},
    {"id": "KABR", "name": "Aberdeen SD",           "lat": 45.450, "lon": -98.422, "state": "SD", "tz": "America/Chicago"},
    # ── Midwest / Great Lakes ─────────────────────────────────────────────────
    {"id": "KCLE", "name": "Cleveland OH",          "lat": 41.411, "lon": -81.849, "state": "OH", "tz": "America/New_York"},
    {"id": "KCMH", "name": "Columbus OH",           "lat": 39.998, "lon": -82.892, "state": "OH", "tz": "America/New_York"},
    {"id": "KDAY", "name": "Dayton OH",             "lat": 39.903, "lon": -84.220, "state": "OH", "tz": "America/New_York"},
    {"id": "KTOL", "name": "Toledo OH",             "lat": 41.588, "lon": -83.808, "state": "OH", "tz": "America/New_York"},
    {"id": "KDET", "name": "Detroit MI",            "lat": 42.212, "lon": -83.351, "state": "MI", "tz": "America/Detroit"},
    {"id": "KGRR", "name": "Grand Rapids MI",       "lat": 42.880, "lon": -85.523, "state": "MI", "tz": "America/Detroit"},
    {"id": "KLAN", "name": "Lansing MI",            "lat": 42.778, "lon": -84.588, "state": "MI", "tz": "America/Detroit"},
    {"id": "KFNT", "name": "Flint MI",              "lat": 42.966, "lon": -83.743, "state": "MI", "tz": "America/Detroit"},
    {"id": "KIND",  "name": "Indianapolis IN",      "lat": 39.717, "lon": -86.295, "state": "IN", "tz": "America/Indiana/Indianapolis"},
    {"id": "KFWA", "name": "Fort Wayne IN",         "lat": 40.978, "lon": -85.195, "state": "IN", "tz": "America/Indiana/Indianapolis"},
    {"id": "KEVV", "name": "Evansville IN",         "lat": 38.037, "lon": -87.532, "state": "IN", "tz": "America/Chicago"},
    {"id": "KORD", "name": "Chicago O'Hare IL",     "lat": 41.978, "lon": -87.905, "state": "IL", "tz": "America/Chicago"},
    {"id": "KMDW", "name": "Chicago Midway IL",     "lat": 41.786, "lon": -87.752, "state": "IL", "tz": "America/Chicago"},
    {"id": "KSPI", "name": "Springfield IL",        "lat": 39.844, "lon": -89.678, "state": "IL", "tz": "America/Chicago"},
    {"id": "KMKE", "name": "Milwaukee WI",          "lat": 42.947, "lon": -87.897, "state": "WI", "tz": "America/Chicago"},
    {"id": "KMSN", "name": "Madison WI",            "lat": 43.140, "lon": -89.337, "state": "WI", "tz": "America/Chicago"},
    {"id": "KGRB", "name": "Green Bay WI",          "lat": 44.485, "lon": -88.130, "state": "WI", "tz": "America/Chicago"},
    {"id": "KDLH", "name": "Duluth MN",             "lat": 46.842, "lon": -92.194, "state": "MN", "tz": "America/Chicago"},
    {"id": "KMSP", "name": "Minneapolis MN",        "lat": 44.882, "lon": -93.222, "state": "MN", "tz": "America/Chicago"},
    {"id": "KRST", "name": "Rochester MN",          "lat": 43.908, "lon": -92.500, "state": "MN", "tz": "America/Chicago"},
    # ── Kentucky / Arkansas ───────────────────────────────────────────────────
    {"id": "KSDF", "name": "Louisville KY",         "lat": 38.174, "lon": -85.736, "state": "KY", "tz": "America/Kentucky/Louisville"},
    {"id": "KLEX", "name": "Lexington KY",          "lat": 38.036, "lon": -84.606, "state": "KY", "tz": "America/New_York"},
    {"id": "KBWG", "name": "Bowling Green KY",      "lat": 36.975, "lon": -86.419, "state": "KY", "tz": "America/Chicago"},
    {"id": "KLIT", "name": "Little Rock AR",        "lat": 34.729, "lon": -92.224, "state": "AR", "tz": "America/Chicago"},
    {"id": "KFSM", "name": "Fort Smith AR",         "lat": 35.337, "lon": -94.367, "state": "AR", "tz": "America/Chicago"},
    {"id": "KTXK", "name": "Texarkana AR",          "lat": 33.453, "lon": -94.013, "state": "AR", "tz": "America/Chicago"},
    # ── Mountain West ─────────────────────────────────────────────────────────
    {"id": "KDEN", "name": "Denver CO",             "lat": 39.856, "lon": -104.674, "state": "CO", "tz": "America/Denver"},
    {"id": "KCOS", "name": "Colorado Springs CO",   "lat": 38.806, "lon": -104.701, "state": "CO", "tz": "America/Denver"},
    {"id": "KPUB", "name": "Pueblo CO",             "lat": 38.289, "lon": -104.497, "state": "CO", "tz": "America/Denver"},
    {"id": "KGJT", "name": "Grand Junction CO",     "lat": 39.124, "lon": -108.527, "state": "CO", "tz": "America/Denver"},
    {"id": "KABQ", "name": "Albuquerque NM",        "lat": 35.040, "lon": -106.610, "state": "NM", "tz": "America/Denver"},
    {"id": "KTUS", "name": "Tucson AZ",             "lat": 32.117, "lon": -110.941, "state": "AZ", "tz": "America/Phoenix"},
    {"id": "KPHX", "name": "Phoenix AZ",            "lat": 33.438, "lon": -112.013, "state": "AZ", "tz": "America/Phoenix"},
    {"id": "KYUM", "name": "Yuma AZ",               "lat": 32.657, "lon": -114.606, "state": "AZ", "tz": "America/Phoenix"},
    # ICAO is KFLG (Flagstaff Pulliam), not KFGZ - same silent-0-obs issue
    # as KHTS above, caught the same way.
    {"id": "KFLG", "name": "Flagstaff AZ",          "lat": 35.138, "lon": -111.671, "state": "AZ", "tz": "America/Phoenix"},
    {"id": "KSLC", "name": "Salt Lake City UT",     "lat": 40.788, "lon": -111.980, "state": "UT", "tz": "America/Denver"},
    {"id": "KLAS", "name": "Las Vegas NV",          "lat": 36.080, "lon": -115.152, "state": "NV", "tz": "America/Los_Angeles"},
    {"id": "KRNO", "name": "Reno NV",               "lat": 39.499, "lon": -119.768, "state": "NV", "tz": "America/Los_Angeles"},
    {"id": "KBOI", "name": "Boise ID",              "lat": 43.564, "lon": -116.223, "state": "ID", "tz": "America/Boise"},
    {"id": "KTWF", "name": "Twin Falls ID",         "lat": 42.482, "lon": -114.488, "state": "ID", "tz": "America/Boise"},
    {"id": "KLWS", "name": "Lewiston ID",           "lat": 46.374, "lon": -117.015, "state": "ID", "tz": "America/Los_Angeles"},
    # ── Montana / Wyoming ────────────────────────────────────────────────────
    {"id": "KBZN", "name": "Bozeman MT",            "lat": 45.778, "lon": -111.154, "state": "MT", "tz": "America/Denver"},
    {"id": "KBIL", "name": "Billings MT",           "lat": 45.808, "lon": -108.543, "state": "MT", "tz": "America/Denver"},
    {"id": "KGTF", "name": "Great Falls MT",        "lat": 47.480, "lon": -111.371, "state": "MT", "tz": "America/Denver"},
    {"id": "KMSO", "name": "Missoula MT",           "lat": 46.916, "lon": -114.091, "state": "MT", "tz": "America/Denver"},
    {"id": "KHLN", "name": "Helena MT",             "lat": 46.607, "lon": -111.983, "state": "MT", "tz": "America/Denver"},
    {"id": "KCYS", "name": "Cheyenne WY",           "lat": 41.156, "lon": -104.812, "state": "WY", "tz": "America/Denver"},
    {"id": "KCOD", "name": "Cody WY",               "lat": 44.520, "lon": -109.023, "state": "WY", "tz": "America/Denver"},
    # ── Pacific Northwest ─────────────────────────────────────────────────────
    {"id": "KSEA", "name": "Seattle-Tacoma WA",     "lat": 47.449, "lon": -122.309, "state": "WA", "tz": "America/Los_Angeles"},
    {"id": "KBLI", "name": "Bellingham WA",         "lat": 48.793, "lon": -122.538, "state": "WA", "tz": "America/Los_Angeles"},
    {"id": "KOLM", "name": "Olympia WA",            "lat": 46.970, "lon": -122.903, "state": "WA", "tz": "America/Los_Angeles"},
    {"id": "KGEG", "name": "Spokane WA",            "lat": 47.620, "lon": -117.534, "state": "WA", "tz": "America/Los_Angeles"},
    {"id": "KYKM", "name": "Yakima WA",             "lat": 46.568, "lon": -120.545, "state": "WA", "tz": "America/Los_Angeles"},
    {"id": "KPDX", "name": "Portland OR",           "lat": 45.589, "lon": -122.598, "state": "OR", "tz": "America/Los_Angeles"},
    {"id": "KEUG", "name": "Eugene OR",             "lat": 44.125, "lon": -123.212, "state": "OR", "tz": "America/Los_Angeles"},
    {"id": "KMFR", "name": "Medford OR",            "lat": 42.374, "lon": -122.873, "state": "OR", "tz": "America/Los_Angeles"},
    {"id": "KRDM", "name": "Redmond OR",            "lat": 44.254, "lon": -121.150, "state": "OR", "tz": "America/Los_Angeles"},
    # ── California ────────────────────────────────────────────────────────────
    {"id": "KSFO", "name": "San Francisco CA",      "lat": 37.619, "lon": -122.375, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KOAK", "name": "Oakland CA",            "lat": 37.721, "lon": -122.221, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KSJC", "name": "San Jose CA",           "lat": 37.362, "lon": -121.929, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KSMF", "name": "Sacramento CA",         "lat": 38.695, "lon": -121.591, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KFAT", "name": "Fresno CA",             "lat": 36.776, "lon": -119.718, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KBFL", "name": "Bakersfield CA",        "lat": 35.434, "lon": -119.057, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KLAX", "name": "Los Angeles CA",        "lat": 33.943, "lon": -118.408, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KBUR", "name": "Burbank CA",            "lat": 34.201, "lon": -118.359, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KLGB", "name": "Long Beach CA",         "lat": 33.818, "lon": -118.152, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KSNA", "name": "Santa Ana CA",          "lat": 33.676, "lon": -117.868, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KSAN", "name": "San Diego CA",          "lat": 32.734, "lon": -117.190, "state": "CA", "tz": "America/Los_Angeles"},
    {"id": "KPSP", "name": "Palm Springs CA",       "lat": 33.829, "lon": -116.506, "state": "CA", "tz": "America/Los_Angeles"},
]

# Quick lookup by station ID
_STATION_LOOKUP = {s["id"]: s for s in MAJOR_CONUS_STATIONS}


def get_station(station_id: str) -> dict | None:
    """Look up one station's catalog entry by ICAO code.

    Parameters
    ----------
    station_id : str
        4-letter ICAO station code, e.g. "KDCA".

    Returns
    -------
    dict or None
        None if station_id is not in MAJOR_CONUS_STATIONS. Otherwise
        the matching dict, with keys id, name, lat, lon, state, tz.
    """
    return _STATION_LOOKUP.get(station_id)
