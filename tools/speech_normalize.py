#!/usr/bin/env python3
"""Text normalization for speech: numbers, symbols and abbreviations written out as words, before the sentence split
(validation/speech-normalize-v1, the "rules" arm). American English. Deterministic; nothing but the matched spans
changes. The style follows the experiment's guide: cardinals without "and"; years, decades and house numbers in pairs;
"point" decimals; "three dollars and fifty cents"; "seven forty-five", "seven oh five", "nine o'clock"; phone numbers
digit by digit ("oh" for zero); titles and abbreviations in full; "a.m."/"p.m." and acronyms left as written.

Matching is ASCII ((?a): digits, word characters, spaces and word boundaries are ASCII), so the native port (lfm/speech_normalize.hpp) can match it
byte for byte. Changes after validation/speech-normalize-v1 chose these rules (its held-out misses): "AM"/"PM" in
capitals count as a.m./p.m. after a time; "911" and a 5-digit number after "zip" or "postal" are read digit by digit;
a degree sign without C or F no longer takes the space after it ("20° today" read "twenty degreestoday").

The rules, in order (each sees the previous one's output, and the words it writes hold no digits):
  phone numbers; money; numeric dates (M/D/YYYY); times; month abbreviations before a number; month-day dates;
  temperatures; percentages; addresses (house number + capitalized name + street suffix); decades; numbers with units;
  ordinals; fractions; ranges; "#N" and "No. N"; math symbols between spaces; negative numbers; decimals; integers
  (years by context, long or zero-led digit strings digit by digit, otherwise cardinals); titles and abbreviations.
An abbreviation or unit that ends a sentence (end of text, or a space and a capital follow) keeps its period; titles
and "vs." (a name follows them) never do.
Usage: speech_normalize.py "TEXT"   (prints the normalized text)
"""
import re
import sys

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (10 ** 3, "thousand")]
ORDINAL_WORDS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                 "nine": "ninth", "twelve": "twelfth"}
MONTHS = "January February March April May June July August September October November December".split()
MONTH_ABBR = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "Jun": "June", "Jul": "July",
              "Aug": "August", "Sep": "September", "Sept": "September", "Oct": "October", "Nov": "November",
              "Dec": "December"}
YEAR_CUES = {"in", "since", "by", "from", "until", "till", "of", "during", "year", "circa", "before", "after", "around",
             "early", "late", "mid", "spring", "summer", "fall", "autumn", "winter", "through", "to"}
STREET = {"St": "Street", "Street": "Street", "Ave": "Avenue", "Avenue": "Avenue", "Rd": "Road", "Road": "Road",
          "Blvd": "Boulevard", "Boulevard": "Boulevard", "Dr": "Drive", "Drive": "Drive", "Ln": "Lane", "Lane": "Lane",
          "Way": "Way", "Ct": "Court", "Court": "Court", "Pl": "Place", "Place": "Place"}
UNITS = {"km": ("kilometer", "kilometers"), "kg": ("kilogram", "kilograms"), "cm": ("centimeter", "centimeters"),
         "mm": ("millimeter", "millimeters"), "mg": ("milligram", "milligrams"), "ml": ("milliliter", "milliliters"),
         "mph": ("mile per hour", "miles per hour"), "kph": ("kilometer per hour", "kilometers per hour"),
         "ft": ("foot", "feet"), "lb": ("pound", "pounds"), "lbs": ("pound", "pounds"), "oz": ("ounce", "ounces"),
         "mi": ("mile", "miles"), "GB": ("gigabyte", "gigabytes"), "MB": ("megabyte", "megabytes"),
         "kWh": ("kilowatt hour", "kilowatt hours"), "hrs": ("hour", "hours"), "hr": ("hour", "hours"),
         "min": ("minute", "minutes"), "sec": ("second", "seconds")}
CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
            "€": ("euro", "euros", "cent", "cents")}
TITLES = {"vs": "versus", "Mr": "Mister", "Mrs": "Missus", "Ms": "Miz", "Dr": "Doctor", "Prof": "Professor", "Mt": "Mount",
          "Jr": "Junior", "Sr": "Senior", "Gov": "Governor", "Sen": "Senator", "Rep": "Representative",
          "Gen": "General", "Capt": "Captain", "Lt": "Lieutenant", "Sgt": "Sergeant"}
ABBREVIATIONS = {"etc": "et cetera", "e.g": "for example", "i.e": "that is", "approx": "approximately",
                 "Apt": "Apartment", "Dept": "Department", "Inc": "Incorporated", "Corp": "Corporation",
                 "Co": "Company", "Ave": "Avenue", "Blvd": "Boulevard", "Rd": "Road", "est": "established",
                 "min": "minutes", "max": "maximum", "govt": "government"}


def cardinal(n):
    if n < 0:
        return "minus " + cardinal(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + cardinal(n % 100) if n % 100 else "")
    for value, name in SCALES:
        if n >= value:
            return cardinal(n // value) + " " + name + (" " + cardinal(n % value) if n % value else "")
    raise ValueError(n)


def ordinal(n):
    words = cardinal(n)
    head, sep, last = words.rpartition("-" if "-" in words.split(" ")[-1] else " ")
    if last in ORDINAL_WORDS:
        last = ORDINAL_WORDS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def pairs(n):
    """Years and house numbers: 1984 nineteen eighty-four, 1905 nineteen oh five, 1900 nineteen hundred, 2000 two
    thousand, 2005 two thousand five, 2026 twenty twenty-six, 221 two twenty-one, 100 one hundred."""
    if n < 100 or n % 1000 == 0 and n < 10000 or 2000 <= n < 2010:
        return cardinal(n)
    if n < 1000:
        return cardinal(n) if n % 100 == 0 else ONES[n // 100] + " " + (cardinal(n % 100) if n % 100 >= 10 else
                                                                         "oh " + ONES[n % 100])
    hi, lo = divmod(n, 100)
    if lo == 0:
        return cardinal(hi) + " hundred"
    return cardinal(hi) + " " + (cardinal(lo) if lo >= 10 else "oh " + ONES[lo])


def plural_pairs(words):
    """'nineteen ninety' -> 'nineteen nineties' (decades)."""
    head, sep, last = words.rpartition(" ")
    last = last[:-1] + "ies" if last.endswith("y") else last + "s"
    return head + sep + last


def digits(s, zero="zero"):
    return " ".join(zero if c == "0" else ONES[int(c)] for c in s)


def number(s):
    """A plain number string (commas allowed, a decimal part allowed) in words."""
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return cardinal(int(whole or "0")) + " point " + digits(frac)
    return cardinal(int(s))


def amount(value_words, one, many, is_one):
    return f"{value_words} {one if is_one else many}"


def normalize(text):
    t = text
    # phone numbers: 555-0142, (555) 123-4567, 555-123-4567
    t = re.sub(r"(?a)\(?\b(\d{3})\)?[-. ](\d{3})-(\d{4})\b",
               lambda m: ", ".join(digits(g, "oh") for g in m.groups()), t)
    t = re.sub(r"(?a)(?<![\d.,-])\b(\d{3})-(\d{4})\b(?![-\d])", lambda m: ", ".join(digits(g, "oh") for g in m.groups()), t)

    # money: $3.50, $129, $1.2 million, £5, €20.05
    def money(m):
        one, many, sub_one, sub_many = CURRENCY[m.group(1)]
        value, scale = m.group(2).replace(",", ""), m.group(4)
        if scale:
            return f"{number(value)} {scale} {many}"
        whole, _, frac = value.partition(".")
        cents = int((frac + "00")[:2]) if frac else 0
        words = amount(cardinal(int(whole)), one, many, int(whole) == 1) if int(whole) or not cents else ""
        if cents:
            words += (" and " if words else "") + amount(cardinal(cents), sub_one, sub_many, cents == 1)
        return words
    t = re.sub(r"(?a)([$£€])(\d[\d,]*(?:\.\d+)?)(\s+(million|billion|trillion|thousand)\b)?", money, t)
    # numeric dates (US month/day/year)
    t = re.sub(r"(?a)\b(1[0-2]|0?[1-9])/(3[01]|[12]\d|0?[1-9])/(\d{4})\b",
               lambda m: f"{MONTHS[int(m.group(1)) - 1]} {ordinal(int(m.group(2)))}, {pairs(int(m.group(3)))}", t)

    # times: 7:45, 9:00, 10:30 (o'clock only when no a.m./p.m. follows)
    def clock(m):
        h, mm = int(m.group(1)), int(m.group(2))
        if mm == 0:
            return cardinal(h) + ("" if m.group(3) else " o'clock")
        return cardinal(h) + " " + (cardinal(mm) if mm >= 10 else "oh " + ONES[mm])
    t = re.sub(r"(?a)\b([01]?\d|2[0-3]):([0-5]\d)\b(?=(\s*[aApP]\.?[mM]\b\.?)?)", clock, t)
    # month abbreviations before a number; month-day dates
    t = re.sub(r"(?a)\b(" + "|".join(MONTH_ABBR) + r")\.(?=\s+\d)", lambda m: MONTH_ABBR[m.group(1)], t)
    t = re.sub(r"(?a)\b(" + "|".join(MONTHS) + r")\s+(3[01]|[12]\d|0?[1-9])(st|nd|rd|th)?\b(?![\d:])",
               lambda m: f"{m.group(1)} {ordinal(int(m.group(2)))}", t)
    # temperatures: -5°C, 41°F, 20°
    t = re.sub(r"(?a)(?<![\w.])([-−])?(\d+(?:\.\d+)?)\s*°(?:\s*([CF])\b)?",
               lambda m: ("minus " if m.group(1) else "") + number(m.group(2)) + " degree" +
               ("" if m.group(2) == "1" else "s") + {"C": " Celsius", "F": " Fahrenheit", None: ""}[m.group(3)], t)
    # percentages
    t = re.sub(r"(?a)(?<![\w.])([-−])?(\d[\d,]*(?:\.\d+)?)\s*%",
               lambda m: ("minus " if m.group(1) else "") + number(m.group(2)) + " percent", t)
    # addresses: 1600 Main St., 221 Baker Street
    t = re.sub(r"(?a)\b(\d{1,5}) ((?:[A-Z][a-z]+ )+)(" + "|".join(STREET) + r")\b(\.)?",
               lambda m: f"{pairs(int(m.group(1)))} {m.group(2)}{STREET[m.group(3)]}" +
               ("." if m.group(4) and re.match(r"(?a)\s*$|\s+[A-Z]", t[m.end():]) else ""), t)
    # decades: 1990s, '90s handled as written
    t = re.sub(r"(?a)\b(1\d|20)(\d0)s\b", lambda m: plural_pairs(pairs(int(m.group(1) + m.group(2)))), t)

    # numbers with units: 12 km, 2,800 ft, 0.5 mg
    def unit(m):  # a unit's own period goes, unless it also ends the sentence
        one, many = UNITS[m.group(2)]
        end = "." if m.group(3) and re.match(r"(?a)\s*$|\s+[A-Z\"'“(]", t[m.end():]) else ""
        return number(m.group(1)) + " " + (one if m.group(1) == "1" else many) + end
    t = re.sub(r"(?a)(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(" + "|".join(sorted(UNITS, key=len, reverse=True)) + r")\b(\.)?",
               unit, t)
    # ordinals: 1st, 22nd, 3rd
    t = re.sub(r"(?a)\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), t)

    # fractions: 1/2, 3/4, 2/3
    def fraction(m):
        n, d = int(m.group(1)), int(m.group(2))
        if d == 2:
            return cardinal(n) + (" half" if n == 1 else " halves")
        if d == 4:
            return cardinal(n) + (" quarter" if n == 1 else " quarters")
        return cardinal(n) + " " + ordinal(d) + ("" if n == 1 else "s")
    t = re.sub(r"(?a)(?<![\w/.])(\d+)/(\d+)\b(?!/)", fraction, t)

    # ranges: 25-30, 1990-1995
    def span(m):
        a, b = int(m.group(1)), int(m.group(2))
        both_years = all(1000 <= v <= 2099 for v in (a, b))
        return (pairs(a) if both_years else cardinal(a)) + " to " + (pairs(b) if both_years else cardinal(b))
    t = re.sub(r"(?a)(?<![\w.,-])(\d+)\s?[-–]\s?(\d+)\b(?![-\d])", span, t)
    # "#12", "No. 14"
    t = re.sub(r"(?a)#\s?(\d+)\b", lambda m: "number " + cardinal(int(m.group(1))), t)
    t = re.sub(r"(?a)\bNo\.\s?(\d+)\b", lambda m: "number " + cardinal(int(m.group(1))), t)
    # math symbols with spaces around them; & anywhere between words
    t = re.sub(r"(?a)(?<=\s)=(?=\s)", "equals", t)
    t = re.sub(r"(?a)(?<=\s)\+(?=\s)", "plus", t)
    t = re.sub(r"(?a)(?<=\s)×(?=\s)", "times", t)
    t = re.sub(r"(?a)(?<=\s)&(?=\s)", "and", t)
    # negative numbers
    t = re.sub(r"(?a)(?<![\w.])[-−](?=\d)", "minus ", t)
    # decimals
    t = re.sub(r"(?a)(?<![\w.])(\d+)\.(\d+)\b", lambda m: number(m.group(0)), t)

    # integers: years by context, digit strings, cardinals
    def integer(m):
        s = m.group(0)
        if "," in s:
            return cardinal(int(s.replace(",", "")))
        if len(s) > 1 and s[0] == "0" or len(s) >= 7:
            return digits(s)
        n = int(s)
        before_words = re.findall(r"(?a)[A-Za-z]+", t[:m.start()][-30:])
        if s == "911" or len(s) == 5 and any(w.lower() in ("zip", "postal") for w in before_words[-3:]):
            return digits(s)  # an emergency number; a ZIP code ("zip code is 12345")
        if len(s) == 4 and 1000 <= n <= 2099:
            before = t[:m.start()].rstrip()
            prev = re.findall(r"(?a)[A-Za-z]+", before[-20:])
            after = t[m.end():m.end() + 2]
            if (prev and prev[-1].lower() in YEAR_CUES) or re.search(r"(?a)(" + "|".join(MONTHS) + r") \w+,?$", before) \
                    or re.match(r"(?a)\s*[.,;:!?]|\s*$", after) and not re.search(r"(?a)\d", before[-3:]):
                return pairs(n)
        return cardinal(n)
    t = re.sub(r"(?a)(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d,]*\d)|(?<![\w.,])\d+(?![\d,]*\d)", integer, t)

    # titles and abbreviations
    def abbreviation(m):
        word = m.group(1)
        follow = t[m.end():]
        if word == "St":  # Saint before a name, Street otherwise
            return "Saint" if re.match(r"(?a)\s+[A-Z]", follow) else "Street" + period(follow)
        if word == "Dr" and not re.match(r"(?a)\s+[A-Z]", follow):  # a name follows Doctor; a street name precedes Drive
            return "Drive" + period(follow)
        if word in TITLES:
            return TITLES[word]
        return ABBREVIATIONS[word] + period(follow)

    def period(follow):
        return "." if re.match(r"(?a)\s*$|\s+[A-Z\"'“(]", follow) else ""
    keys = sorted(set(TITLES) | set(ABBREVIATIONS) | {"St"}, key=len, reverse=True)
    t = re.sub(r"(?a)(?<![\w.])(" + "|".join(re.escape(k) for k in keys) + r")\.(?=[\s,;:!?)]|$)", abbreviation, t)
    t = re.sub(r"(?a)(?<![\w.])vs(?=\s)", "versus", t)
    return re.sub(r"(?a)[ \t]{2,}", " ", t)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(normalize(sys.argv[1]))
