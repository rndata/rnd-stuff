import logging
from datetime import date, datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

LOG = logging.getLogger(__name__)
MIN_YEAR = 1968
URL = "https://www.bullionbypost.eu/fixes/gold/USD/{year}/"
YEARS = list(range(MIN_YEAR, date.today().year + 1))

FF = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:94.0) Gecko/20100101 Firefox/94.0"


def read_year(year):
    LOG.info(f'Reading year {year}')
    resp = requests.get(URL.format(year=year), headers={'User-Agent': FF})
    resp.raise_for_status()
    return resp.text


def parse_year(text):
    soup = BeautifulSoup(text, 'html.parser')
    soup
    tables = soup.find_all('table')
    tbody = tables[2].tbody
    ch = tbody.find_all('tr', class_=None)

    dates = []
    prices = []
    for r in ch:
        date_tag = r.find('td', class_="fix-date")
        price_str1 = r.find(
            'td',
            class_="selected-currency"
        ).string
        price_str2 = r.find(
            'td',
            class_="selected-currency right-column"
        ).string
        d = datetime.strptime(date_tag.string, '%d %b %Y').date()

        price_str = price_str1
        if price_str == '-':
            price_str = price_str2

        if price_str == '-':
            continue

        price = float(price_str.replace(',', '.').replace('\xa0', ''))
        dates.append(d)
        prices.append(price)

    return dates, prices


def gold():
    data = {
        'date': [],
        'price': [],
    }
    for year in YEARS:
        year_text = read_year(year)
        dates, prices = parse_year(year_text)
        data['date'].extend(dates)
        data['price'].extend(prices)

    return pd.DataFrame(
        data['price'],
        columns=['price'],
        index=pd.DatetimeIndex(data['date']),
    )
