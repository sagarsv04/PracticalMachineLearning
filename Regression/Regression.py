
import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')

impdf = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

impdf['HL_PCT'] = (impdf['Adj. High'] - impdf['Adj. Close']) / impdf['Adj. Close'] * 100.0

impdf['PCT_Change'] = (impdf['Adj. Close'] - impdf['Adj. Open']) / impdf['Adj. Open'] * 100.0

impdf = impdf[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecastCol = 'Adj. Close'

impdf.fillna(-99999, inplace = True)

forecastOut = int(math.ceil(0.01*len(impdf)))

impdf['label'] = impdf[forecastCol].shift(-forecastOut)

impdf.dropna(inplace=True)
