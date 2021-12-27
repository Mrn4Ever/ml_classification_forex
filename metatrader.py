import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz


#Get Price Data from MetaTrader5 (Symbol, Timeframe)
def getPrices(symbol, timeframe):
    if not mt5.initialize():
        print("[Metatrader5 initialization faild !]=",mt5.last_error())
        return pd.Dataframe()
    print ('[MetaTrader5]-> Initialization passed!')    
    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime(2000, 1, 1, tzinfo=timezone)
    utc_to   = datetime.now()
    rates    = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    print (f'[MetaTrader5]-> Symbol {symbol} [{str(timeframe)}]')  
    print (f'[MetaTrader5]-> from {utc_from:%Y-%m-%d} to {utc_to:%Y-%m-%d}')   
    print (f'[MetaTrader5]-> Loaded data length: {len(rates)}')
    mt5.shutdown()

    df = pd.DataFrame(rates)   
    df['time']=pd.to_datetime(df['time'], unit='s')
    df = df.set_index('time')
    df = df.rename(columns={"tick_volume": "volume"})
    return df[['open','high','low','close','volume']]


symbol          = 'EURUSD'
timeframe       = mt5.TIMEFRAME_H1
fileName        = f'./data/{symbol}-{str(timeframe)}.csv'

df              = getPrices(symbol=symbol, timeframe=timeframe)
df.to_csv(fileName)