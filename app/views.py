from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter

import pandas as pd
import numpy as np
import json

import yfinance as yf
import datetime as dt
import qrcode

from .models import Project

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm




# The Home page when Server loads up
def index(request):
    # ================================================= Left Card Plot =========================================================
    # Here we use yf.download function 
    data = yf.download(
        
        # passes the ticker
        tickers=['RELIANCE.NS', 'SBIN.NS', 'TATASTEEL.NS', 'BPCL.NS', 'TCS.NS', 'ITC.NS'],
        
        group_by = 'ticker',
         
        threads=True, # Set thread value to true
        
        # used for access data[ticker]
        period='1mo', 
        interval='1d'
    
    )

    data.reset_index(level=0, inplace=True)



    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['RELIANCE.NS']['Adj Close'], name="RELIANCE.NS")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['SBIN.NS']['Adj Close'], name="SBIN.NS")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['TATASTEEL.NS']['Adj Close'], name="TATASTEEL.NS")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['BPCL.NS']['Adj Close'], name="BPCL.NS")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['TCS.NS']['Adj Close'], name="TCS.NS")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['ITC.NS']['Adj Close'], name="ITC.NS")
            )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')


    # ================================================ To show recent stocks ==============================================
    
    df1 = yf.download(tickers = 'RELIANCE.NS', period='1d', interval='1d')
    df2 = yf.download(tickers = 'SBIN.NS', period='1d', interval='1d')
    df3 = yf.download(tickers = 'BAJAJFINSV.NS', period='1d', interval='1d')
    df4 = yf.download(tickers = 'ADANIENT.NS', period='1d', interval='1d')
    df5 = yf.download(tickers = 'BANKINDIA.NS', period='1d', interval='1d')
    df6 = yf.download(tickers = 'ASIANPAINT.NS', period='1d', interval='1d')

    df1.insert(0, "Ticker", "RELIANCE.NS")
    df2.insert(0, "Ticker", "SBIN.NS")
    df3.insert(0, "Ticker", "BAJAJFINSV.NS")
    df4.insert(0, "Ticker", "ADANIENT.NS")
    df5.insert(0, "Ticker", "BANKINDIA.NS")
    df6.insert(0, "Ticker", "ASIANPAINT.NS")

    df = pd.concat([df1, df2, df3, df4, df5, df6], axis=0)
    df.reset_index(level=0, inplace=True)
    df.columns = ['Date', 'Ticker', 'Op`en', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    df = df.astype(convert_dict)
    df.drop('Date', axis=1, inplace=True)

    json_records = df.reset_index().to_json(orient ='records')
    recent_stocks = []
    recent_stocks = json.loads(json_records)

    # ========================================== Page Render section =====================================================

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })

def search(request):
    return render(request, 'search.html', {})

def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('E:\\Stock-Prediction-System-Application-main\\app\\Data\\Tickers2.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })


# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value, period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        return render(request, 'API_Down.html', {})

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        return render(request, 'Invalid_Days_Format.html', {})

    Valid_Ticker = ["360ONE.NS","3MINDIA.NS","ABB.NS","ACC.NS","AIAENG.NS","APLAPOLLO.NS","AUBANK.NS","AARTIDRUGS.NS","AARTIIND.NS","AAVAS.NS","ABBOTINDIA.NS","ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS","ADANIPOWER.NS","ATGL.NS","ADANITRANS.NS","AWL.NS","ABCAPITAL.NS","ABFRL.NS","AEGISCHEM.NS","AETHER.NS","AFFLE.NS","AJANTPHARM.NS","APLLTD.NS","ALKEM.NS","ALKYLAMINE.NS","AMARAJABAT.NS","AMBER.NS","AMBUJACEM.NS","ANGELONE.NS","ANURAS.NS","APARINDS.NS","APOLLOHOSP.NS","APOLLOTYRE.NS","APTUS.NS","ACI.NS","ASAHIINDIA.NS","ASHOKLEY.NS","ASIANPAINT.NS","ASTERDM.NS","ASTRAL.NS","ATUL.NS","AUROPHARMA.NS","AVANTIFEED.NS","DMART.NS","AXISBANK.NS","BASF.NS","BEML.NS","BLS.NS","BSE.NS","BAJAJ-AUTO.NS","BAJAJELEC.NS","BAJFINANCE.NS","BAJAJFINSV.NS","BAJAJHLDNG.NS","BALAMINES.NS","BALKRISIND.NS","BALRAMCHIN.NS","BANDHANBNK.NS","BANKBARODA.NS","BANKINDIA.NS","MAHABANK.NS","BATAINDIA.NS","BAYERCROP.NS","BERGEPAINT.NS","BDL.NS","BEL.NS","BHARATFORG.NS","BHEL.NS","BPCL.NS","BHARTIARTL.NS","BIKAJI.NS","BIOCON.NS","BIRLACORPN.NS","BSOFT.NS","BLUEDART.NS","BLUESTARCO.NS","BBTC.NS","BORORENEW.NS","BOSCHLTD.NS","BRIGADE.NS","BCG.NS","BRITANNIA.NS","MAPMYINDIA.NS","CCL.NS","CESC.NS","CGPOWER.NS","CRISIL.NS","CSBBANK.NS","CAMPUS.NS","CANFINHOME.NS","CANBK.NS","CGCL.NS","CARBORUNIV.NS","CASTROLIND.NS","CEATLTD.NS","CENTRALBK.NS","CDSL.NS","CENTURYPLY.NS","CENTURYTEX.NS","CERA.NS","CHALET.NS","CHAMBLFERT.NS","CHEMPLASTS.NS","CHOLAHLDNG.NS","CHOLAFIN.NS","CIPLA.NS","CUB.NS","CLEAN.NS","COALINDIA.NS","COCHINSHIP.NS","COFORGE.NS","COLPAL.NS","CAMS.NS","CONCOR.NS","COROMANDEL.NS","CRAFTSMAN.NS","CREDITACC.NS","CROMPTON.NS","CUMMINSIND.NS","CYIENT.NS","DCMSHRIRAM.NS","DLF.NS","DABUR.NS","DALBHARAT.NS","DATAPATTNS.NS","DEEPAKFERT.NS","DEEPAKNTR.NS","DELHIVERY.NS","DELTACORP.NS","DEVYANI.NS","DIVISLAB.NS","DIXON.NS","LALPATHLAB.NS","DRREDDY.NS","EIDPARRY.NS","EIHOTEL.NS","EPL.NS","EASEMYTRIP.NS","EDELWEISS.NS","EICHERMOT.NS","ELGIEQUIP.NS","EMAMILTD.NS","ENDURANCE.NS","ENGINERSIN.NS","EQUITASBNK.NS","ESCORTS.NS","EXIDEIND.NS","FDC.NS","NYKAA.NS","FEDERALBNK.NS","FACT.NS","FINEORG.NS","FINCABLES.NS","FINPIPE.NS","FSL.NS","FIVESTAR.NS","FORTIS.NS","GRINFRA.NS","GAIL.NS","GMMPFAUDLR.NS","GMRINFRA.NS","GALAXYSURF.NS","GARFIBRES.NS","GICRE.NS","GLAND.NS","GLAXO.NS","GLENMARK.NS","MEDANTA.NS","GOCOLORS.NS","GODFRYPHLP.NS","GODREJAGRO.NS","GODREJCP.NS","GODREJIND.NS","GODREJPROP.NS","GRANULES.NS","GRAPHITE.NS","GRASIM.NS","GESHIP.NS","GREENPANEL.NS","GRINDWELL.NS","GUJALKALI.NS","GAEL.NS","FLUOROCHEM.NS","GUJGASLTD.NS","GNFC.NS","GPPL.NS","GSFC.NS","GSPL.NS","HEG.NS","HCLTECH.NS","HDFCAMC.NS","HDFCBANK.NS","HDFCLIFE.NS","HFCL.NS","HLEGLAS.NS","HAPPSTMNDS.NS","HAVELLS.NS","HEROMOTOCO.NS","HIKAL.NS","HINDALCO.NS","HGS.NS","HAL.NS","HINDCOPPER.NS","HINDPETRO.NS","HINDUNILVR.NS","HINDZINC.NS","POWERINDIA.NS","HOMEFIRST.NS","HONAUT.NS","HUDCO.NS","HDFC.NS","ICICIBANK.NS","ICICIGI.NS","ICICIPRULI.NS","ISEC.NS","IDBI.NS","IDFCFIRSTB.NS","IDFC.NS","IFBIND.NS","IIFL.NS","IRB.NS","ITC.NS","ITI.NS","INDIACEM.NS","IBULHSGFIN.NS","IBREALEST.NS","INDIAMART.NS","INDIANB.NS","IEX.NS","INDHOTEL.NS","IOC.NS","IOB.NS","IRCTC.NS","IRFC.NS","INDIGOPNTS.NS","IGL.NS","INDUSTOWER.NS","INDUSINDBK.NS","INFIBEAM.NS","NAUKRI.NS","INFY.NS","INGERRAND.NS","INTELLECT.NS","INDIGO.NS","IPCALAB.NS","JBCHEPHARM.NS","JKCEMENT.NS","JBMA.NS","JKLAKSHMI.NS","JKPAPER.NS","JMFINANCIL.NS","JSWENERGY.NS","JSWSTEEL.NS","JAMNAAUTO.NS","JSL.NS","JINDALSTEL.NS","JINDWORLD.NS","JUBLFOOD.NS","JUBLINGREA.NS","JUBLPHARMA.NS","JUSTDIAL.NS","JYOTHYLAB.NS","KPRMILL.NS","KEI.NS","KNRCON.NS","KPITTECH.NS","KRBL.NS","KSB.NS","KAJARIACER.NS","KALPATPOWR.NS","KALYANKJIL.NS","KANSAINER.NS","KARURVYSYA.NS","KEC.NS","KENNAMET.NS","RUSTOMJEE.NS","KFINTECH.NS","KOTAKBANK.NS","KIMS.NS","L&TFH.NS","LTTS.NS","LICHSGFIN.NS","LTIM.NS","LAXMIMACH.NS","LT.NS","LATENTVIEW.NS","LAURUSLABS.NS","LXCHEM.NS","LEMONTREE.NS","LICI.NS","LINDEINDIA.NS","LUPIN.NS","LUXIND.NS","MMTC.NS","MRF.NS","MTARTECH.NS","LODHA.NS","MGL.NS","M&MFIN.NS","M&M.NS","MAHINDCIE.NS","MHRIL.NS","MAHLIFE.NS","MAHLOG.NS","MANAPPURAM.NS","MRPL.NS","MARICO.NS","MARUTI.NS","MASTEK.NS","MFSL.NS","MAXHEALTH.NS","MAZDOCK.NS","MEDPLUS.NS","MFL.NS","METROBRAND.NS","METROPOLIS.NS","MSUMI.NS","MOTILALOFS.NS","MPHASIS.NS","MCX.NS","MUTHOOTFIN.NS","NATCOPHARM.NS","NBCC.NS","NCC.NS","NHPC.NS","NLCINDIA.NS","NMDC.NS","NOCIL.NS","NTPC.NS","NH.NS","NATIONALUM.NS","NAVINFLUOR.NS","NAZARA.NS","NESTLEIND.NS","NETWORK18.NS","NAM-INDIA.NS","NUVOCO.NS","OBEROIRLTY.NS","ONGC.NS","OIL.NS","OLECTRA.NS","PAYTM.NS","OFSS.NS","ORIENTELEC.NS","POLICYBZR.NS","PCBL.NS","PIIND.NS","PNBHOUSING.NS","PNCINFRA.NS","PVR.NS","PAGEIND.NS","PATANJALI.NS","PERSISTENT.NS","PETRONET.NS","PFIZER.NS","PHOENIXLTD.NS","PIDILITIND.NS","PEL.NS","PPLPHARMA.NS","POLYMED.NS","POLYCAB.NS","POLYPLEX.NS","POONAWALLA.NS","PFC.NS","POWERGRID.NS","PRAJIND.NS","PRESTIGE.NS","PRINCEPIPE.NS","PRSMJOHNSN.NS","PGHH.NS","PNB.NS","QUESS.NS","RBLBANK.NS","RECLTD.NS","RHIM.NS","RITES.NS","RADICO.NS","RVNL.NS","RAIN.NS","RAINBOW.NS","RAJESHEXPO.NS","RALLIS.NS","RCF.NS","RATNAMANI.NS","RTNINDIA.NS","RAYMOND.NS","REDINGTON.NS","RELAXO.NS","RELIANCE.NS","RBA.NS","ROSSARI.NS","ROUTE.NS","SBICARD.NS","SBILIFE.NS","SJVN.NS","SKFINDIA.NS","SRF.NS","MOTHERSON.NS","SANOFI.NS","SAPPHIRE.NS","SCHAEFFLER.NS","SHARDACROP.NS","SHOPERSTOP.NS","SHREECEM.NS","RENUKA.NS","SHRIRAMFIN.NS","SHYAMMETL.NS","SIEMENS.NS","SOBHA.NS","SOLARINDS.NS","SONACOMS.NS","SONATSOFTW.NS","STARHEALTH.NS","SBIN.NS","SAIL.NS","SWSOLAR.NS","STLTECH.NS","SUMICHEM.NS","SPARC.NS","SUNPHARMA.NS","SUNTV.NS","SUNDARMFIN.NS","SUNDRMFAST.NS","SUNTECK.NS","SUPRAJIT.NS","SUPREMEIND.NS","SUVENPHAR.NS","SUZLON.NS","SWANENERGY.NS","SYNGENE.NS","TCIEXP.NS","TCNSBRANDS.NS","TTKPRESTIG.NS","TV18BRDCST.NS","TVSMOTOR.NS","TMB.NS","TANLA.NS","TATACHEM.NS","TATACOMM.NS","TCS.NS","TATACONSUM.NS","TATAELXSI.NS","TATAINVEST.NS","TATAMTRDVR.NS","TATAMOTORS.NS","TATAPOWER.NS","TATASTEEL.NS","TTML.NS","TEAMLEASE.NS","TECHM.NS","TEJASNET.NS","NIACL.NS","RAMCOCEM.NS","THERMAX.NS","TIMKEN.NS","TITAN.NS","TORNTPHARM.NS","TORNTPOWER.NS","TCI.NS","TRENT.NS","TRIDENT.NS","TRIVENI.NS","TRITURBINE.NS","TIINDIA.NS","UCOBANK.NS","UFLEX.NS","UNOMINDA.NS","UPL.NS","UTIAMC.NS","ULTRACEMCO.NS","UNIONBANK.NS","UBL.NS","MCDOWELL-N.NS","VGUARD.NS","VMART.NS","VIPIND.NS","VAIBHAVGBL.NS","VTL.NS","VARROC.NS","VBL.NS","MANYAVAR.NS","VEDL.NS","VIJAYA.NS","VINATIORGA.NS","IDEA.NS","VOLTAS.NS","WELCORP.NS","WELSPUNIND.NS","WESTLIFE.NS","WHIRLPOOL.NS","WIPRO.NS","YESBANK.NS","ZFCVINDIA.NS","ZEEL.NS","ZENSARTECH.NS","ZOMATO.NS","ZYDUSLIFE.NS","ZYDUSWELL.NS","ECLERX.NS"]

    if ticker_value not in Valid_Ticker:
        return render(request, 'Invalid_Ticker.html', {})
    
    if number_of_days < 0:
        return render(request, 'Negative_Days.html', {})
    
    if number_of_days > 365:
        return render(request, 'Overflow_days.html', {})
    

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (INR per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')



    # ========================================== Machine Learning ==========================================


    try:
        df_ml = yf.download(tickers = ticker_value, period='3mo', interval='1h')
    except:
        ticker_value = 'RELIANCE.NS'
        df_ml = yf.download(tickers = ticker_value, period='3mo', interval='1m')

    # Fetching ticker values from Yahoo Finance API 
    df_ml = df_ml[['Adj Close']]
    forecast_out = int(number_of_days)
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)
    # Splitting data for Test and Train
    X = np.array(df_ml.drop(['Prediction'],1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
    # Applying Linear Regression
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    # Prediction Score
    confidence = clf.score(X_test, y_test)
    # Predicting for 'n' days stock data
    forecast_prediction = clf.predict(X_forecast)
    forecast = forecast_prediction.tolist()


    # ========================================== Plotting predicted data ======================================


    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    # ========================================== Display Ticker Info ==========================================

    ticker = pd.read_csv('app/Data/Tickers2.csv')
    to_search = ticker_value
    ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap','Country','IPO_Year', 'Volume', 'Sector', 'Industry']
    for i in range(0,ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            Last_Sale = (ticker.Last_Sale[i])
            Net_Change = ticker.Net_Change[i]
            Percent_Change = ticker.Percent_Change[i]
            Market_Cap = ticker.Market_Cap[i]
            Country = ticker.Country[i]
            IPO_Year = ticker.IPO_Year[i]
            Volume = ticker.Volume[i]
            Sector = ticker.Sector[i]
            Industry = ticker.Industry[i]
            break

    # ========================================== Page Render section ==========================================
    

    return render(request, "result.html", context={ 'plot_div': plot_div, 
                                                    'confidence' : confidence,
                                                    'forecast': forecast,
                                                    'ticker_value':ticker_value,
                                                    'number_of_days':number_of_days,
                                                    'plot_div_pred':plot_div_pred,
                                                    'Symbol':Symbol,
                                                    'Name':Name,
                                                    'Last_Sale':Last_Sale,
                                                    'Net_Change':Net_Change,
                                                    'Percent_Change':Percent_Change,
                                                    'Market_Cap':Market_Cap,
                                                    'Country':Country,
                                                    'IPO_Year':IPO_Year,
                                                    'Volume':Volume,
                                                    'Sector':Sector,
                                                    'Industry':Industry,
                                                    })
 