
import pandas_datareader as pdr
import streamlit as st
import pandas as pd
import numpy as np
import time
import yfinance as yf
from PIL import Image
import numpy.core.multiarray
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

ticker = ['AAPL', 'MSFT', 'AMZN', 'NFLX', 'ZM',
          'TWTR', '^GSPC', 'IBB', 'FB', 'MSCI', '^IXIC']
start = dt.datetime(2020, 1, 1)
st.set_option('deprecation.showPyplotGlobalUse', False)


def portfolio(ticker, no_months, no_simulations, amount):
    # p = pdr.get_data_yahoo(ticker, start_date)['Adj Close']
    por = yf.Tickers(ticker)
    por = por.history(period=f'{no_months}mo')
    p = por['Close']
    log_rtn = np.log(p/p.shift())
    n = no_simulations
    u = len(p.columns)
    weights = np.zeros((n, u))
    exp_rtns = np.zeros(n)
    exp_vols = np.zeros(n)
    sharp_ratios = np.zeros(n)
    lt = ([])
    for i in range(n):
        weight = np.random.random(u)
        weight /= weight.sum()
        weights[i] = weight
        exp_rtns[i] = np.sum(log_rtn.mean()*weight)*252
        exp_vols[i] = np.sqrt(
            np.dot(weight.T, np.dot(log_rtn.cov()*252, weight)))
        sharp_ratios[i] = exp_rtns[i]/exp_vols[i]

    for t in range(u):
        y = [p.columns[t], (weights[sharp_ratios.argmax()][t])
             * 100, (weights[sharp_ratios.argmax()][t])*amount]

        lt.append(y)

    # lt.append([sharp_ratios.max(), exp_rtns.max()])

    fig, ax = plt.subplots()
    ax.scatter(exp_vols, exp_rtns, c=sharp_ratios)
    ax.scatter(exp_vols[sharp_ratios.argmax()],
               exp_rtns[sharp_ratios.argmax()], c='r')
    ax.set_xlabel('Expected Volatitiy/Risk')
    ax.set_ylabel('Expected Portifolio Returns')
    ax.set_title('My Portifolio', c='blue')
    st.pyplot()

    return lt


def main():

    st.title("Welcome To my Portfolio App")
    img = Image.open('portfolio.jpg')
    st.image(img)
    st.header('Predict the percentage of investment in your Portifolio')
    tkr = st.multiselect('Choose Company Ticker Making your Portfolio', ticker)
    amnt = int(st.number_input('Amount to Invest in your Portfolio'))
    # start_date = st.date_input('Start Date')
    months = int(st.number_input('Number of Months'))
    simulations = int(st.number_input('Number of simulations'))

    if st.button('Predict'):
        progress = st.progress(0)
        for p in range(100):
            time.sleep(0.1)
            progress.progress(p+1)

        lt = portfolio(ticker=tkr, no_months=months,
                       no_simulations=simulations, amount=amnt)

        for i in lt:
            st.subheader(
                f'Percentage to invest in {i[0]} is {round(i[1],2)}% - Ksh {round(i[2],2)}/=')

        st.balloons()


if __name__ == "__main__":
    main()
