from vars import *

print("Current day is ", current_day)
print("Start price on day ", current_day, " is £", round(start_price,2))
print("End price on day ", current_day + x_a[-1], " is £", round(end_price, 2))
print("Gain/loss for last ", x_a[-1], " days is £", round(end_price - start_price, 2))

if 0.1 < volatility < 0.5:
    print("Volatility medium")
if volatility == 0.1:
    print("Volatility low")
if volatility >= 0.5:
    print("Volatility high")

if trend > 0.5:
    print("Trend up")
if trend < 0.5:
    print("Trend down")
if trend == 0.5:
    print("Trend neutral")