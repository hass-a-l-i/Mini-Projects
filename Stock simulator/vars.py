start_price = 50

"volatility => max 1"
volatility = 0.1

"trend => higher = lower trend"
t = 5
trend = t / 10

x_a = [0]
y_a = [start_price]

"update each time"
current_day = 0

"days"
days = 50

import loop

end_price = y_a[-1]
