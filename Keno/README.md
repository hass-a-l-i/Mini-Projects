# Keno Game Analysis and Report

---

## 1. Simulation Program

Write a program in any language you prefer that can simulate this game, with the following parameters and output requirements:

1. **Game setup**: Total number is 80 and Number of draws is 20.  
2. **Selections**: A list of your selected numbers (default length 10).  
3. **Result**: The program should output the 20 randomly drawn numbers and the total number of selections being matched.


## 2. Probability and Fair Odds

Compute the probability for each category of correct matches: **5, 6, 7, 8, 9, 10**.  
Determine what the fair decimal odds would be for each winning category independently.  


## 3. Gambling Decision

Suppose I am offering a Keno game with the following payout table (unit stake = £1):

- Matching 5: £3.00  
- Matching 6: £15.00  
- Matching 7: £100.00  
- Matching 8: £1,000.00  
- Matching 9: £25,000.00  
- Matching 10: £2,500,000.00  

If you were a gambler, would you play this Keno game, and why?  


## 4. Insurance Premium

Suppose you are a director of an insurance company with £20 million capital (Kelly bankroll).  
The game provider asks you to give an insurance quote for **matching 10 numbers** in the above Keno game.  

- If somebody wins by matching exactly 10 numbers, the insurance company must pay the full **£2.5 million**.  
- How much would you charge (insurance premium) to insure each **£1 entry**?

## 5. Expected Value Function

Compute the expected value \(a(i,j)\), using the above Keno pay-outs and unit stake settings.  

- \(i\): the number of draws that have already taken place.  
- \(j\): the total number of matched selections after the \(i\)-th draw.  
- \(a(i,j)\): the expected value at the \(i\)-th draw given that you currently have \(j\) matched selections.  

For simplicity, assume:  

7 <= i <= 19

7 <= j <= 9

## 6. Detecting Biased Draws

Imagine a Keno machine experiencing technical problems where some numbers are being selected with a slightly higher probability.

1. Describe how you can mathematically/statistically justify which number(s) are being overly selected.  
2. Write code to support your justification.  
