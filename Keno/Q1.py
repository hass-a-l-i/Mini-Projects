import numpy as np

keno_list = np.array(range(1, 81))
keno_card = keno_list.reshape(8, 10)


def check_int(usr_input):
    while True:
        try:
            return int(usr_input)
        except ValueError:
            print("Non-integer input. Please try again.")
            usr_input = input()


def choose_numbers(length):
    print("You will now choose", length, "unique integers between 1 and 80.")
    print("Please input each integer when prompted, then press Enter.")
    chosen_list = []
    for i in range(1, length + 1):
        print("Choose number", i, "of", length, ":")
        number = input()
        number = check_int(number)
        if 0 < number <= 80 and number not in chosen_list:
            chosen_list.append(number)
        else:
            while True:
                print("Input out of bounds or already chosen. Please try again.")
                number = input()
                number = check_int(number)
                if 0 < number <= 80 and number not in chosen_list:
                    chosen_list.append(number)
                    break
    print("Your chosen numbers are listed below.")
    print(chosen_list)
    return chosen_list


def keno_game():
    print("Keno Game Simulator")
    print("Please input the total number of selections you would like to make (from 1 - 20). "
          "Press Enter now for the default of 10.")
    length = input()
    if length == "":
        length = 10
    length = check_int(length)
    if 0 < length <= 20:
        pass
    else:
        while True:
            print("Input out of bounds. Please try again.")
            length = input()
            length = check_int(length)
            if 0 < length <= 20:
                break
    chosen_list = choose_numbers(length)
    rand_list = np.random.choice(keno_list, 20, replace=False)
    print("The output of this rounds draw is listed below:")
    print(list(rand_list))
    print("Matched numbers:", [i for i in chosen_list if i in rand_list])
    count = 0
    for i in chosen_list:
        if i in rand_list:
            count = count + 1
    print("Number of successful matches:", count)
    if count >= 5:
        print("Payout is expected as", count, "is at least 5.")
    else:
        print("Payout not expected as", count, "is less than 5.")


if __name__ == "__main__":
    keno_game()

