import random

def monty_hall_n_doors(num_doors=100, num_simulations=10000):
    stick_wins = 0
    switch_wins = 0

    for _ in range(num_simulations):
        # Randomly place the car behind one of the N doors
        car_behind = random.randint(0, num_doors - 1)
        # Contestant's initial choice
        contestant_choice = random.randint(0, num_doors - 1)

        # Host opens N-2 doors, all goats, excluding contestant's choice and the car
        remaining_doors = [door for door in range(num_doors) if door != contestant_choice and door != car_behind]
        host_opens = random.sample(remaining_doors, num_doors - 2)

        # Determine the door to switch to (the only remaining unopened door)
        switch_to = [door for door in range(num_doors) if door != contestant_choice and door not in host_opens][0]

        # Check if sticking wins
        if contestant_choice == car_behind:
            stick_wins += 1
        # Check if switching wins
        if switch_to == car_behind:
            switch_wins += 1

    return stick_wins, switch_wins

# Run the simulation for 100 doors
stick_wins, switch_wins = monty_hall_n_doors(num_doors=100)

# Calculate win percentages
stick_win_percentage = (stick_wins / 10000) * 100
switch_win_percentage = (switch_wins / 10000) * 100

stick_win_percentage, switch_win_percentage

stick, switch = [],[]
for i in range(3,100):
    stick_wins, switch_wins = monty_hall_n_doors(num_doors=i)
    stick_win_percentage = (stick_wins / 10000) * 100
    switch_win_percentage = (switch_wins / 10000) * 100
    stick.append(stick_win_percentage)
    switch.append(switch_win_percentage)
  # print(f"{i} doors: {stick_win_percentage:.2f}% stick, {switch_win_percentage:.2f}% switch")


import matplotlib.pyplot as plt
plt.plot([i for i in range(3,100)], stick)
plt.plot([i for i in range(3,100)], switch)
plt.show()
