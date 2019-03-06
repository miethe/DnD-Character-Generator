import random

class Dice_Roller:

    def roll_n_d_x(self, count, die_size):
        rolls = []
        for _ in range(count):
            rolls.append(random.randrange(1,die_size))
        return rolls

    def drop_low_and_tally(self, rolls):
        if not rolls:
            return rolls
        rolls.sort()
        return sum(rolls[1:])

    def roll_ndx_y_times_drop_lowest(self, times_rolled, die_size, total_iterations):
        total_rolls = []
        if times_rolled is 0 or die_size is 0 or total_iterations is 0:
            return total_rolls
        for _ in range(total_iterations):
            rolls = self.roll_n_d_x(times_rolled, die_size)
            total_rolls.append(self.drop_low_and_tally(rolls))
        total_rolls.sort()
        return total_rolls[1:]

if __name__ == "__main__":
    print(Dice_Roller().roll_ndx_y_times_drop_lowest(4, 6, 7))
