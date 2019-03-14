import random

class Dice_Roller:

    def roll_n_d_x(self, n_count, x_die_size):
        rolls = []
        for _ in range(n_count):
            #randrange is [x,y)
            rolls.append(random.randrange(1,x_die_size+1))
        return rolls

    def drop_low_and_tally(self, rolls):
        if not rolls:
            return rolls
        rolls.sort()
        return sum(rolls[1:])

    def roll_ndx_y_times(self, times_rolled, die_size, total_iterations, drop_lowest = False):
        total_rolls = []
        if times_rolled is 0 or die_size is 0 or total_iterations is 0:
            return total_rolls
        for _ in range(total_iterations):
            rolls = self.roll_n_d_x(times_rolled, die_size)
            total_rolls.append(self.drop_low_and_tally(rolls))
        if drop_lowest:
            total_rolls.sort()
            return total_rolls[1:]
        else:
            return total_rolls

def _average(lst):
    return sum(lst) / len(lst)

if __name__ == "__main__":
    print(_average(Dice_Roller().roll_ndx_y_times(4,6, 1000)))
    #print(Dice_Roller().roll_ndx_y_times_drop_lowest(4, 6, 7))
