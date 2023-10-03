""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Progress bar class. Store counter (int) of parsing elements and number of done items. Based on them compute percentage
of done work and print progress bar in stdout.
####################################################################################################################"""
import sys

class PBar:

    def __init__(self, all_items=0, print_by=1000):
        """Progress bar parameters"""
        self.progress_done = 0
        self.progress_all = all_items
        self.bar_size = 80
        self.print_by = print_by

    def increase_done(self, by=1):
        self.progress_done += by
        return self.print_progress()

    def flush_done(self, new_value):
        self.progress_done = new_value
        return self.print_progress()

    def flush_progress(self, new_all_items):
        self.progress_all = new_all_items

    def print_progress(self, pass_check=False):
        if pass_check or (self.progress_done % self.print_by) == 0:
            done_prc = self.progress_done / self.progress_all
            hs_size = int(done_prc * self.bar_size)
            sys.stdout.write(
                "\rProgress: |{}{}| {:.2f}% Done.".format("#" * hs_size, "-" * (self.bar_size - hs_size),
                                                          done_prc * 100.0))
            sys.stdout.flush()
            return True

        return False
