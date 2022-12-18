from time import time
from dataclasses import dataclass

@dataclass
class ProcessTracking:
    process_name: str
    last_start: float
    status: int = 0
    calls_number: int = 0
    time_taken: float = 0

class TimeInference(object):
    STARTED = 0
    ENDED = 1

    def __init__(self, remove_first_call = True):
        self.remove_first_call = remove_first_call
        self.trackings = []

    def __del__(self):

        total_time_taken_avg = 0.
        for P in self.trackings:
            valid_calls_number = P.calls_number - self.remove_first_call;
            if valid_calls_number == 0:
                valid_calls_number += 1

            time_taken_avg = P.time_taken / valid_calls_number

            print(f"Process '{P.process_name}' took {time_taken_avg} s [{valid_calls_number} calls]")

            total_time_taken_avg += time_taken_avg;

        print(f"In total, all processes took: {total_time_taken_avg} s")

    def track(self, process_name):
        curr_time = time()

        is_process_found = False
        process = None
        for P in self.trackings:
            if P.process_name == process_name:
                process = P
                is_process_found = True
                break

        if is_process_found:
            if(process.status == self.STARTED):

                process.calls_number += 1
                process.status = self.ENDED

                if self.remove_first_call and process.calls_number == 1:
                    return

                time_taken = curr_time - process.last_start
                process.time_taken += time_taken
            else:
                process.last_start = curr_time
                process.status = self.STARTED
        else:
            process = ProcessTracking(process_name, curr_time, self.STARTED)
            self.trackings.append(process)
