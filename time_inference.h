#include <chrono>
#include <vector>
#include <stdio.h>

class TimeInference
{
    private:

    bool remove_first_call;

    enum ProcessTrackingStatus
    {
        STARTED = 0, ENDED = 1
    };

    struct ProcessTracking
    {
        std::string process_name;
        size_t calls_number = 0;
        float time_taken = 0.f;
        std::chrono::steady_clock::time_point last_start;
        ProcessTrackingStatus status = ProcessTrackingStatus::STARTED; 
    };

    std::vector<ProcessTracking*> trackings;

    public:

    TimeInference(bool remove_first_call = true):remove_first_call(remove_first_call){}
    
    ~TimeInference()
    {
        float total_time_taken_avg = 0.f;
        for(auto &P : trackings)
        {
            int valid_calls_number = P->calls_number - remove_first_call;
            if(valid_calls_number == 0)
            {
                valid_calls_number += 1;
            }

            float time_taken_avg = P->time_taken / (float)valid_calls_number;

            std::cout << "Process '" << P->process_name << "'";
            std::cout << " took " << time_taken_avg << " ms [" << valid_calls_number << " calls]\n";

            total_time_taken_avg += time_taken_avg;
        }

        std::cout << "In total, all processes took: ";
        std::cout << total_time_taken_avg <<" ms\n";
    }

    void track(std::string process_name)
    {
        auto curr_time = std::chrono::steady_clock::now();

        bool is_process_found = false;
        ProcessTracking *process;
        for(auto &P : trackings)
        {
            if(P->process_name == process_name) 
            { 
                process = P;
                is_process_found = true;
                break;
            }
        }

        if(is_process_found)
        {
            if(process->status == ProcessTrackingStatus::STARTED)
            {
                process->calls_number += 1;
                process->status = ProcessTrackingStatus::ENDED;

                if(remove_first_call && process->calls_number == 1)
                    return;

                auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - process->last_start).count();
                process->time_taken += time_taken;
            }
            else
            {
                process->last_start = curr_time;
                process->status = ProcessTrackingStatus::STARTED;                
            }
        }
        else
        {
            process = new ProcessTracking;
            process->process_name = process_name;
            process->last_start = curr_time;
            process->status = ProcessTrackingStatus::STARTED;
            trackings.push_back(process);
        }
    }
};