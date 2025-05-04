import re
import numpy as np

def parse_fjs_to_pyjobshop_format(path, delta=50, max_modes_per_task=1):
    with open(path, "r") as f:
        header = re.findall(r'\S+', f.readline())
        num_jobs, num_machines, _ = map(int, header)

        data = []
        job_deadlines = {}

        for job_idx in range(num_jobs):
            tokens = re.findall(r'\S+', f.readline())
            i = 1  # skip the number of tasks
            job_data = []
            total_max_duration = 0

            while i < len(tokens):
                num_options = int(tokens[i])
                task_options = []
                max_duration = 0

                for k in range(min(num_options, max_modes_per_task)):
                    m_idx = i + 1 + 2 * k
                    d_idx = i + 2 + 2 * k
                    if d_idx >= len(tokens):  # safety check
                        break
                    machine_id = int(tokens[m_idx]) - 1
                    duration = int(tokens[d_idx])
                    task_options.append((duration, machine_id))
                    max_duration = max(max_duration, duration)

                job_data.append(task_options)
                total_max_duration += max_duration
                i += 1 + 2 * num_options

            data.append(job_data)
            job_deadlines[job_idx] = total_max_duration + delta

    return num_machines, data, job_deadlines
