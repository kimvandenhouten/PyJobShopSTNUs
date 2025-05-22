import os
import random

def add_random_deadlines(file_path, output_path=None, min_deadline=1):

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Get horizon from metadata
    horizon = None
    for line in lines:
        if 'horizon' in line.lower():
            horizon = int(line.split(':')[-1].strip())
            break
    if horizon is None:
        raise ValueError(f"Horizon not found in {file_path}")

    # Extract job numbers from "PRECEDENCE RELATIONS"
    try:
        job_start = lines.index("PRECEDENCE RELATIONS:\n")
        req_start = lines.index("REQUESTS/DURATIONS:\n")
    except ValueError as e:
        raise ValueError(f"Missing section in {file_path}: {e}")

    precedence_lines = lines[job_start+2:req_start-1]
    job_numbers = []
    for line in precedence_lines:
        if line.strip() == '':
            continue
        parts = line.strip().split()
        jobnr = int(parts[0])
        if jobnr != 1 and jobnr != len(precedence_lines):  # skip supersource/sink
            job_numbers.append(jobnr)

    # Parse durations for each job from REQUESTS/DURATIONS section
    durations_by_job = {}
    for line in lines[req_start + 2:]:
        line = line.strip()
        if line == '' or line.startswith('*'):
            break
        parts = line.split()
        if len(parts) >= 3:
            job = int(parts[0])
            duration = int(parts[2])  # jobnr mode duration ...
            if job not in durations_by_job:
                durations_by_job[job] = []
            durations_by_job[job].append(duration)

    # Choose random subset of jobs to assign deadlines
    num_with_deadline = random.randint(1, len(job_numbers))
    jobs_with_deadline = random.sample(job_numbers, num_with_deadline)

    # Assign deadlines with minimum set to the job's shortest mode duration
    deadlines = {}
    for job in jobs_with_deadline:
        min_mode_duration = min(durations_by_job.get(job, [min_deadline]))
        min_valid_deadline = max(min_deadline, min_mode_duration)
        deadlines[job] = random.randint(min_valid_deadline, horizon)

    # Format DEADLINES section
    deadline_section = ["DEADLINES:\n", "jobnr.  deadline\n"]
    for job, deadline in sorted(deadlines.items()):
        deadline_section.append(f"  {job:<5}   {deadline}\n")
    deadline_section.append("************************************************************************\n")

    # Append and write file
    output_lines = lines + deadline_section
    if output_path is None:
        output_path = file_path.replace(".mm", "_with_deadlines.mm")
    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    print(f"✔ Deadlines added to {output_path}")

def process_all_mm_files(input_dir, output_dir=None, seed=42):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        random.seed(filename)
        if filename.endswith(".mm"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename) if output_dir else None
            try:
                add_random_deadlines(input_path, output_path=output_path)
            except Exception as e:
                print(f"⚠ Error processing {filename}: {e}")

# Example usage
process_all_mm_files("C:\\Users\\andre\\PyJobShopSTNUs\\rcpsp_max\\data\\j10_mm", output_dir="data\instances_with_deadlines")
