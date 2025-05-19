import subprocess

scripts = [
    "AGI_Elo/pipeline/classification.py",
    "AGI_Elo/pipeline/detection.py",
    "AGI_Elo/pipeline/question_answering.py",
    "AGI_Elo/pipeline/coding.py",
    "AGI_Elo/pipeline/motion_prediction.py",
    "AGI_Elo/pipeline/motion_planning.py",
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)

    # Print script output
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error in {script}:")
        print(result.stderr)
        break
