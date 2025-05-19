import subprocess

scripts = [
    "AGI-Elo/pipeline/classification.py",
    "AGI-Elo/pipeline/detection.py",
    "AGI-Elo/pipeline/question_answering.py",
    "AGI-Elo/pipeline/coding.py",
    "AGI-Elo/pipeline/motion_prediction.py",
    "AGI-Elo/pipeline/motion_planning.py",
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
