#!/bin/bash

# Script to run bokeh_viz/bokeh_visualizations.py for all 6 tasks

# Create task array
TASKS=(
    "vision/classification/ImageNet/viz"
    "vision/detection/COCO/viz"
    "language/question_answering/MMLU/viz"
    "language/code_generation/LiveCodeBench/viz"
    "action/motion_prediction/Waymo/viz"
    "action/motion_planning/NAVSIM/viz"
)

# Process each task
for TASK_PATH in "${TASKS[@]}"; do
    # Extract components from task path
    TYPE=$(echo "$TASK_PATH" | cut -d'/' -f1)
    TASK=$(echo "$TASK_PATH" | cut -d'/' -f2)
    DATASET=$(echo "$TASK_PATH" | cut -d'/' -f3)
    SPLIT=$(echo "$TASK_PATH" | cut -d'/' -f4)
    
    echo "========================================="
    echo "Processing task: $TASK_PATH"
    echo "Type: $TYPE, Task: $TASK, Dataset: $DATASET, Split: $SPLIT"
    
    # Set file paths
    TEST_CASE_RATINGS="/home/ziggy/AGI-Elo/bokeh_viz/elo_results/${TYPE}/${TASK}/${DATASET}/${SPLIT}/rating_Glicko_test_case_1.pkl"
    MODEL_RATINGS="/home/ziggy/AGI-Elo/bokeh_viz/elo_results/${TYPE}/${TASK}/${DATASET}/${SPLIT}/rating_Glicko_model_1.pkl"
    MATCH_RESULTS="/home/ziggy/AGI-Elo/bokeh_viz/elo_results/${TYPE}/${TASK}/${DATASET}/${SPLIT}/matches"
    OUTPUT_DIR="/home/ziggy/AGI-Elo/results_html/${TYPE}/${TASK}"
    
    # Print information
    echo "Running Bokeh visualization:"
    echo "  Test case ratings: $TEST_CASE_RATINGS"
    echo "  Model ratings: $MODEL_RATINGS"
    echo "  Match results: $MATCH_RESULTS"
    echo "  Output directory: $OUTPUT_DIR"
    echo ""
    
    # Run Python script
    python bokeh_visualizations.py \
        --test_case_ratings "$TEST_CASE_RATINGS" \
        --model_ratings "$MODEL_RATINGS" \
        --match_results "$MATCH_RESULTS" \
        --output_dir "$OUTPUT_DIR"
    
    # Check if execution was successful
    if [ $? -eq 0 ]; then
        echo "✅ Visualization for $TASK_PATH successfully created in $OUTPUT_DIR directory!"
    else
        echo "❌ Error: Visualization generation failed for $TASK_PATH."
    fi
    echo "========================================="
    echo ""
done

echo "All tasks completed!"
