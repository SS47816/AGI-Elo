#!/bin/bash

# Script to run bokeh_viz/bokeh_visualizations.py for all 6 tasks

# 创建任务数组
TASKS=(
    "vision/classification/ImageNet/viz"
    "vision/detection/COCO/viz"
    "language/question_answering/MMLU/viz"
    "language/code_generation/LiveCodeBench/viz"
    "action/motion_prediction/Waymo/viz"
    "action/motion_planning/NAVSIM/viz"
)

# 循环处理每个任务
for TASK_PATH in "${TASKS[@]}"; do
    # 从任务路径中提取组件
    TYPE=$(echo "$TASK_PATH" | cut -d'/' -f1)
    TASK=$(echo "$TASK_PATH" | cut -d'/' -f2)
    DATASET=$(echo "$TASK_PATH" | cut -d'/' -f3)
    SPLIT=$(echo "$TASK_PATH" | cut -d'/' -f4)
    
    echo "========================================="
    echo "处理任务: $TASK_PATH"
    echo "类型: $TYPE, 任务: $TASK, 数据集: $DATASET, 分割: $SPLIT"
    
    # 设置文件路径
    TEST_CASE_RATINGS="/home/ziggy/AGI-Elo/bokeh_viz/elo_results/${TYPE}/${TASK}/${DATASET}/${SPLIT}/rating_Glicko_test_case_1.pkl"
    MODEL_RATINGS="/home/ziggy/AGI-Elo/bokeh_viz/elo_results/${TYPE}/${TASK}/${DATASET}/${SPLIT}/rating_Glicko_model_1.pkl"
    MATCH_RESULTS="/home/ziggy/AGI-Elo/bokeh_viz/elo_results/${TYPE}/${TASK}/${DATASET}/${SPLIT}/matches"
    OUTPUT_DIR="/home/ziggy/AGI-Elo/results_html/${TYPE}/${TASK}"
    
    # 打印信息
    echo "运行Bokeh可视化:"
    echo "  测试案例评级: $TEST_CASE_RATINGS"
    echo "  模型评级: $MODEL_RATINGS"
    echo "  比赛结果: $MATCH_RESULTS"
    echo "  输出目录: $OUTPUT_DIR"
    echo ""
    
    # 运行Python脚本
    python bokeh_visualizations.py \
        --test_case_ratings "$TEST_CASE_RATINGS" \
        --model_ratings "$MODEL_RATINGS" \
        --match_results "$MATCH_RESULTS" \
        --output_dir "$OUTPUT_DIR"
    
    # 检查是否成功执行
    if [ $? -eq 0 ]; then
        echo "✅ $TASK_PATH 的可视化已成功创建在 $OUTPUT_DIR 目录中!"
    else
        echo "❌ 错误: $TASK_PATH 的可视化生成失败。"
    fi
    echo "========================================="
    echo ""
done

echo "所有任务处理完成!"
